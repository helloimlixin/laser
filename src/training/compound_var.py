"""Audited CelebA-HQ compound-pair VAR experiment with a frozen tokenizer."""
from contextlib import nullcontext
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
from omegaconf import OmegaConf
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from src.models.compound_var import CompoundLaserVAR, compound_decompose
from src.models.scratch_var import build_scratch_tokenizer
from src.models.sparse_token_codec import token_temperatures, validate_tokenized_checkpoint
from src.models.rqvae.lpips import LPIPS
from src.original_rq_training import atomic_json, file_sha256
from src.training.var_laser import Experiment, HFSquareImages, optimizer_groups, save_checkpoint, restore_rng
from src.training.checkpoint_upload import CheckpointUploader
from src.training.wandb_checkpoints import publish_checkpoint_bundle
from src.training.distributed_failure import fatal_worker_error
from src.training.compound_resume import prepare_resume, epoch_fraction, schedule_position
from src.data.var_token_cache import VARTokenCache, restore_cached_codes
from utils.lr_control import lr_wd_annealing


class CompoundExperiment(Experiment):
    def __init__(self, cfg):
        self.cfg = cfg
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
        self.base = Path(cfg.output_dir)
        self.mode = cfg.compound.mode
        self.out = self.base / self.mode
        self.out.mkdir(parents=True, exist_ok=True)
        self.execution = cfg.get('execution', {})
        self.checkpoints = Path(self.execution.get('checkpoint_dir', str(self.out))) if self.mode == 'train' else self.out
        if self.mode == 'preflight':
            self.checkpoints = Path(self.execution.get('preflight_checkpoint_dir', str(self.out)))
        self.checkpoints.mkdir(parents=True, exist_ok=True)
        self.kind, self.initialization = 'laser', 'scratch'
        self.num_classes = 1 if str(cfg.data.get('dataset', '')).lower() == 'ffhq' else 2
        self.inception, self.run = None, None
        self.uploader = None
        self.stop = False
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, 'stop', True))
        signal.signal(signal.SIGINT, lambda *_: setattr(self, 'stop', True))
        self.train = HFSquareImages(cfg.data.root, 'train', True, cfg.seed, 256, resize_crop=False)
        self.val = HFSquareImages(cfg.data.root, 'validation', False, cfg.seed, 256, resize_crop=False)
        self.vae = build_scratch_tokenizer(cfg.model, cfg.seed)
        checkpoint = torch.load(cfg.compound.tokenizer_checkpoint, map_location='cpu', weights_only=False, mmap=True)
        validate_tokenized_checkpoint(self.vae.quantize, checkpoint)
        self.vae.load_state_dict(checkpoint['model'], strict=True)
        self.tokenizer_progress = checkpoint['progress']
        del checkpoint
        self.vae.to(self.device).requires_grad_(False).eval()
        sha = [file_sha256(cfg.compound.tokenizer_checkpoint) if self.rank == 0 else None]
        dist.broadcast_object_list(sha)
        self.tokenizer_sha256 = sha[0]
        self.atom_temperatures = self.coefficient_temperatures = None
        self.train_tokens = self.val_tokens = None
        self.token_cache_sha256 = None
        if self.mode != 'audit':
            calibration = json.loads((self.base / 'audit/calibration.json').read_text())
            if calibration['tokenizer_sha256'] != self.tokenizer_sha256:
                raise ValueError('Calibration belongs to another tokenizer')
            self.atom_temperatures = calibration['atom_temperatures']
            self.coefficient_temperatures = calibration['coefficient_temperatures']
            if self.vae.quantize.tokenized_sparse_policy is not None:
                expected = token_temperatures(self.vae.quantize)
                if (self.atom_temperatures, self.coefficient_temperatures) != expected:
                    raise ValueError('Stage-2 temperatures differ from the stage-1 sparse-token policy')

        cache_directory = self.execution.get('token_cache_dir')
        if cache_directory and self.mode in ('train', 'preflight'):
            self.train_tokens = VARTokenCache(cache_directory, 'train', cfg.seed)
            self.val_tokens = VARTokenCache(cache_directory, 'validation', cfg.seed)
            manifest = self.train_tokens.manifest
            if manifest['tokenizer_sha256'] != self.tokenizer_sha256:
                raise ValueError('Token cache belongs to another tokenizer')
            if manifest['calibration_sha256'] != file_sha256(self.base / 'audit/calibration.json'):
                raise ValueError('Token cache uses different stochastic calibration')
            for split, dataset in [('train', self.train), ('validation', self.val)]:
                if manifest['dataset_fingerprints'][split] != dataset.dataset._fingerprint:
                    raise ValueError('Token cache does not match the image dataset')
            if self.rank == 0:
                for name, record in manifest['files'].items():
                    if file_sha256(Path(cache_directory) / name) != record['sha256']:
                        raise ValueError(f'Corrupted token cache file: {name}')
            dist.barrier()
            self.token_cache_sha256 = file_sha256(Path(cache_directory) / 'manifest.json')
        if self.rank == 0:
            OmegaConf.save(cfg, self.out / 'resolved-config.yaml', resolve=True)
            atomic_json(self.out / 'initialization.json', dict(tokenizer_sha256=self.tokenizer_sha256,
                         tokenizer_progress=self.tokenizer_progress, prior='scratch', world_size=self.world,
                         spatial_positions=sum(p*p for p in cfg.model.patch_nums),
                         compound_formulation='p(atom | past_pairs, past_scales) p(coeff | atom, past_pairs, past_scales)',
                         coefficient_grid='unchanged per-scale 257-bin asinh tokenizer grid'))

    @torch.no_grad()
    def encode(self, images):
        with self.amp():
            return self.vae.quant_conv(self.vae.encoder(images.to(self.device))).float()

    def codes(self, latent, stochastic=False, generator=None):
        return compound_decompose(self.vae.quantize, latent, stochastic=stochastic, generator=generator,
                                  atom_temperatures=self.atom_temperatures,
                                  coefficient_temperatures=self.coefficient_temperatures)

    @torch.no_grad()
    def audit(self):
        if self.world != 1:
            raise ValueError('Calibration uses one GPU and fixed image subsets')
        q = self.vae.quantize
        batches = {}
        for name, dataset in [('train', self.train), ('validation', self.val)]:
            ids = np.random.default_rng(761).choice(len(dataset), 64, replace=False).tolist()
            images = torch.stack([dataset[i][0] for i in ids]).to(self.device)
            latent = torch.cat([self.encode(x) for x in images.split(8)])
            codes = self.codes(latent)
            batches[name] = dict(images=images, latent=latent, codes=codes, indices=ids)
        baseline = batches['train']['codes']
        offset, scales = 0, []
        for pn in q.v_patch_nums:
            values = baseline['physical_coefficients'][:, offset:offset + pn*pn]
            scales.append(float(values.square().mean().clamp_min(1e-5)))
            offset += pn*pn
        perceptual = LPIPS().eval().requires_grad_(False).to(self.device)

        def metrics(batch, codes):
            mse = float((codes['latent'] - batch['latent']).square().mean())
            lpips, pixel_mse = [], []
            for start in range(0, len(batch['images']), 8):
                with self.amp():
                    images = self.vae.fhat_to_img(codes['latent'][start:start+8])
                    lpips.append(perceptual(images, batch['images'][start:start+8]).float())
                pixel_mse.append(((images.float()-batch['images'][start:start+8])/2).square().mean())
            return dict(latent_mse=mse, lpips=float(torch.stack(lpips).mean()),
                        psnr=float(-10 * torch.stack(pixel_mse).mean().log10()),
                        clipping_fraction=float(codes['clip_fraction']))

        reference = {name: metrics(batch, batch['codes']) for name, batch in batches.items()}
        results = []
        policy = q.tokenized_sparse_policy
        for strength in ((None,) if policy is not None else (.0005, .002, .008, .032)):
            if policy is not None:
                self.atom_temperatures, self.coefficient_temperatures = token_temperatures(q)
            else:
                self.atom_temperatures = [s * strength for s in scales]
                self.coefficient_temperatures = [s * strength * .25 for s in scales]
            row = dict(strength=strength, atom_temperatures=self.atom_temperatures,
                       coefficient_temperatures=self.coefficient_temperatures)
            for name, batch in batches.items():
                codes = self.codes(batch['latent'], True, torch.Generator(device=self.device).manual_seed(9901))
                row[name] = metrics(batch, codes)
                row[name]['changed_support_fraction'] = float((codes['atoms'] != batch['codes']['atoms']).any(-1).float().mean())
                row[name]['relative_latent_mse'] = row[name]['latent_mse'] / reference[name]['latent_mse']
                row[name]['extra_lpips'] = row[name]['lpips'] - reference[name]['lpips']
                reconstructed, inputs = q.from_codes(codes['atoms'], codes['coefficients'])
                torch.testing.assert_close(inputs, codes['inputs'])
                torch.testing.assert_close(reconstructed, codes['latent'])
            row['accepted'] = (row['validation']['relative_latent_mse'] <= 1.05 and
                               row['validation']['extra_lpips'] <= .005)
            results.append(row)
            print(json.dumps(row), flush=True)
        accepted = [row for row in results if row['accepted']]
        if not accepted:
            raise RuntimeError('No stochastic calibration passed the reconstruction gate')
        selected = accepted[-1]
        atomic_json(self.out / 'calibration.json', dict(
            tokenizer_sha256=self.tokenizer_sha256, reference=reference, trials=results,
            atom_temperatures=selected['atom_temperatures'],
            coefficient_temperatures=selected['coefficient_temperatures'],
            tokenized_sparse_policy=policy,
            source=('Frozen stage-1 sparse-token policy; no post-training temperature change' if policy is not None else
                    'Church physical-distance soft coefficients and stochastic OMP; temperatures recalibrated for VAR'),
            stochastic_policy='online complete multiscale trajectories, recomputed after sampled earlier scales',
            dataset_indices={name: b['indices'] for name, b in batches.items()}))
        torch.save(dict(latent=batches['train']['latent'][:16].cpu(),
                        labels=torch.tensor([self.train[i][1] for i in batches['train']['indices'][:16]])),
                   self.out / 'preflight-latents.pt')
        self.log('audit_complete', selected_strength=selected['strength'],
                 support_changes=selected['validation']['changed_support_fraction'],
                 relative_latent_mse=selected['validation']['relative_latent_mse'])

    def sampling_options(self):
        options = OmegaConf.to_container(self.cfg.get('sampling', OmegaConf.create({})), resolve=True)
        allowed = {'atom_temperature', 'coefficient_temperature', 'coefficient_top_p'}
        if options.keys() - allowed:
            raise ValueError('Unknown unconditional sampling option')
        return options

    def fid_selection(self):
        evaluation = self.cfg.get('evaluation', {})
        protocol = evaluation.get('selection_protocol', 'heldout_uint8')
        if protocol == 'heldout_uint8':
            return None
        if protocol not in ('rq_train_50k', 'matched_train_50k'):
            raise ValueError('Unknown prior FID selection protocol')
        if int(evaluation.get('full_samples', 0)) != 50000:
            raise ValueError('RQ training-reference selection requires 50000 samples')
        if not evaluation.get('rq_fid_reference') or not evaluation.get('rq_fid_reference_sha256'):
            raise ValueError('RQ training-reference selection requires verified reference statistics')
        manifest = evaluation.get('fid_reference_manifest')
        if protocol == 'matched_train_50k' and not manifest:
            raise ValueError('Matched training-reference selection requires a reference manifest')
        selection = dict(protocol=protocol, samples=50000,
                    reference_sha256=evaluation['rq_fid_reference_sha256'],
                    batch_size=int(evaluation['batch_size']), seed=73000,
                    pixel_protocol='FP32 decoder; continuous float [0,1]; no uint8 rounding')
        if manifest:
            selection['reference_manifest_sha256'] = file_sha256(manifest)
        return selection

    def selection_fid(self, model, epoch):
        selection = self.fid_selection()
        if selection is not None:
            from src.training.rq_reference_evaluation import evaluate_rq_reference
            record = evaluate_rq_reference(self, model, epoch, selection['samples'],
                self.cfg.evaluation.rq_fid_reference, selection['reference_sha256'], 'selection')
            return record['fid'] if self.rank == 0 else None
        count = self.cfg.evaluation.full_samples
        self.generate(model, epoch, count)
        if self.rank == 0:
            return json.loads((self.out / f'generation-epoch{epoch:03d}-{count}.json').read_text())['pytorch_fid_diagnostic']
        return None

    def build_prior(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.cfg.seed + 1001)
            model = CompoundLaserVAR(self.vae, depth=self.cfg.model.depth, num_classes=self.num_classes,
                                      local_width=self.cfg.compound.local_width,
                                      dropout=self.cfg.compound.dropout,
                                      atom_loss_weight=self.cfg.compound.atom_loss_weight,
                                      coefficient_top_p=self.cfg.compound.coefficient_top_p,
                                      scale_loss_weights=self.cfg.compound.get('scale_loss_weights'))
        if self.num_classes == 1:
            model.cond_drop_rate = 0.
        return model.to(self.device)

    @torch.no_grad()
    def validate(self, model, epoch, count=None):
        model.eval()
        total = torch.zeros(3, device=self.device, dtype=torch.float64)
        per_scale = torch.zeros(len(self.cfg.model.patch_nums), 3, device=self.device, dtype=torch.float64)
        n = count or self.cfg.evaluation.validation_images
        selected = np.random.default_rng(123).permutation(len(self.val))[:n][self.rank::self.world]
        dropout, model.cond_drop_rate = model.cond_drop_rate, 0.
        try:
            for item in self.loader(self.val_tokens if self.val_tokens is not None else self.val,
                                    self.cfg.evaluation.batch_size, selected):
                if self.val_tokens is not None:
                    labels = item['labels']
                    codes = restore_cached_codes(self.vae.quantize, item)
                else:
                    images, labels = item
                    codes = self.codes(self.encode(images))
                with self.amp():
                    from src.models.multiscale_laser_var import VAR
                    features = VAR.forward(model, labels.to(self.device), codes['inputs'])
                    a, c = model.token_logits(features, codes['atoms'], codes['coefficients'])
                ae = torch.nn.functional.cross_entropy(a.float().flatten(0, 2), codes['atoms'].flatten(), reduction='none').reshape_as(codes['atoms'])
                ce = torch.nn.functional.cross_entropy(c.float().flatten(0, 2), codes['coefficients'].flatten(), reduction='none').reshape_as(codes['coefficients'])
                total += torch.stack((ae.mean(), ce.mean(), ae.new_ones(()))) * len(labels)
                for scale, (lo, hi) in enumerate(model.begin_ends):
                    per_scale[scale] += torch.stack((ae[:, lo:hi].mean(), ce[:, lo:hi].mean(), ae.new_ones(()))) * len(labels)
        finally:
            model.cond_drop_rate = dropout
        dist.all_reduce(total)
        dist.all_reduce(per_scale)
        ae, ce = (total[:2] / total[2]).tolist()
        self.log('validation', epoch=epoch, atom_nll=ae, coefficient_nll=ce, joint_nll=ae+ce)
        if self.rank == 0:
            atomic_json(self.out / f'validation-epoch{epoch:03d}.json', dict(
                epoch=epoch, images=int(total[2]), atom_nll=ae, coefficient_nll=ce, joint_nll=ae+ce,
                per_scale=(per_scale[:, :2] / per_scale[:, 2:]).tolist()))
        return ae + ce

    def upload(self, paths, epoch):
        if self.run is None or not self.cfg.compound.upload_checkpoints:
            return
        if self.uploader is None:
            staging = Path('/tmp/laser-checkpoint-transfers') / self.base.name / self.out.name
            self.uploader = CheckpointUploader(staging, self._upload_snapshot)
        started = time.monotonic()
        self.uploader.submit(paths, epoch)
        self.log('checkpoint_upload_queued', epoch=epoch, staging_seconds=time.monotonic()-started)

    def _upload_snapshot(self, paths, epoch):
        extras = [self.base/'audit/calibration.json', self.out/'resolved-config.yaml',
                  self.base/'fine-tune-initialization.json']
        extras.extend(self.out.glob('rq-reference-selection-epoch*.json'))
        publish_checkpoint_bundle(self.run, self.cfg.wandb.id + '-checkpoints', paths, epoch,
            metadata=dict(tokenizer_sha256=self.tokenizer_sha256, sampling=self.sampling_options(),
                          fid_selection=self.fid_selection()), extras=extras,
            receipt_path=self.out/'checkpoint-upload.json', best_name='prior-best-fid.pt',
            best_alias='best-fid')

    def train_prior(self):
        cfg = self.cfg.prior
        preflight = self.mode == 'preflight'
        model = self.build_prior()
        optimizer = torch.optim.AdamW(optimizer_groups(model), lr=cfg.lr, betas=(.9, .95),
                                      weight_decay=cfg.weight_decay, fused=True)
        state = dict(epoch=0, batch=0, step=0, best_validation=math.inf, best_fid=math.inf)
        path = self.checkpoints / 'prior-last.pt'
        checkpoint = None
        reference_updates = math.ceil(len(self.train) / self.world) // cfg.batch_size // cfg.accumulation
        reference_epochs = int(cfg.epochs)
        migration = None
        if path.exists() and self.cfg.resume:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            if checkpoint['tokenizer_sha256'] != self.tokenizer_sha256:
                raise ValueError('Tokenizer changed on resume')
            saved_cache = checkpoint.get('token_cache_sha256')
            if saved_cache is not None and saved_cache != self.token_cache_sha256:
                raise ValueError('The stochastic token cache changed on resume')
            if saved_cache is None and self.token_cache_sha256 and not self.execution.get('allow_cache_transition', False):
                raise ValueError('Switching online training to a token cache requires an explicit transition flag')
            state, reference_updates, migration = prepare_resume(
                checkpoint, self.contract(), self.world, len(self.train),
                allow_layout_change=self.execution.get('allow_layout_change', False),
                allow_epoch_extension=self.execution.get('allow_epoch_extension', False))
            reference_epochs = int(checkpoint.get('schedule_reference_epochs',
                                                  checkpoint['training_config']['prior']['epochs']))
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        wrapped = DDP(model, device_ids=[self.device.index], broadcast_buffers=False, gradient_as_bucket_view=True)
        if checkpoint:
            if self.rank < len(checkpoint['rng']):
                restore_rng(checkpoint)
            else:
                seed = self.cfg.seed + self.rank + state['step'] * 1000003
                torch.manual_seed(seed)
                np.random.seed(seed % 2**32)
                random.seed(seed)
            del checkpoint
        if migration is not None and self.rank == 0:
            name = 'epoch-extension.json' if migration.get('kind') == 'epoch_extension' else 'layout-migration.json'
            atomic_json(self.out / name, migration)
        if not preflight and self.rank == 0:
            import wandb
            self.run = wandb.init(entity=self.cfg.wandb.entity, project=self.cfg.wandb.project,
                                  id=self.cfg.wandb.id, name=self.cfg.wandb.id, resume='allow',
                                  group=self.cfg.wandb.group, dir=self.execution.get('wandb_dir', str(self.out)), mode=self.cfg.wandb.mode,
                                  config=OmegaConf.to_container(self.cfg, resolve=True))
            self.run.config.update(OmegaConf.to_container(self.cfg, resolve=True), allow_val_change=True)
            if migration is not None:
                self.run.summary['resume_migration'] = migration
            if self.train_tokens is not None:
                self.run.summary['token_cache'] = dict(manifest_sha256=self.token_cache_sha256,
                    variants=self.train_tokens.variants, selection='complete multiscale trajectory per image',
                    first_resumed_step=state['step'])
                cache_artifact = wandb.Artifact(self.cfg.wandb.id + '-token-cache', type='dataset',
                    metadata=dict(manifest_sha256=self.token_cache_sha256, tokenizer_sha256=self.tokenizer_sha256))
                for name in (*self.train_tokens.manifest['files'].keys(), 'manifest.json'):
                    cache_artifact.add_file(str(Path(self.execution['token_cache_dir']) / name), name=name, policy='immutable')
                self.run.log_artifact(cache_artifact)
            artifact = wandb.Artifact(self.cfg.wandb.id + '-provenance', type='experiment')
            for name in ('resolved-config.yaml', 'initialization.json'):
                artifact.add_file(str(self.out / name), name=name)
            artifact.add_file(str(self.base / 'audit/calibration.json'), name='calibration.json')
            if (self.base / 'fine-tune-initialization.json').exists():
                artifact.add_file(str(self.base / 'fine-tune-initialization.json'), name='fine-tune-initialization.json')
            if (self.base / 'source-manifest.json').exists():
                artifact.add_file(str(self.base / 'source-manifest.json'), name='source-manifest.json')
            runtime = self.base / 'runtime'
            if runtime.exists():
                artifact.add_dir(str(runtime / 'src'), name='source/src')
                artifact.add_dir(str(runtime / 'configs'), name='source/configs')
                artifact.add_file(str(runtime / 'train.py'), name='source/train.py')
            self.run.log_artifact(artifact)
            if path.exists():
                self.upload([path, self.checkpoints / 'prior-best-fid.pt', self.checkpoints / 'prior-best-validation.pt'],
                            state['epoch'])
        self.log('prior_setup', parameters=sum(p.numel() for p in model.parameters()),
                 global_batch=cfg.batch_size * cfg.accumulation * self.world,
                 spatial_sequence_length=model.L, sparse_pairs_per_image=model.sparse_pairs_per_image,
                 precomputed_tokens=self.train_tokens is not None)
        fixed = torch.load(self.base / 'audit/preflight-latents.pt', weights_only=True) if preflight else None
        preflight_losses = []
        if preflight:
            fixed_codes = self.codes(fixed['latent'].to(self.device))
            fixed_labels = fixed['labels'].to(self.device)
            model.cond_drop_rate = 0.
        extras = dict(tokenizer_sha256=self.tokenizer_sha256, training_config=self.contract(), world_size=self.world,
                      schedule_reference_updates=reference_updates, schedule_reference_epochs=reference_epochs,
                      token_cache_sha256=self.token_cache_sha256)
        if (self.base / 'fine-tune-initialization.json').exists():
            extras['fine_tune_initialization'] = json.loads((self.base / 'fine-tune-initialization.json').read_text())
        for epoch in range(state['epoch'], 1 if preflight else cfg.epochs):
            self.train.epoch = epoch
            training_data = self.train_tokens if self.train_tokens is not None else self.train
            training_data.epoch = epoch
            sampler = DistributedSampler(training_data, self.world, self.rank, shuffle=True, seed=self.cfg.seed)
            sampler.set_epoch(epoch)
            loader = DataLoader(training_data, batch_size=cfg.batch_size, sampler=sampler,
                                num_workers=self.cfg.data.workers, pin_memory=True, drop_last=True,
                                generator=torch.Generator().manual_seed(self.cfg.seed + epoch + self.rank))
            updates = len(loader) // cfg.accumulation
            initial_batch = state['batch']
            initial_fraction = state.get('epoch_fraction', initial_batch / (updates * cfg.accumulation))
            iterator = range(self.cfg.compound.preflight_steps) if preflight else loader
            model.train()
            optimizer.zero_grad(set_to_none=True)
            tick, metrics = time.monotonic(), torch.zeros(3, device=self.device)
            for batch, item in enumerate(iterator):
                if batch < state['batch']:
                    continue
                if not preflight and batch >= updates * cfg.accumulation:
                    break
                if preflight:
                    codes, labels = fixed_codes, fixed_labels
                    accumulation = 1
                    for group in optimizer.param_groups:
                        group['lr'] = cfg.lr
                else:
                    if self.train_tokens is not None:
                        labels = item['labels']
                        codes = restore_cached_codes(self.vae.quantize, item, self.coefficient_temperatures)
                    else:
                        images, labels = item
                        codes = self.codes(self.encode(images), stochastic=True)
                    labels = labels.to(self.device)
                    accumulation = cfg.accumulation
                    if batch % accumulation == 0:
                        lr_wd_annealing('lin0', optimizer, cfg.lr, cfg.weight_decay, cfg.weight_decay,
                                        schedule_position(epoch + state.get('epoch_fraction', initial_fraction),
                                                          reference_updates, reference_epochs),
                                        cfg.warmup_epochs * reference_updates, reference_epochs * reference_updates,
                                        wp0=.005, wpe=cfg.final_lr_ratio)
                sync = (batch + 1) % accumulation == 0
                with (nullcontext() if sync else wrapped.no_sync()), self.amp():
                    loss, parts = wrapped(labels, codes['inputs'], codes['atoms'], codes['coefficients'],
                                          codes['coefficient_probabilities'])
                    (loss / accumulation).backward()
                metrics += torch.cat((loss.detach()[None], parts)) / accumulation
                if not sync:
                    continue
                gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                state.update(epoch=epoch, batch=batch+1, step=state['step']+1)
                if not preflight:
                    state['epoch_fraction'] = epoch_fraction(initial_fraction, initial_batch, batch+1,
                                                             updates * cfg.accumulation)
                if preflight:
                    preflight_losses.append(float(loss.detach()))
                if state['step'] == 1 or state['step'] % self.cfg.logging.every_steps == 0:
                    dist.all_reduce(metrics)
                    metrics /= self.world
                    self.log('prior', epoch=epoch, step=state['step'], objective=float(metrics[0]),
                             atom_ce=float(metrics[1]), coefficient_soft_ce=float(metrics[2]),
                             gradient_norm=float(gradient), lr=optimizer.param_groups[0]['lr'],
                             images_per_second=(len(labels) * accumulation * self.world) / (time.monotonic()-tick),
                             peak_memory_gib=torch.cuda.max_memory_allocated()/2**30)
                tick = time.monotonic()
                metrics.zero_()
                stop = self.stop_requested()
                upload_error = torch.tensor(int(self.rank == 0 and self.uploader is not None and
                                               self.uploader.error is not None), device=self.device)
                dist.all_reduce(upload_error, op=dist.ReduceOp.MAX)
                if upload_error.item():
                    save_checkpoint(path, model, optimizer, state, extras)
                    raise RuntimeError('Background checkpoint upload failed; training state saved')
                if state['step'] == 1 or state['step'] % self.cfg.logging.checkpoint_every_steps == 0 or stop:
                    save_checkpoint(path, model, optimizer, state, extras)
                if stop:
                    if self.uploader is not None:
                        self.uploader.close()
                    if self.run:
                        self.run.finish(exit_code=0)
                    return
            state.update(epoch=epoch+1, batch=0, epoch_fraction=0.)
            if preflight:
                initial, final = np.mean(preflight_losses[:5]), np.mean(preflight_losses[-5:])
                if not final < initial * .85:
                    raise RuntimeError(f'Overfit check failed: {initial} -> {final}')
                # Exercise real images, fresh stochastic supports/coefficients,
                # and the production microbatch/accumulation before launch.
                repeated_support_change = None
                stochastic_losses = []
                optimizer.zero_grad(set_to_none=True)
                for microbatch, item in zip(range(2 * cfg.accumulation), loader):
                    if self.train_tokens is not None:
                        labels = item['labels']
                        codes = restore_cached_codes(self.vae.quantize, item, self.coefficient_temperatures)
                    else:
                        images, labels = item
                        codes = self.codes(self.encode(images), stochastic=True)
                    if repeated_support_change is None:
                        latent = fixed['latent'].to(self.device)
                        first = self.codes(latent, stochastic=True)
                        second = self.codes(latent, stochastic=True)
                        repeated_support_change = float((first['atoms'] != second['atoms']).any(-1).float().mean())
                        del latent, first, second
                    sync = (microbatch + 1) % cfg.accumulation == 0
                    with (nullcontext() if sync else wrapped.no_sync()), self.amp():
                        loss, _ = wrapped(labels.to(self.device), codes['inputs'], codes['atoms'],
                                          codes['coefficients'], codes['coefficient_probabilities'])
                        (loss / cfg.accumulation).backward()
                    stochastic_losses.append(float(loss.detach()))
                    if sync:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        state['step'] += 1
                if not repeated_support_change > 0:
                    raise RuntimeError('Repeated visits did not change stochastic targets')
                with torch.random.fork_rng(devices=[self.device.index]):
                    self.validate(model, 0, count=64)
                save_checkpoint(path, model, optimizer, state, extras)
                # Exercises two-rank sampling and both sides of FID collectives.
                with torch.random.fork_rng(devices=[self.device.index]):
                    self.generate(model, 0, 32)
                restored = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
                assert len(restored['rng']) == self.world
                assert restored['optimizer']['state']
                if self.rank == 0:
                    atomic_json(self.out / 'complete.json', dict(initial_loss=float(initial), final_loss=float(final),
                                steps=state['step'], checkpoint_has_optimizer=True, rng_ranks=self.world,
                                finite_gradients=True, distributed_sampling_and_fid=True,
                                production_batch_per_rank=cfg.batch_size, accumulation=cfg.accumulation,
                                precomputed_tokens=self.train_tokens is not None,
                                stochastic_losses=stochastic_losses,
                                repeated_support_change=repeated_support_change))
                break
            with torch.random.fork_rng(devices=[self.device.index]):
                validation = self.validate(model, epoch+1)
            best = validation < state['best_validation']
            state['best_validation'] = min(validation, state['best_validation'])
            save_checkpoint(path, model, optimizer, state, extras)
            if best:
                save_checkpoint(self.checkpoints / 'prior-best-validation.pt', model, optimizer, state, extras)
            if epoch == 0 or (epoch+1) % self.cfg.evaluation.every_epochs == 0 or epoch+1 == cfg.epochs:
                with torch.random.fork_rng(devices=[self.device.index]):
                    fid = [self.selection_fid(model, epoch+1)]
                dist.broadcast_object_list(fid)
                if fid[0] < state['best_fid']:
                    state['best_fid'] = fid[0]
                    state['best_fid_epoch'] = epoch+1
                    save_checkpoint(self.checkpoints / 'prior-best-fid.pt', model, optimizer, state, extras)
                if self.run:
                    self.run.summary['best_fid'] = state['best_fid']
                    self.run.summary['best_fid_epoch'] = state.get('best_fid_epoch')
                    self.run.summary['fid_selection'] = self.fid_selection()
                save_checkpoint(path, model, optimizer, state, extras)
                if self.rank == 0:
                    self.upload([path, self.checkpoints / 'prior-best-fid.pt', self.checkpoints / 'prior-best-validation.pt'], epoch+1)
                dist.barrier()
        if self.rank == 0 and not preflight:
            self.upload([path, self.checkpoints / 'prior-best-fid.pt', self.checkpoints / 'prior-best-validation.pt'], cfg.epochs)
            if self.uploader is not None:
                self.log('checkpoint_upload_wait', epoch=cfg.epochs)
                self.uploader.close()
            atomic_json(self.out / 'complete.json', state)
        if self.run:
            self.run.finish()

    def contract(self):
        cfg = OmegaConf.to_container(self.cfg, resolve=True)
        contract = {key: cfg[key] for key in ('model', 'prior', 'compound', 'seed', 'data')}
        # A new sampler changes best-FID selection. Do not silently compare a
        # resumed checkpoint's old best score with a different sampling policy.
        if cfg.get('sampling'):
            contract['sampling'] = cfg['sampling']
        selection = self.fid_selection()
        if selection is not None:
            contract['fid_selection'] = selection
        return contract


def run(cfg):
    os.environ.setdefault('TORCH_HOME', '/workspace/tmp/official-rqvae-eval-cache')
    os.environ.setdefault('NCCL_NVLS_ENABLE', '0')
    torch.set_num_threads(4)
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl', timeout=timedelta(minutes=30))
    torch.manual_seed(cfg.seed + dist.get_rank())
    np.random.seed(cfg.seed + dist.get_rank())
    random.seed(cfg.seed + dist.get_rank())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    experiment = None
    try:
        experiment = CompoundExperiment(cfg)
        if cfg.compound.mode == 'audit':
            experiment.audit()
        else:
            if cfg.compound.mode == 'train' and not (Path(cfg.output_dir) / 'preflight/complete.json').exists():
                raise RuntimeError('Successful preflight is required before production')
            experiment.train_prior()
    except BaseException as error:
        fatal_worker_error(error, Path(cfg.output_dir)/cfg.compound.mode, dist.get_rank())
    finally:
        dist.destroy_process_group()
