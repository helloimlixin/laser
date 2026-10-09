"""Fresh-image physical-pair baseline with audited scratch/resume state."""
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import signal
import sys
import tempfile
import time

BASE = Path(os.environ['LASER_RUN_BASE'])
OUT = Path(os.environ['LASER_PERSISTENT_BASE'])
ROOT = BASE/'source'
PHASE = os.environ['LASER_PHASE']
VERIFY = OUT/'verification'/PHASE
sys.path[:0] = [str(ROOT), str(ROOT/'runtime')]

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from src.training import rqtransformer as training
from src.training import k4_checkpoint_io as checkpoint_io
from src.training.background_checkpoint import BackgroundCheckpointWriter
from src.training.fresh_images import EpochImageFolder
from src.training.coefficient_jitter import jittered_scalar_targets, jitter_moments
from src.training.coefficient_diagnostics import coefficient_diagnostic_sums, coefficient_diagnostic_metrics
from src.training import original_generation_logging as original_logging
from src.models.physical_pair_scalar_prior import PhysicalPairScalarRQTransformer
from src.models.rqtransformer.attentions import AttentionBlock


def record(name, value):
    path = VERIFY/name
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(value, time=time.time()), indent=2, default=str)+'\n')
    temporary.replace(path)


record('process-rank'+os.environ['RANK']+'.json', dict(pid=os.getpid()))
CURRENT_STEP = 0
REAL_RUN = None
SUB_BIN = os.environ.get('LASER_SUB_BIN_TARGETS') == '1'
FFHQ_ADAPT = os.environ.get('LASER_FFHQ_DECODER_OBJECTIVE') == '1'
REQUESTED_SIGMA_BINS = os.environ.get('LASER_COEFFICIENT_SIGMA_BINS')
SIGMA_BINS = float(REQUESTED_SIGMA_BINS) if REQUESTED_SIGMA_BINS is not None else 25.5875
assert math.isfinite(SIGMA_BINS) and SIGMA_BINS > 0
assert not (SUB_BIN and REQUESTED_SIGMA_BINS is not None)
assert not FFHQ_ADAPT or (REQUESTED_SIGMA_BINS is not None and not SUB_BIN)
NOISE = jitter_moments(.01, .025)
TARGET_TEMPERATURE = 2 * ((.01 if SUB_BIN else SIGMA_BINS) * (6 / 2047))**2
CLEAN_COEFFICIENTS = None
CLEAN_ATOMS = None
COEFFICIENT_AUX = None
DIAGNOSTIC_SUMS = None
DIAGNOSTIC_METRICS = {}
if FFHQ_ADAPT:
    from src.training import imagenet_ffhq_adapter as ffhq_adapter
    ffhq_adapter.verify_archive()
    training.build_model = ffhq_adapter.build_model
    training.CompoundLaserRQTransformer = ffhq_adapter.ImageNetFFHQCompound
native_parser = training.build_parser
ARGS = None
def parser():
    result = native_parser()
    native_parse = result.parse_args
    def parse(*args, **kwargs):
        global ARGS
        ARGS = native_parse(*args, **kwargs)
        if FFHQ_ADAPT:
            assert ARGS.compound_tokens and not ARGS.physical_pair_context
            assert ARGS.compound_pair_autoregressive
            assert ARGS.compound_micro_transformer_layers == 2
            assert ARGS.compound_depth_specific_coeff_heads and ARGS.compound_distribution_geometry
            assert ARGS.atom_loss_weight == 1.5 and ARGS.geometry_loss_weight == .05
            assert ARGS.geometry_start_epoch == 2 and ARGS.geometry_warmup_epochs == 3
            assert ARGS.geometry_top_k == 4 and ARGS.coeff_regression_weight == ARGS.coeff_crps_weight == 0
            ARGS.ffhq_decoder_archive_sha256 = ffhq_adapter.ARCHIVE_SHA256
            ARGS.ffhq_objective_archive_sha256 = ffhq_adapter.ARCHIVE_SHA256
            ARGS.ffhq_adapted_to_imagenet = True
        else:
            assert ARGS.physical_pair_context and not ARGS.compound_tokens
        assert ARGS.token_cache is None and ARGS.init_stage2_checkpoint is None
        assert math.isclose(ARGS.coeff_target_temperature, TARGET_TEMPERATURE, rel_tol=1e-12)
        assert ARGS.coeff_target_space == 'normalized'
        if SUB_BIN:
            ARGS.coefficient_target_distribution = 'bounded continuous jitter before nearest-bin quantization'
            ARGS.coefficient_noise_sigma_bins = NOISE['sigma_bins']
            ARGS.coefficient_noise_cap_bins = NOISE['cap_bins']
            ARGS.coefficient_noise_actual_rms_bins = NOISE['actual_rms_bins']
            ARGS.coefficient_gaussian_temperature_active = False
        else:
            ARGS.coefficient_target_distribution = 'Gaussian kernel evaluated on uniform bin centers'
            ARGS.coefficient_noise_sigma_bins = SIGMA_BINS
            ARGS.coefficient_gaussian_temperature_active = True
        ARGS.checkpoint_fid_metric = original_logging.FID_PROTOCOL
        assert ARGS.metric_backend == 'original-rqvae' and ARGS.fid_real_split == 'train'
        assert ARGS.fid_num_samples == 50000 and ARGS.fid_validation_reference_stats is None
        from src.rqvae_metrics import DistributedOriginalRQVAEMetrics
        assert DistributedOriginalRQVAEMetrics.__module__ == 'src.rqvae_metrics'
        record('generation-protocol-rank'+os.environ['RANK']+'.json', dict(
            passed=True, metric_backend=ARGS.metric_backend, generated_images=50000,
            reference_stats=str(ARGS.fid_reference_stats), reference_split='full_train',
            metric_class=DistributedOriginalRQVAEMetrics.__module__+'.'+DistributedOriginalRQVAEMetrics.__name__,
            generation_metric_keys=[original_logging.FID_KEY, original_logging.IS_KEY],
            companion_metrics=False))
        assert ARGS.total_batch_size == 2048 and ARGS.lr == .0005
        assert ARGS.epochs == ARGS.lr_schedule_epochs == 100 and ARGS.min_lr == 0
        if PHASE == 'preflight':
            assert not ARGS.resume and ARGS.resume_checkpoint is None
        else:
            assert ARGS.resume and ARGS.resume_checkpoint.parent == OUT/'train/checkpoints'
            ARGS.resume_checkpoint = checkpoint_io._checkpoint_upload_source(ARGS.resume_checkpoint)
        return ARGS
    result.parse_args = parse
    return result
training.build_parser = parser


native_checkpoint_load = torch.load
def load_checkpoint(source, *args, **kwargs):
    payload = native_checkpoint_load(source, *args, **kwargs)
    if (PHASE == 'production' and ARGS is not None
            and isinstance(source, (str, os.PathLike))
            and Path(source) == ARGS.resume_checkpoint):
        policy = json.loads((OUT/'original-fid-only-policy.json').read_text())
        payload = original_logging.rebase_checkpoint_fids(payload, policy['previous_evaluations'])
        record('fid-protocol-resume-rank'+os.environ['RANK']+'.json', dict(
            passed=True, checkpoint_fid_metric=payload['config']['checkpoint_fid_metric'],
            best_fid=payload.get('best_fid'), global_step=payload['global_step']))
    return payload
torch.load = load_checkpoint


native_images = training.source_image_dataset
def images(dataset, root, transform, *, split='train'):
    if dataset == 'imagenet' and split == 'train':
        ready = json.loads((root/'training-ready.json').read_text())
        assert ready['passed'] and ready['training_images'] == 1281167
        result = EpochImageFolder(root/split, transform=transform, augmentation_seed=261001)
        assert len(result) == 1281167 and len(result.classes) == 1000
        return result
    return native_images(dataset, root, transform, split=split)
training.source_image_dataset = images


native_aux_init = training.LaserAux.__init__
def aux_init(self, *args, **kwargs):
    kwargs['clamp_coeffs'] = False
    native_aux_init(self, *args, **kwargs)
    expected_bins = torch.linspace(-3, 3, 2048, device=self.coeff_bins.device,
                                   dtype=self.coeff_bins.dtype)
    assert torch.equal(self.coeff_bins, expected_bins), 'Require the FFHQ 8.17 uniform-bin policy'
    assert not any(p.requires_grad for p in self.parameters())
training.LaserAux.__init__ = aux_init
native_encode = training.LaserAux.encode_sparse_components
@torch.no_grad()
def encode(self, image_batch, **kwargs):
    global CLEAN_ATOMS, CLEAN_COEFFICIENTS, COEFFICIENT_AUX
    results = None
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.autocast(device_type=image_batch.device.type, enabled=False):
            if not hasattr(self, '_fixed_dictionary_gram'):
                self._fixed_dictionary_gram = self.dictionary.float().t() @ self.dictionary.float()
            for chunk in image_batch.split(32):
                values = native_encode(self, chunk.float(), dictionary_gram=self._fixed_dictionary_gram, **kwargs)
                if results is None:
                    results = [[] for _ in values]
                for accumulated, value in zip(results, values):
                    accumulated.append(value)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    encoded = tuple(torch.cat(values) for values in results)
    CLEAN_ATOMS, CLEAN_COEFFICIENTS = encoded[0].detach(), encoded[1].detach()
    COEFFICIENT_AUX = self
    return encoded
training.LaserAux.encode_sparse_components = encode


native_targets = training.LaserAux.sparse_targets
target_probed = False
def targets(self, atoms, coefficients, **kwargs):
    global target_probed, CLEAN_COEFFICIENTS, COEFFICIENT_AUX
    CLEAN_COEFFICIENTS = coefficients.detach()
    COEFFICIENT_AUX = self
    if SUB_BIN:
        assert not self.soft_target_physical
        value = jittered_scalar_targets(atoms, coefficients, self.coeff_bins,
            num_atoms=self.num_atoms, stochastic=kwargs.get('stochastic', True),
            hard=kwargs.get('hard', False), compact=kwargs.get('compact', False),
            sigma_bins=NOISE['sigma_bins'], cap_bins=NOISE['cap_bins'])
    else:
        value = native_targets(self, atoms, coefficients, **kwargs)
    if not target_probed:
        tokens, (_, probabilities) = value
        entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1).mean()
        assert torch.isfinite(probabilities).all()
        center_measured_sigma_bins = None
        if SUB_BIN:
            assert float(entropy) <= math.log(2) + 1e-6
            support = probabilities.nonzero(as_tuple=True)
            clean = coefficients[support[:-1]]
            in_range = (clean >= self.coeff_bins[0]) & (clean <= self.coeff_bins[-1])
            width = float((self.coeff_bins[-1]-self.coeff_bins[0])/(len(self.coeff_bins)-1))
            errors = (self.coeff_bins[support[-1]]-clean).abs()/width
            maximum = float(errors[in_range].max())
            assert maximum <= .5251, maximum
            assert int((probabilities > 0).sum(-1).max()) <= 2
        else:
            assert float(entropy) > 4.
            zero_coefficients = coefficients.new_zeros(1, 1, 1, self.sparsity_level)
            zero_atoms = torch.arange(self.sparsity_level, device=atoms.device).view_as(zero_coefficients)
            _, (_, center_probabilities) = native_targets(self, zero_atoms, zero_coefficients,
                temp=TARGET_TEMPERATURE, stochastic=False, compact=True)
            centers = self.coeff_bins.double()
            center_probability = center_probabilities[0, 0, 0, 0].double()
            center_mean = (center_probability * centers).sum()
            center_sigma = (center_probability * (centers-center_mean).square()).sum().sqrt()
            width = float((centers[-1]-centers[0])/(len(centers)-1))
            center_measured_sigma_bins = float(center_sigma) / width
            assert math.isclose(center_measured_sigma_bins, SIGMA_BINS, rel_tol=1e-4)
        assert bool((atoms.sort(-1).values.diff(dim=-1) > 0).all())
        record('targets-rank'+os.environ['RANK']+'.json', dict(passed=True,
            coefficient_entropy_nats=float(entropy),
            coefficient_sigma_bins=NOISE['sigma_bins'] if SUB_BIN else SIGMA_BINS,
            center_measured_sigma_bins=center_measured_sigma_bins,
            coefficient_noise_cap_bins=NOISE['cap_bins'] if SUB_BIN else None,
            coefficient_noise_actual_rms_bins=NOISE['actual_rms_bins'] if SUB_BIN else None,
            maximum_in_range_quantized_target_error_bins=maximum if SUB_BIN else None,
            coefficient_out_of_range_fraction=float(((coefficients < self.coeff_bins[0]) | (coefficients > self.coeff_bins[-1])).float().mean()),
            coefficient_target_distribution='bounded pre-quantization jitter' if SUB_BIN else 'Gaussian on centers',
            coefficient_quantizer='uniform', coefficient_bin_centers=2048,
            coefficient_bin_range=[-3., 3.], coefficient_bin_width=6/2047,
            temperature_floor_bypassed=SUB_BIN,
            fresh_images=True, frozen_encoder_fp32=True, encoder_chunk_size=32,
            coefficient_clipping=False, unique_omp_supports=True, temperature=TARGET_TEMPERATURE))
        target_probed = True
    return value
training.LaserAux.sparse_targets = targets
if FFHQ_ADAPT:
    native_compound_targets = training.LaserAux.compound_coeff_ids
    @torch.no_grad()
    def compound_targets(self, coefficients, **kwargs):
        ids, probabilities = native_compound_targets(self, coefficients, **kwargs)
        if not target_probed:
            _, (_, scalar_probabilities) = targets(self, CLEAN_ATOMS, coefficients,
                temp=kwargs['temp'], stochastic=False, compact=True, hard=False)
            assert torch.equal(probabilities, scalar_probabilities)
            record('ffhq-targets-rank'+os.environ['RANK']+'.json', dict(passed=True,
                normalized_targets=True, compound_and_scalar_gaussian_probabilities_identical=True,
                stochastic_coefficient_context=kwargs.get('stochastic', True),
                sigma_bins=SIGMA_BINS, clean_physical_geometry_targets=True))
        return ids, probabilities
    training.LaserAux.compound_coeff_ids = compound_targets


def wrap(model, backend, device, world):
    assert world == 4 and backend == 'ddp'
    if FFHQ_ADAPT:
        assert type(model) is ffhq_adapter.ImageNetFFHQCompound
        assert list(model.block_size) == [8, 8, 4]
        assert len(model.coeff_micro_transformer.blocks) == 2
        assert len(model.coeff_classifier) == 4 and model.contribution_head is None
        # The maintained trainer assigns its own geometry version after model
        # construction. Correct that label before it builds recovery metadata.
        ARGS.compound_geometry_version = 'ffhq_v4_archived_expected_atom_times_expected_coefficient'
    else:
        assert type(model) is PhysicalPairScalarRQTransformer
        assert list(model.block_size) == [8, 8, 8]
    assert model.config.vocab_size_cond == 1000
    assert len(model.body_transformer.blocks) == 42 and len(model.head_transformer.blocks) == 6
    record('architecture-rank'+os.environ['RANK']+'.json', dict(passed=True,
        parameters=sum(p.numel() for p in model.parameters()), scalar_shape=list(model.block_size),
        decoder=model.decoder_type, device=torch.cuda.get_device_name(device), world_size=world))
    if os.environ.get('LASER_COMPILE_BLOCKS', '1') == '1':
        for block in model.head_transformer.blocks:
            block.attn.short_attention_backend = 'compiled'
        for block in model.modules():
            if isinstance(block, AttentionBlock):
                eager = block.forward
                compiled = torch.compile(eager, fullgraph=True, dynamic=True)
                def forward(x, module=block, eager=eager, compiled=compiled):
                    return compiled(x) if module.training and torch.is_grad_enabled() else eager(x)
                block.forward = forward
    return DDP(model, device_ids=[device.index], broadcast_buffers=False,
               gradient_as_bucket_view=True, bucket_cap_mb=100)
training.wrap_distributed_model = wrap
compiled_objective = torch.compile(training.physical_pair_objective, fullgraph=True, dynamic=True)
def measured_objective(atom_logits, coeff_logits, atoms, probabilities, bins, *args, **kwargs):
    global DIAGNOSTIC_SUMS
    loss = compiled_objective(atom_logits, coeff_logits, atoms, probabilities, bins, *args, **kwargs)
    if (CURRENT_STEP + 1) % 10 == 0:
        sums = coefficient_diagnostic_sums(coeff_logits, CLEAN_COEFFICIENTS, bins, probabilities)
        DIAGNOSTIC_SUMS = sums if DIAGNOSTIC_SUMS is None else DIAGNOSTIC_SUMS + sums
    return loss
training.physical_pair_objective = measured_objective
if FFHQ_ADAPT:
    objective_probed = False
    def measured_compound_objective(atom_logits, coeff_logits, prediction, atoms,
                                    probabilities, physical, **kwargs):
        global DIAGNOSTIC_SUMS, objective_probed
        loss, values = ffhq_adapter.compound_objective(atom_logits, coeff_logits, prediction,
            atoms, probabilities, physical, **kwargs)
        if not objective_probed:
            with torch.no_grad():
                settings = dict(kwargs, geometry_weight=.05)
                probe_loss, probe = ffhq_adapter.compound_objective(atom_logits, coeff_logits,
                    prediction, atoms, probabilities, physical, **settings)
                assert torch.isfinite(probe_loss) and float(probe['geometry']) > 0
                assert torch.equal(physical, COEFFICIENT_AUX.physical_contributions(atoms, CLEAN_COEFFICIENTS))
                record('ffhq-objective-rank'+os.environ['RANK']+'.json', dict(passed=True,
                    archive_sha256=ffhq_adapter.ARCHIVE_SHA256,
                    clean_geometry_target_verified=True, atom_weight=kwargs['atom_weight'],
                    scheduled_geometry_weight=kwargs['geometry_weight'],
                    active_geometry_probe_weight=.05, geometry=float(probe['geometry']),
                    pair_mse=float(probe['geometry_pair_mse']),
                    spatial_mse=float(probe['geometry_spatial_mse'])))
            objective_probed = True
        if (CURRENT_STEP + 1) % 10 == 0:
            sums = coefficient_diagnostic_sums(coeff_logits, CLEAN_COEFFICIENTS,
                COEFFICIENT_AUX.coeff_bins, probabilities)
            DIAGNOSTIC_SUMS = sums if DIAGNOSTIC_SUMS is None else DIAGNOSTIC_SUMS + sums
        return loss, values
    training.compound_objective = measured_compound_objective


WRITER = BackgroundCheckpointWriter()
def save(payload, target):
    checkpoint_io.atomic_torch_save(payload, target, background=WRITER)
training.atomic_torch_save = save
def snapshot(source, destination):
    WRITER.wait()
    local = checkpoint_io._checkpoint_upload_source(Path(source).resolve())
    descriptor, name = tempfile.mkstemp(prefix='best-', suffix='.pt', dir=BASE/'checkpoint-staging')
    os.close(descriptor)
    temporary = Path(name)
    temporary.unlink()
    os.link(local, temporary)
    checkpoint_io._persist_serialized_checkpoint(temporary, Path(destination))
    folder = Path(destination).parent/'.checkpoint-data'
    retained = {p.resolve() for p in folder.parent.glob('*.pt') if p.is_symlink()}
    for path in folder.glob('*.pt'):
        if path not in retained:
            cached = checkpoint_io._local_checkpoint_paths(path)
            path.unlink(missing_ok=True)
            for item in cached or ():
                item.unlink(missing_ok=True)
training.snapshot_checkpoint = snapshot
def upload(*args, **kwargs):
    WRITER.wait()
    return checkpoint_io.upload_selected_checkpoint_files(*args, **kwargs)
training.upload_selected_checkpoint_files = upload


class LoggedRun:
    def __init__(self, run):
        self.run = run
    def __getattr__(self, name):
        return getattr(self.run, name)
    def define_metric(self, name, **kwargs):
        if name in ('val/fid', 'val/inception_score', 'val/inception_score_std'):
            return None
        return self.run.define_metric(name, **kwargs)
    def log(self, data, *args, **kwargs):
        if 'train/loss' in data:
            data = dict(data, **DIAGNOSTIC_METRICS)
            if DIAGNOSTIC_METRICS:
                mae = DIAGNOSTIC_METRICS['train/coeff_mode_mae_bins']
                previous = self.run.summary.get('diagnostics/coeff_mode_mae_best_bins', mae)
                self.run.summary.update({
                    'diagnostics/coeff_mode_mae_last_bins': mae,
                    'diagnostics/coeff_mode_mae_best_bins': min(mae, previous),
                    'diagnostics/coeff_mode_mae_global_step': data['train/global_step']})
        if 'val/fid' in data:
            return original_logging.log_generation(self.run, data, OUT/'train')
        else:
            return self.run.log(data, *args, **kwargs)
    def finish(self, *args, **kwargs):
        WRITER.wait()
        return self.run.finish(*args, **kwargs)


import wandb
native_wandb_init = wandb.init
def wandb_init(*args, **kwargs):
    global REAL_RUN
    kwargs['resume'] = 'never' if PHASE == 'preflight' else 'must'
    run = native_wandb_init(*args, **kwargs)
    REAL_RUN = run
    original_logging.clear_legacy_summaries(run)
    original_logging.define_metrics(run)
    policy_path = OUT/'original-fid-only-policy.json'
    if policy_path.exists():
        policy = json.loads(policy_path.read_text())
        original_logging.restore_summary(run, policy['previous_evaluations'], OUT/'train')
    # Keep this metric's summary scalar: nested min/last summaries became stale
    # after the preflight resume. Explicit scalar best/last diagnostics above
    # retain both values without that aggregation/resume ambiguity.
    run.define_metric('train/coeff_*', step_metric='train/global_step')
    run.define_metric('train/coeff_mode_mae_bins', step_metric='train/global_step')
    run.config.update(dict(stage2_initialization='scratch', stage1_reconstruction_fid=4.210914134979248,
        parent_run='helloimlixin-rutgers/laser/imagenet-rfid421-tiny-jitter-fresh-8h100-20261008',
        recipe_reference_run='helloimlixin-rutgers/laser/imagenet-rfid421-fid15554-epoch64-aggressive6e-8h100-20261005',
        training_augmentation='Resize256 RandomCrop256 RandomHorizontalFlip(0.5), fresh each epoch',
        frozen_encoder_precision='FP32', coefficient_clipping=False,
        coefficient_quantizer='uniform', coefficient_bin_centers=2048,
        coefficient_bin_range=[-3., 3.], coefficient_bin_width=6/2047,
        coefficient_noise_sigma_bins=NOISE['sigma_bins'] if SUB_BIN else SIGMA_BINS,
        coefficient_noise_cap_bins=NOISE['cap_bins'] if SUB_BIN else None,
        coefficient_noise_actual_rms_bins=NOISE['actual_rms_bins'] if SUB_BIN else None,
        coefficient_gaussian_temperature_active=not SUB_BIN,
        coefficient_target_distribution='bounded continuous jitter before nearest-bin quantization' if SUB_BIN else 'Gaussian on centers',
        checkpoint_fid_metric=original_logging.FID_PROTOCOL,
        generation_metric_keys=[original_logging.FID_KEY, original_logging.IS_KEY],
        fid_num_samples=50000, fid_real_split='train', fid_validation_reference_stats=None,
        supersedes_run='imagenet-rfid421-repaired-subbin-scratch-4h200-20261008' if REQUESTED_SIGMA_BINS is not None else None,
        correction=('physical scalar-pair decoder, fresh images, sub-bin bounded target jitter; fresh optimizer and cosine' if SUB_BIN else
                    f'physical scalar-pair decoder, fresh images, Gaussian sigma {SIGMA_BINS:g} bins; fresh optimizer and cosine'),
        historical_k2_fid=16.368215560913086, historical_k2_metric_protocol='torchmetrics validation50k; not directly comparable to original-rqvae training-reference FID'), allow_val_change=True)
    if FFHQ_ADAPT:
        run.config.update(dict(recipe_reference_run='helloimlixin-rutgers/laser/ffhqcmp0804205803',
            decoder='FFHQ8.17 archived compound-v4 adapted to ImageNet1.4B/K4/1000classes',
            objective='archived atom-weighted CE + clean physical pair/spatial geometry',
            ffhq_archive_sha256=ffhq_adapter.ARCHIVE_SHA256,
            compound_geometry_version=ARGS.compound_geometry_version,
            compound_training_seen_atom_mask=False, compound_full_pair_depth_history=True,
            supersedes_run='imagenet-rfid421-sigma200bins-scratch-4h200-20261008',
            correction='User requested adapting FFHQ8.17 decoder and objective; new stage2 weights and Adam'), allow_val_change=True)
    for name in ('diagnosis.json', 'plan.json', 'source-code.tar.gz', 'source-manifest.json',
                 'test-verification.json', 'train.yaml', 'environment.txt',
                 'validation-reference-verification.json', 'tokenizer-gpu-preflight.json', 'sub-bin-policy.json',
                 'coefficient-noise-policy.json', 'noise-distribution-verification.json',
                 'ffhq-adaptation-policy.json', 'ffhq-recipe-verification.json',
                 'original-fid-only-policy.json', 'original-fid-only-tests.json'):
        path = OUT/name
        if path.exists():
            run.save(str(path), base_path=str(OUT), policy='now')
    if PHASE == 'production':
        checkpoint_io.upload_selected_checkpoint_files(run,
            last_checkpoint=OUT/'train/checkpoints/last.pt', best_fid=[],
            upload_dir=OUT/'train/wandb_checkpoints')
        for name in ('stage1-tokenizer.pt', 'imagenet_256_train.npz'):
            run.save(str(BASE/'inputs'/name), base_path=str(BASE/'inputs'), policy='now')
        weights = BASE/'torch-cache/hub/checkpoints'
        for path in weights.glob('*.pth'):
            run.save(str(path), base_path=str(weights), policy='now')
        run.save(str(OUT/'preflight-verification.json'), base_path=str(OUT), policy='now')
    record('wandb.json', dict(id=run.id, url=run.url, online=run.settings.mode == 'online'))
    if PHASE == 'production':
        run.summary['continuation/target_reached'] = False
    return LoggedRun(run)
wandb.init = wandb_init


native_step = torch.optim.AdamW.step
updates = 0
def step(optimizer, *args, **kwargs):
    global updates, CURRENT_STEP, DIAGNOSTIC_SUMS, DIAGNOSTIC_METRICS
    ages = {int(value['step']) for value in optimizer.state.values()}
    if updates == 0:
        assert (not ages) if PHASE == 'preflight' else len(ages) == 1
        record('startup-rank'+os.environ['RANK']+'.json', dict(passed=True,
            empty_optimizer=not optimizer.state, restored_steps=sorted(ages),
            lr=optimizer.param_groups[0]['lr'], phase=PHASE, from_scratch=PHASE == 'preflight'))
    result = native_step(optimizer, *args, **kwargs)
    updates += 1
    ages = {int(value['step']) for value in optimizer.state.values()}
    assert len(ages) == 1
    age = ages.pop()
    CURRENT_STEP = age
    if DIAGNOSTIC_SUMS is not None:
        dist.all_reduce(DIAGNOSTIC_SUMS)
        DIAGNOSTIC_METRICS = coefficient_diagnostic_metrics(
            DIAGNOSTIC_SUMS, COEFFICIENT_AUX.coeff_bins, COEFFICIENT_AUX.coeff_scales)
        assert all(math.isfinite(value) for value in DIAGNOSTIC_METRICS.values())
        if dist.get_rank() == 0:
            record(f'coefficient-metrics-step{age:07d}.json',
                   dict(DIAGNOSTIC_METRICS, global_step=age, globally_reduced=True,
                        continuous_clean_targets=True))
        DIAGNOSTIC_SUMS = None
    if updates in (1, 2, 20) or updates % 10 == 0:
        record('progress-rank'+os.environ['RANK']+'.json', dict(passed=True,
            step=age, optimizer_states=len(optimizer.state), pid=os.getpid(),
            lr=optimizer.param_groups[0]['lr'], peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
    return result
torch.optim.AdamW.step = step


def graceful_stop(*_):
    if ARGS is not None:
        # The native loop saves a complete state at the next optimizer boundary.
        ARGS.max_optimizer_steps = max(1, updates)
signal.signal(signal.SIGTERM, graceful_stop)
signal.signal(signal.SIGINT, graceful_stop)


if __name__ == '__main__':
    try:
        sys.argv = [str(ROOT/'train.py'), *sys.argv[1:]]
        runpy.run_path(str(ROOT/'train.py'), run_name='__main__')
    finally:
        WRITER.close()
