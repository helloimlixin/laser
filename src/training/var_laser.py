"""Resumable ImageNet/CelebA-HQ tokenizer -> next-scale VAR pipeline."""
from contextlib import contextmanager, nullcontext
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import shutil
import subprocess
import sys
import time
from types import ModuleType

import numpy as np
from omegaconf import OmegaConf
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler, Subset
from torchvision import transforms
from torchvision.datasets import STL10
from torchvision.transforms import InterpolationMode
from torchvision.utils import save_image

from src.models.multiscale_laser_var import LaserVQVAE, LaserVAR, VQVAE, UPSTREAM, UPSTREAM_REVISION
from src.models.scratch_var import build_scratch_tokenizer, build_scratch_prior, load_finetune_tokenizer, state_digest, shared_prior_digest
from src.data.var_images import Images, load_manifests
from src.models.discriminator import NLayerDiscriminator
from src.models.rqvae.lpips import LPIPS
from src.original_rq_training import FeatureMoments, atomic_json, file_sha256
from src.training.checkpoint_upload import CheckpointUploader
from src.training.wandb_checkpoints import publish_checkpoint_bundle
from src.training.distributed_failure import fatal_worker_error
from utils.lr_control import lr_wd_annealing

ROOT = Path(__file__).resolve().parents[2]
# Load only the released FID implementation; its package initializer imports
# unrelated text/CLIP metrics and their optional dependencies.
_metrics = ModuleType("_laser_var_metrics")
_metrics.__path__ = [str(ROOT / "third_party/rq-vae-transformer/rqvae/metrics")]
sys.modules.setdefault(_metrics.__name__, _metrics)
from _laser_var_metrics.fid import get_inception_model, frechet_distance


@contextmanager
def evaluation_rng():
    """Evaluation frequency and checkpoint recovery must not alter training RNG."""
    numpy_state, python_state = np.random.get_state(), random.getstate()
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)


class HFSquareImages(Dataset):
    """Deterministic square view of a cached Hugging Face image split."""
    def __init__(self, root, split, train, seed, image_size,
                 torchvision_stl_fallback=False, resize_crop=True):
        hf_path = Path(root) / 'hf'
        if hf_path.is_dir():
            from datasets import load_from_disk
            self.dataset = load_from_disk(str(hf_path))[split]
            self.huggingface = True
        elif torchvision_stl_fallback:
            self.dataset = STL10(str(root), split=split, download=False)
            self.huggingface = False
        else:
            raise FileNotFoundError(f'Cached image dataset not found: {hf_path}')
        self.seed, self.epoch = int(seed), 0
        target = int(image_size)
        resize = max(target, round(target * 1.125)) if resize_crop else target
        operations = [transforms.Resize(resize, interpolation=InterpolationMode.LANCZOS)]
        if resize_crop:
            operations.append(transforms.RandomCrop(target) if train else transforms.CenterCrop(target))
        if train:
            operations.append(transforms.RandomHorizontalFlip())
        operations.extend([transforms.ToTensor(), transforms.Normalize(.5, .5)])
        self.transform = transforms.Compose(operations)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # Fixed evaluation subsets commonly contain numpy.int64 indices;
        # Hugging Face Dataset accepts Python integers only.
        index = int(index)
        item = self.dataset[index]
        image, label = (item['image'], item['label']) if self.huggingface else item
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed + self.epoch * 10000019 + index)
            return self.transform(image), int(label)


def optimizer_groups(model):
    no_decay_keys = ("pos_1LC", "pos_start", "lvl_embed", "gamma", "beta", "ada_gss", "scale_mul")
    groups = [dict(params=[], wd_sc=0., lr_sc=1.), dict(params=[], wd_sc=1., lr_sc=1.)]
    for name, p in model.named_parameters():
        if p.requires_grad:
            exempt = p.ndim == 1 or name.endswith("bias") or any(k in name for k in no_decay_keys)
            groups[0 if exempt else 1]["params"].append(p)
    return groups


def save_checkpoint(path, model, optimizer, state, extras=None):
    rng = dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
               numpy=np.random.get_state(), python=random.getstate())
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, rng)
    if dist.get_rank() == 0:
        payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                       progress=state, rng=gathered, **(extras or {}))
        tmp = path.with_suffix(".tmp")
        # A file object gives every archive the same internal prefix. Identical
        # last/best states then deduplicate in W&B despite different filenames.
        with tmp.open('wb') as stream:
            torch.save(payload, stream)
        tmp.replace(path)
    dist.barrier()


def restore_rng(checkpoint):
    values = checkpoint["rng"][dist.get_rank()]
    torch.set_rng_state(values["torch"])
    if values['cuda'] is not None:
        torch.cuda.set_rng_state(values["cuda"])
    np.random.set_state(values["numpy"])
    random.setstate(values["python"])


class Experiment:
    def __init__(self, cfg):
        self.cfg = cfg
        self.initialization = cfg.model.get('initialization', 'pretrained')
        self.kind = cfg.model.get('bottleneck', 'laser')
        if self.initialization not in ('scratch', 'pretrained', 'finetune'):
            raise ValueError('Model initialization must be explicitly scratch, pretrained, or finetune')
        if self.initialization == 'scratch' and cfg.model.pretrained_vae is not None:
            raise ValueError('Scratch training forbids pretrained_vae')
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.device = (torch.device('cpu') if cfg.get('cpu_smoke', False) else
                       torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))))
        self.out = Path(cfg.output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.checkpoints = Path(cfg.get('execution', {}).get('checkpoint_dir', str(self.out)))
        self.checkpoints.mkdir(parents=True, exist_ok=True)
        self.initial_discriminator = None
        self.finetune_receipt = {}
        dataset_name = str(cfg.data.get('dataset', 'imagenet')).lower()
        data_root = Path(cfg.data.root).expanduser().resolve()
        self.dataset = dataset_name
        self.manifest_data = (dataset_name == 'imagenet' or
                              (dataset_name == 'celebahq' and not (data_root / 'hf').is_dir()))
        if self.manifest_data:
            manifests = load_manifests(cfg.data.manifests, self.dataset)
            self.num_classes = len(manifests['train']['classes'])
            data_fingerprints = {
                f'{split}_manifest_sha256': file_sha256(Path(cfg.data.manifests)/f'{split}-manifest.json')
                for split in ('train', 'val')}
        elif dataset_name in {'stl10', 'celebahq', 'ffhq'}:
            hf_path = data_root / 'hf'
            if hf_path.is_dir():
                files = sorted(hf_path.rglob('*.arrow'))
                if not files:
                    raise ValueError(f'No cached Arrow files found under {hf_path}')
                data_fingerprints = {f'arrow_{i:02d}_sha256': file_sha256(p) for i, p in enumerate(files)}
            else:
                if dataset_name != 'stl10':
                    raise FileNotFoundError(f'Cached {dataset_name} dataset not found: {hf_path}')
                binary = data_root / 'stl10_binary'
                data_fingerprints = {
                    name.replace('.', '_') + '_sha256': file_sha256(binary / name)
                    for name in ('train_X.bin', 'train_y.bin', 'test_X.bin', 'test_y.bin')}
        else:
            raise ValueError(f'Unsupported VAR dataset: {dataset_name}')
        if self.rank == 0:
            contract = {key:OmegaConf.to_container(cfg[key], resolve=True)
                        for key in ('model', 'tokenizer', 'prior')}
            contract.update(seed=int(cfg.seed), world_size=self.world,
                            dataset=self.dataset, data_root=cfg.data.get('identity_root', str(data_root)),
                            **data_fingerprints)
            if self.manifest_data:
                contract['horizontal_flip'] = bool(cfg.data.get('horizontal_flip', False))
            contract_path = self.out/'training-contract.json'
            if contract_path.exists():
                saved_contract = json.loads(contract_path.read_text())
                comparable_saved = json.loads(json.dumps(saved_contract))
                comparable_new = json.loads(json.dumps(contract))
                extensions = {}
                for phase in ('tokenizer', 'prior'):
                    old_epochs = int(comparable_saved[phase].pop('epochs'))
                    new_epochs = int(comparable_new[phase].pop('epochs'))
                    if new_epochs < old_epochs:
                        raise ValueError(f'{phase} epochs cannot decrease on resume')
                    if new_epochs != old_epochs:
                        extensions[phase] = dict(from_epochs=old_epochs, to_epochs=new_epochs)
                if comparable_saved != comparable_new:
                    raise ValueError('Training configuration changed; use a new output directory')
                if extensions:
                    atomic_json(self.out/f'contract-extension-{int(time.time())}.json', extensions)
                    atomic_json(contract_path, contract)
            else:
                atomic_json(contract_path, contract)
        self.stop = False
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "stop", True))
        signal.signal(signal.SIGINT, lambda *_: setattr(self, "stop", True))
        self.run = None
        self.uploader = None
        if self.rank == 0:
            import wandb
            self.run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project,
                                 id=cfg.wandb.id, name=cfg.wandb.id,
                                 resume="allow", mode=cfg.wandb.mode,
                                 dir=os.environ.get("WANDB_DIR", str(cfg.get('execution', {}).get('wandb_dir', self.out))),
                                 group=cfg.wandb.get('group'),
                                 config=OmegaConf.to_container(cfg, resolve=True))
            self.run.config.update(OmegaConf.to_container(cfg, resolve=True), allow_val_change=True)
            OmegaConf.save(cfg, self.out / "resolved-config.yaml", resolve=True)
        self.global_log_step = 0
        if self.manifest_data:
            self.train = Images(data_root / "train", manifests["train"], True, cfg.seed, self.dataset,
                                horizontal_flip=cfg.data.get('horizontal_flip', False))
            self.val = Images(data_root / "val", manifests["val"], False, cfg.seed, self.dataset)
        elif dataset_name == 'stl10':
            self.num_classes = 10
            self.train = HFSquareImages(data_root, 'train', True, cfg.seed, cfg.data.image_size, True)
            self.val = HFSquareImages(data_root, 'test', False, cfg.seed, cfg.data.image_size, True)
        else:
            self.num_classes = 1 if dataset_name == 'ffhq' else 2
            self.train = HFSquareImages(data_root, 'train', True, cfg.seed,
                                        cfg.data.image_size, resize_crop=False)
            self.val = HFSquareImages(data_root, 'validation', False, cfg.seed,
                                      cfg.data.image_size, resize_crop=False)
        if self.initialization in ('scratch', 'finetune'):
            self.vae = build_scratch_tokenizer(cfg.model, cfg.seed)
            if self.initialization == 'finetune':
                source = Path(cfg.model.init_tokenizer_checkpoint)
                payload = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
                indices = load_finetune_tokenizer(self.vae, payload, cfg.model.source_patch_nums)
                if cfg.tokenizer.get('init_discriminator', False):
                    self.initial_discriminator = payload['discriminator']
                self.finetune_receipt = dict(source_checkpoint=str(source), source_checkpoint_sha256=file_sha256(source),
                    source_progress=payload.get('progress'), retained_source_scale_indices=indices,
                    optimizer='fresh', discriminator='source weights' if self.initial_discriminator else 'fresh')
                del payload
        else:
            self.vae = LaserVQVAE(pretrained=cfg.model.pretrained_vae, sparsity=cfg.model.sparsity,
                                 coefficient_bins=cfg.model.coefficient_bins, ch=cfg.model.vae_width,
                                 channels=cfg.model.channels, atoms=cfg.model.atoms,
                                 patch_nums=tuple(cfg.model.patch_nums))
        self.initialization_receipt = dict(
            initialization=self.initialization, bottleneck=self.kind, seed=int(cfg.seed),
            pretrained_checkpoint=str(cfg.model.pretrained_vae) if self.initialization == 'pretrained' else None,
            **self.finetune_receipt,
            model_sha256=state_digest(self.vae), shared_backbone_sha256=state_digest(self.vae, shared_only=True))
        if self.kind == 'laser' and self.vae.quantize.tokenized_sparse_policy is not None:
            self.initialization_receipt['tokenized_sparse_policy'] = self.vae.quantize.tokenized_sparse_policy
        if self.rank == 0:
            # Runtime-only PyTorch containers need not include the git CLI.
            # Our pinned upstream snapshot has a detached, exact-commit HEAD.
            revision = (subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
                        if shutil.which('git') else (UPSTREAM/'.git/HEAD').read_text().strip())
            if revision != UPSTREAM_REVISION:
                raise ValueError(f"Expected FoundationVision/VAR revision {UPSTREAM_REVISION}, got {revision}")
            files = [ROOT / "src/models/multiscale_laser_var.py", ROOT/'src/models/scratch_var.py', Path(__file__), ROOT / "src/models/dictionary_learner.py", ROOT / "src/models/sparse_token_codec.py"]
            sites = sum(p*p for p in cfg.model.patch_nums)
            bits = sites*math.ceil(math.log2(cfg.model.atoms))
            if self.kind == 'laser':
                bits = sites*cfg.model.sparsity*(math.ceil(math.log2(cfg.model.atoms))+math.ceil(math.log2(cfg.model.coefficient_bins)))
            atomic_json(self.out / "provenance.json", dict(
                upstream_revision=revision, source_sha256={str(p.relative_to(ROOT)): file_sha256(p) for p in files},
                vae_sha256=file_sha256(cfg.model.pretrained_vae) if cfg.model.pretrained_vae else None,
                data_fingerprints=data_fingerprints, dataset=dataset_name,
                **(dict(manifest_sha256={s: data_fingerprints[f'{s}_manifest_sha256'] for s in ('train', 'val')},
                        training_horizontal_flip=bool(cfg.data.get('horizontal_flip', False)))
                   if self.manifest_data else {}),
                train_images=len(self.train), val_images=len(self.val), classes=self.num_classes,
                image_size=int(cfg.data.image_size), downsampling=16,
                spatial_sites=sum(p*p for p in cfg.model.patch_nums),
                sparse_pairs=sites*cfg.model.sparsity if self.kind=='laser' else None,
                nominal_bits_per_image=bits, **self.initialization_receipt,
                protocol=("Aligned full-frame 256px RGB; unconditional face label 0; disjoint fixed splits"
                          if self.manifest_data and dataset_name == 'celebahq' else
                          "256px square images; seeded horizontal flip during training; fixed validation images"
                          if dataset_name in {'celebahq', 'ffhq'} else
                          "Fresh resize/crop during training; deterministic validation views")))
            if self.initialization == 'scratch' and not (self.checkpoints/'initial-tokenizer.pt').exists():
                torch.save(dict(model=self.vae.state_dict(), **self.initialization_receipt), self.checkpoints/'initial-tokenizer.pt')
            execution = cfg.get('execution', {})
            if not cfg.smoke_steps and execution.get('publish_provenance', False):
                artifact = wandb.Artifact(cfg.wandb.id + '-provenance', type='experiment')
                for path in (self.out/'provenance.json', self.out/'resolved-config.yaml',
                             self.out/'training-contract.json', data_root/'manifest.json'):
                    artifact.add_file(str(path), name=path.name)
                artifact.add_dir(str(ROOT/'src'), name='source/src')
                artifact.add_dir(str(ROOT/'configs'), name='source/configs')
                artifact.add_dir(str(ROOT/'scripts/tools'), name='source/scripts/tools')
                self.run.log_artifact(artifact)
            if not cfg.smoke_steps and execution.get('upload_dataset', False):
                artifact = wandb.Artifact(execution.dataset_artifact, type='dataset',
                    metadata=dict(dataset=dataset_name, classes=self.num_classes,
                                  train_images=len(self.train), validation_images=len(self.val),
                                  manifest_sha256=file_sha256(data_root/'manifest.json')))
                for path in sorted((data_root/'hf').rglob('*')):
                    if path.is_file():
                        artifact.add_file(str(path), name=str(path.relative_to(data_root)), policy='immutable')
                artifact.add_file(str(data_root/'manifest.json'), name='manifest.json', policy='immutable')
                self.run.log_artifact(artifact)
        self.vae.to(self.device)
        self.inception = None

    def log(self, phase, **values):
        if self.rank == 0:
            record = dict(phase=phase, time=time.time(), **values)
            atomic_json(self.out / "status.json", record)
            print(json.dumps(record), flush=True)
            if self.run:
                self.run.log({phase + "/" + k: v for k, v in values.items() if isinstance(v, (int, float))})

    def amp(self):
        return torch.autocast("cuda", dtype=torch.bfloat16) if self.device.type == 'cuda' else nullcontext()

    def media_path(self, name):
        directory = Path(self.cfg.get('execution', {}).get('media_dir', str(self.out)))
        directory.mkdir(parents=True, exist_ok=True)
        return directory/name

    def upload_tokenizer(self, epoch, close=False):
        execution = self.cfg.get('execution', {})
        if self.rank != 0 or self.cfg.smoke_steps or not execution.get('upload_checkpoints', False):
            return
        if self.uploader is None:
            staging = Path('/tmp/laser-checkpoint-transfers') / self.cfg.wandb.id
            self.uploader = CheckpointUploader(staging, self._upload_tokenizer_snapshot)
        self.uploader.submit([self.checkpoints/'tokenizer-last.pt', self.checkpoints/'tokenizer-best.pt'], epoch)
        if close:
            self.log('checkpoint_upload_wait', epoch=epoch)
            self.uploader.close()

    def _upload_tokenizer_snapshot(self, paths, epoch):
        publish_checkpoint_bundle(self.run, self.cfg.wandb.id + '-checkpoints', paths, epoch,
            metadata=dict(initialization=self.initialization),
            extras=[self.out/'resolved-config.yaml', self.out/'provenance.json'],
            receipt_path=self.out/'checkpoint-upload.json', best_name='tokenizer-best.pt',
            best_alias='best-rfid')

    def record_first_batch(self, phase, images, labels, sampler):
        """Record inputs without consuming random state or changing sample order."""
        if self.rank != 0 or (self.out/f'{phase}-first-batch.json').exists():
            return
        atomic_json(self.out/f'{phase}-first-batch.json', dict(
            rank=self.rank, epoch=0, batch_size=len(images),
            pixels_sha256=hashlib.sha256(images.contiguous().numpy().tobytes()).hexdigest(),
            labels_sha256=hashlib.sha256(labels.contiguous().numpy().tobytes()).hexdigest(),
            epoch_sampler_sha256=hashlib.sha256(np.asarray(list(sampler), dtype='<i8').tobytes()).hexdigest()))

    def loader(self, dataset, batch, indices=None):
        if indices is not None:
            dataset = Subset(dataset, indices)
        return DataLoader(dataset, batch_size=batch, num_workers=self.cfg.data.workers,
                          pin_memory=True, persistent_workers=False,
                          generator=torch.Generator().manual_seed(self.cfg.seed + self.rank))

    def upload_checkpoint(self, stage, epoch, step):
        every = int(self.cfg.logging.get('artifact_every_epochs', 0))
        final = epoch == self.cfg["tokenizer" if stage == "tokenizer" else "prior"].epochs
        if self.cfg.smoke_steps or not every or (epoch % every and not final):
            return
        if self.rank == 0 and self.run and self.cfg.wandb.mode == 'online':
            import wandb
            artifact = wandb.Artifact(f'{self.run.id}-{stage}-checkpoints', type='model',
                                      metadata=dict(stage=stage, epoch=epoch, step=step,
                                                    dataset=self.dataset, world_size=self.world))
            names = ['tokenizer-last.pt', 'resolved-config.yaml', 'provenance.json', 'training-contract.json']
            if stage == 'prior':
                names.append('prior-last.pt')
            if stage == 'prior' and (self.out/'prior-best-fid.pt').exists():
                names.extend(['prior-best-fid.pt', 'best-prior.json'])
            for name in names:
                directory = self.checkpoints if name == 'tokenizer-last.pt' else self.out
                artifact.add_file(str(directory/name), name=name, policy='immutable', skip_cache=True)
            if self.manifest_data:
                for split in ('train', 'val'):
                    artifact.add_file(str(Path(self.cfg.data.manifests)/f'{split}-manifest.json'),
                                      name=f'{split}-manifest.json', policy='immutable', skip_cache=True)
            source = os.environ.get('LASER_SOURCE_ARCHIVE')
            if source:
                artifact.add_file(source, name='source.tar.gz', policy='immutable', skip_cache=True)
            self.run.log_artifact(artifact, aliases=['latest', f'epoch-{epoch}']).wait()
        dist.barrier()

    @torch.no_grad()
    def fid_reference(self):
        supplied = self.cfg.evaluation.get('fid_reference')
        if supplied:
            return Path(supplied)
        if not getattr(self, 'manifest_data', False):
            return None
        if self.dataset != 'celebahq':
            raise ValueError('ImageNet requires its explicit FID reference')
        path = self.out/'celebahq-validation-fid.npz'
        exists = [path.exists() if self.rank == 0 else None]
        dist.broadcast_object_list(exists, src=0)
        if exists[0]:
            receipt = json.loads(path.with_suffix('.json').read_text())
            if receipt['manifest_sha256'] != file_sha256(Path(self.cfg.data.manifests)/'val-manifest.json'):
                raise ValueError('FID reference belongs to a different validation manifest')
            return path
        if self.inception is None:
            self.inception = get_inception_model().eval().requires_grad_(False).to(self.device)
        moments = FeatureMoments(self.device)
        for images, _ in self.loader(self.val, self.cfg.evaluation.batch_size,
                                     list(range(self.rank, len(self.val), self.world))):
            pixels = ((images.to(self.device)+1)*127.5).round().clamp(0,255).byte()
            moments.update(self.inception(pixels.float()/255))
        count, mean, covariance = moments.finish()
        if count != len(self.val):
            raise RuntimeError('FID reference count mismatch')
        if self.rank == 0:
            temporary = path.with_suffix('.tmp.npz')
            np.savez(temporary, mu=mean, sigma=covariance, count=count)
            temporary.replace(path)
            atomic_json(path.with_suffix('.json'), dict(count=count, dataset=self.dataset,
                split='val', manifest_sha256=file_sha256(Path(self.cfg.data.manifests)/'val-manifest.json'),
                transform='aligned full-frame Lanczos 256 RGB, uint8', backend='pytorch-fid Inception'))
        dist.barrier()
        return path

    def stop_requested(self):
        deadline = float(os.environ.get("LASER_STOP_TIME_UNIX", "inf"))
        flag = torch.tensor(int(self.stop or time.time() >= deadline), device=self.device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    @torch.no_grad()
    def calibrate(self):
        self.vae.eval()
        self.vae.quantize.coefficient_max.fill_(.1)
        count = self.cfg.tokenizer.calibration_images
        selected = np.random.default_rng(self.cfg.seed + 17).choice(len(self.train), count, replace=False)
        for images, _ in self.loader(self.train, self.cfg.tokenizer.batch_size, selected[self.rank::self.world]):
            with self.amp():
                self.vae.tokenize(images.to(self.device), calibrate=True)
        dist.all_reduce(self.vae.quantize.coefficient_max, op=dist.ReduceOp.MAX)
        self.vae.quantize.coefficient_max.mul_(1.2)
        self.log("calibration", training_images=count)
        if self.rank == 0:
            atomic_json(self.out / "coefficient-ranges.json", self.vae.quantize.coefficient_max.tolist())

    @evaluation_rng()
    @torch.no_grad()
    def reconstruction(self, epoch, count):
        self.vae.eval()
        if self.inception is None:
            self.inception = get_inception_model().eval().requires_grad_(False).to(self.device)
        # Validation images and the frozen Inception network do not change.
        # Keep globally reduced real moments per subset size within this run.
        cache_reference = bool(self.cfg.get('execution', {}).get('cache_reconstruction_reference', False))
        reference_cache = self.__dict__.setdefault('_reconstruction_reference_cache', {})
        reference = reference_cache.get(int(count)) if cache_reference else None
        real = FeatureMoments(self.device) if reference is None else None
        fake = FeatureMoments(self.device)
        baseline = baseline_moments = None
        if self.initialization == 'pretrained' and (epoch == 0 or count == self.cfg.evaluation.full_reconstruction_images):
            baseline = VQVAE(vocab_size=self.cfg.model.atoms, z_channels=self.cfg.model.channels,
                             ch=self.cfg.model.vae_width, v_patch_nums=tuple(self.cfg.model.patch_nums), test_mode=True).to(self.device)
            baseline.load_state_dict(torch.load(self.cfg.model.pretrained_vae, map_location='cpu', weights_only=True))
            baseline_moments = FeatureMoments(self.device)
        totals = torch.zeros(3, device=self.device, dtype=torch.float64)
        selected = np.random.default_rng(7919).permutation(len(self.val))[:count]
        for batch, (images, _) in enumerate(self.loader(self.val, self.cfg.evaluation.batch_size, selected[self.rank::self.world])):
            images = images.to(self.device)
            with self.amp():
                recon, _, _ = self.vae(images)
            recon = recon.float().clamp(-1, 1)
            original = ((images + 1) * .5).clamp(0, 1)
            decoded = (recon + 1) * .5
            if real is not None:
                real.update(self.inception(original))
            fake.update(self.inception(decoded))
            if baseline is not None:
                with self.amp():
                    baseline_recon, _, _ = baseline(images)
                baseline_moments.update(self.inception(((baseline_recon.float()+1)*.5).clamp(0,1)))
            mse = (decoded - original).square().mean((1, 2, 3))
            totals += torch.stack((mse.sum(), (-10 * mse.clamp_min(1e-12).log10()).sum(), mse.new_tensor(len(images))))
            if batch == 0 and self.rank == 0:
                path = self.media_path(f"reconstruction-epoch{epoch:03d}.png")
                save_image(torch.stack((original[:8], decoded[:8]), 1).flatten(0, 1), path, nrow=8)
                if self.run:
                    import wandb
                    self.run.log({"tokenizer/reconstructions": wandb.Image(str(path))})
        r, f = (real.finish() if real is not None else reference), fake.finish()
        base = baseline_moments.finish() if baseline_moments else None
        dist.all_reduce(totals)
        if r[0] != count or f[0] != count:
            raise RuntimeError("Reconstruction evaluation count mismatch")
        if cache_reference:
            reference_cache[int(count)] = r
        quality = [None]
        if self.rank == 0:
            score = float(frechet_distance(r[1], r[2], f[1], f[2]))
            extra = {}
            if base is not None:
                base_score = float(frechet_distance(r[1], r[2], base[1], base[2]))
                extra = dict(released_vq_matched_rfid=base_score, rfid_drift=score-base_score)
            self.log("reconstruction", epoch=epoch, count=count, matched_rfid=score,
                     mse=float(totals[0]/totals[2]), psnr=float(totals[1]/totals[2]), **extra)
            atomic_json(self.out / f"reconstruction-epoch{epoch:03d}-{count}.json",
                        dict(epoch=epoch, count=count, matched_rfid=score, **extra,
                             feature_backend="pytorch-fid Inception; matched original validation images"))
            quality[0] = dict(matched_rfid=score, **extra)
        dist.broadcast_object_list(quality, src=0)
        return quality[0]

    def train_tokenizer(self):
        cfg = self.cfg.tokenizer
        path = self.checkpoints / "tokenizer-last.pt"
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.cfg.seed+101)
            disc = NLayerDiscriminator(norm="group")
        if self.initial_discriminator is not None:
            disc.load_state_dict(self.initial_discriminator, strict=True)
            self.initial_discriminator = None
        if self.rank == 0:
            atomic_json(self.out/'discriminator-initialization.json', dict(seed=int(self.cfg.seed+101), sha256=state_digest(disc)))
        disc.to(self.device)
        perceptual = LPIPS().eval().requires_grad_(False).to(self.device)
        channels_last = bool(self.cfg.get('execution', {}).get('channels_last', False))
        if channels_last:
            for module in (self.vae, disc, perceptual):
                module.to(memory_format=torch.channels_last)
        elif self.cfg.get('execution', {}).get('lpips_channels_last', False):
            perceptual.to(memory_format=torch.channels_last)
        dictionary_params = list((self.vae.quantize.dictionary if self.kind=='laser' else self.vae.quantize.embedding).parameters())
        dictionary_ids = {id(p) for p in dictionary_params}
        optimizer = torch.optim.AdamW([
            {"params": [p for p in self.vae.parameters() if id(p) not in dictionary_ids], "lr": cfg.lr},
            {"params": dictionary_params, "lr": cfg.dictionary_lr}], betas=(.5, .9), weight_decay=0.)
        d_optimizer = torch.optim.Adam(disc.parameters(), lr=cfg.lr, betas=(.5, .9))
        state = dict(epoch=0, batch=0, step=0)
        checkpoint = None
        if path.exists() and self.cfg.resume:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            self.vae.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            disc.load_state_dict(checkpoint["discriminator"])
            d_optimizer.load_state_dict(checkpoint["discriminator_optimizer"])
            state = checkpoint["progress"]
            if channels_last:
                # Resume Adam moments in the same layout as their parameters.
                for opt in (optimizer, d_optimizer):
                    for values in opt.state.values():
                        for key, value in values.items():
                            if torch.is_tensor(value) and value.ndim == 4:
                                values[key] = value.contiguous(memory_format=torch.channels_last)
        elif self.initialization == 'pretrained':
            self.calibrate()
        device_ids = [self.device.index] if self.device.type == 'cuda' else None
        model = DDP(self.vae, device_ids=device_ids, broadcast_buffers=False)
        d_model = DDP(disc, device_ids=device_ids, broadcast_buffers=False)
        if checkpoint:
            restore_rng(checkpoint)
            del checkpoint
        if not path.exists() and not self.cfg.smoke_steps:
            with torch.random.fork_rng(devices=[self.device.index] if self.device.type == 'cuda' else []):
                self.reconstruction(0, self.cfg.evaluation.reconstruction_images)
        best_path = self.checkpoints / 'tokenizer-best.pt'
        best_record_path = self.out / 'tokenizer-best.json'
        best_quality = (json.loads(best_record_path.read_text())['matched_rfid']
                        if best_record_path.exists() else float('inf'))
        for epoch in range(state["epoch"], cfg.epochs):
            self.train.epoch = epoch
            sampler = DistributedSampler(self.train, self.world, self.rank, shuffle=True, seed=self.cfg.seed, drop_last=True)
            sampler.set_epoch(epoch)
            loader = DataLoader(self.train, batch_size=cfg.batch_size, sampler=sampler,
                                num_workers=self.cfg.data.workers, pin_memory=True, drop_last=True,
                                generator=torch.Generator().manual_seed(self.cfg.seed+epoch+self.rank))
            # Drop at most accumulation-1 microbatches for an exact effective batch.
            batches = len(loader) // cfg.accumulation * cfg.accumulation
            if batches == 0:
                raise ValueError("Tokenizer effective batch exceeds the training set")
            model.train()
            optimizer.zero_grad(set_to_none=True)
            d_optimizer.zero_grad(set_to_none=True)
            tick = time.monotonic()
            metrics = torch.zeros(4, device=self.device)
            for batch, (images, labels) in enumerate(loader):
                if batch < state["batch"]:
                    continue
                if batch >= batches:
                    break
                if state['step'] == 0 and batch == 0:
                    self.record_first_batch('tokenizer', images, labels, sampler)
                images = images.to(self.device, non_blocking=True,
                    memory_format=torch.channels_last if channels_last else torch.contiguous_format)
                sync = (batch + 1) % cfg.accumulation == 0
                adversarial = state["step"] >= cfg.adversarial_start
                warmup = int(cfg.get('warmup_steps', 0))
                lr_ratio = min(1., (state['step']+1)/warmup) if warmup else 1.
                optimizer.param_groups[0]['lr'] = cfg.lr*lr_ratio
                optimizer.param_groups[1]['lr'] = cfg.dictionary_lr*lr_ratio
                for group in d_optimizer.param_groups:
                    group['lr'] = cfg.lr*lr_ratio
                for p in disc.parameters():
                    p.requires_grad_(False)
                with (nullcontext() if sync else model.no_sync()), self.amp():
                    reconstructed, _, bottleneck = model(images)
                    reconstruction = F.l1_loss(reconstructed, images) + perceptual(reconstructed, images)
                    gan = -disc(reconstructed).mean() if adversarial else reconstructed.new_zeros(())
                    gan_weight = reconstructed.new_tensor(cfg.adversarial_weight)
                    if adversarial and cfg.get('adaptive_adversarial', False):
                        last_layer = self.vae.decoder.conv_out.weight
                        rec_grad = torch.autograd.grad(reconstruction, last_layer, retain_graph=True)[0]
                        adv_grad = torch.autograd.grad(gan, last_layer, retain_graph=True)[0]
                        gan_weight = (rec_grad.float().norm()/(adv_grad.float().norm()+1e-4)).clamp(0,1e4).detach()*cfg.adversarial_weight
                    loss = reconstruction + bottleneck + gan_weight * gan
                    (loss / cfg.accumulation).backward()
                d_loss = loss.new_zeros(())
                if adversarial:
                    for p in disc.parameters():
                        p.requires_grad_(True)
                    with (nullcontext() if sync else d_model.no_sync()), self.amp():
                        logits = d_model(torch.cat((images, reconstructed.detach())))
                        real_logits, fake_logits = logits.chunk(2)
                        d_loss = (F.relu(1-real_logits).mean()+F.relu(1+fake_logits).mean()) * .5
                        (d_loss/cfg.accumulation).backward()
                    if sync:
                        torch.nn.utils.clip_grad_norm_(disc.parameters(), 1., error_if_nonfinite=True)
                        d_optimizer.step()
                        d_optimizer.zero_grad(set_to_none=True)
                metrics += torch.stack((loss.detach(), reconstruction.detach(), bottleneck.detach(), d_loss.detach())) / cfg.accumulation
                if not sync:
                    continue
                gradient = torch.nn.utils.clip_grad_norm_(self.vae.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                if self.kind == 'laser':
                    self.vae.quantize.dictionary.normalize_dictionary_()
                optimizer.zero_grad(set_to_none=True)
                state = dict(epoch=epoch, batch=batch+1, step=state["step"]+1)
                if state["step"] == 1 or state["step"] % self.cfg.logging.every_steps == 0:
                    dist.all_reduce(metrics)
                    metrics /= self.world
                    ids = self.vae.quantize.last_atom_ids if self.kind=='laser' else self.vae.last_atom_ids
                    counts = torch.bincount(ids.flatten(), minlength=self.cfg.model.atoms).float()
                    dist.all_reduce(counts)
                    probability = counts/counts.sum()
                    perplexity = (-(probability*probability.clamp_min(1e-12).log()).sum()).exp()
                    clipping = self.vae.quantize.last_clip_fraction.clone() if self.kind=='laser' else counts.new_zeros(())
                    dist.all_reduce(clipping)
                    self.log("tokenizer", epoch=epoch, step=state["step"], loss=metrics[0].item(),
                             reconstruction=metrics[1].item(), dictionary_loss=metrics[2].item(),
                             discriminator_loss=metrics[3].item(), gradient_norm=gradient.item(),
                             lr=optimizer.param_groups[0]['lr'], gan_weight=gan_weight.item(),
                             atom_perplexity=perplexity.item(), atom_usage=(counts>0).float().mean().item(),
                             coefficient_clip_fraction=(clipping/self.world).item(),
                             images_per_second=cfg.batch_size*cfg.accumulation*self.world/(time.monotonic()-tick),
                             peak_memory_gib=torch.cuda.max_memory_allocated()/2**30 if self.device.type == 'cuda' else 0.)
                tick = time.monotonic()
                metrics.zero_()
                stop = self.stop_requested() or (self.cfg.smoke_steps and state["step"] >= self.cfg.smoke_steps)
                upload_failed = torch.tensor(int(self.rank == 0 and self.uploader is not None and
                                                self.uploader.error is not None), device=self.device)
                dist.all_reduce(upload_failed, op=dist.ReduceOp.MAX)
                if upload_failed.item():
                    save_checkpoint(path, self.vae, optimizer, state, dict(discriminator=disc.state_dict(), discriminator_optimizer=d_optimizer.state_dict(), initialization=self.initialization_receipt))
                    raise RuntimeError('Tokenizer upload failed; full training state saved')
                if state["step"] == 1 or state["step"] % self.cfg.logging.checkpoint_every_steps == 0 or stop:
                    save_checkpoint(path, self.vae, optimizer, state, dict(discriminator=disc.state_dict(), discriminator_optimizer=d_optimizer.state_dict(), initialization=self.initialization_receipt))
                if stop:
                    self.upload_tokenizer(epoch, close=True)
                    if self.cfg.smoke_steps and self.rank == 0:
                        atomic_json(self.out / 'tokenizer-smoke-complete.json', dict(progress=state, ranks=self.world))
                    return bool(self.cfg.smoke_steps)
            state.update(epoch=epoch+1, batch=0)
            save_checkpoint(path, self.vae, optimizer, state, dict(discriminator=disc.state_dict(), discriminator_optimizer=d_optimizer.state_dict(), initialization=self.initialization_receipt))
            evaluation_every = int(cfg.get('evaluation_every_epochs', 1))
            if (epoch + 1) % evaluation_every == 0 or epoch + 1 == cfg.epochs:
                with torch.random.fork_rng(devices=[self.device.index] if self.device.type == 'cuda' else []):
                    quality = self.reconstruction(epoch+1, self.cfg.evaluation.reconstruction_images)
                if cfg.get('select_best', False) and quality['matched_rfid'] < best_quality:
                    best_quality = quality['matched_rfid']
                    if self.rank == 0:
                        import shutil
                        temporary = best_path.with_suffix('.tmp')
                        shutil.copyfile(path, temporary)
                        temporary.replace(best_path)
                        atomic_json(best_record_path, dict(epoch=epoch+1, count=self.cfg.evaluation.reconstruction_images, **quality))
                    dist.barrier()
            upload_every = int(self.cfg.get('execution', {}).get('upload_every_epochs', 5))
            if epoch == 0 or (epoch + 1) % upload_every == 0:
                self.upload_tokenizer(epoch + 1)
            self.upload_checkpoint("tokenizer", epoch+1, state["step"])
        selected_epoch = cfg.epochs
        if cfg.get('select_best', False):
            selected = torch.load(best_path, map_location='cpu', weights_only=False, mmap=True)
            self.vae.load_state_dict(selected['model'], strict=True)
            selected_epoch = selected['progress']['epoch']
            del selected
        quality = self.reconstruction(selected_epoch, self.cfg.evaluation.full_reconstruction_images)
        self.upload_tokenizer(cfg.epochs, close=True)
        if ((cfg.get('max_matched_rfid') is not None and quality['matched_rfid'] > cfg.max_matched_rfid) or
                (cfg.get('max_rfid_drift') is not None and quality.get('rfid_drift',0) > cfg.max_rfid_drift)):
            self.log('tokenizer_quality_rejected', **quality)
            return False
        if self.rank == 0:
            atomic_json(self.out / 'tokenizer-complete.json', dict(progress=state, last_quality=quality,
                        selected_checkpoint=str(best_path if cfg.get('select_best', False) else path)))
        return True

    @evaluation_rng()
    @torch.no_grad()
    def validate_prior(self, model, epoch):
        model.eval()
        n = self.cfg.evaluation.validation_images
        indices = np.random.default_rng(123).permutation(len(self.val))[:n][self.rank::self.world]
        total = torch.zeros(4, device=self.device)
        for images, labels in self.loader(self.val, self.cfg.evaluation.batch_size, indices):
            with self.amp():
                codes = self.vae.tokenize(images.to(self.device))
                # Upstream VAR drops labels even in eval; explicitly disable it.
                dropout, model.cond_drop_rate = model.cond_drop_rate, 0.
                try:
                    loss, parts = model(labels.to(self.device), codes["inputs"], codes["atoms"], codes["coefficients"])
                finally:
                    model.cond_drop_rate = dropout
            total += torch.stack((loss, parts[0], parts[1], loss.new_tensor(1.))) * len(images)
        dist.all_reduce(total)
        self.log("validation", epoch=epoch, joint_nll=(total[0]/total[3]).item(),
                 atom_nll=(total[1]/total[3]).item(), coefficient_nll=(total[2]/total[3]).item())

    def sampling_options(self):
        return {}

    @evaluation_rng()
    @torch.no_grad()
    def generate(self, model, epoch, count, official=False):
        model.eval()
        if self.inception is None:
            self.inception = get_inception_model().eval().requires_grad_(False).to(self.device)
        moments = FeatureMoments(self.device)
        cfg = self.cfg.prior
        reference = self.fid_reference()
        indices = list(range(self.rank, count, self.world))
        sample_path = self.out / f"samples-epoch{epoch:03d}-{count}.npy"
        if official and self.rank == 0:
            values = np.lib.format.open_memmap(sample_path, mode="w+", dtype=np.uint8, shape=(count, 256, 256, 3))
            del values
        dist.barrier()
        values = np.load(sample_path, mmap_mode="r+") if official else None
        for begin in range(0, len(indices), self.cfg.evaluation.batch_size):
            selected = indices[begin:begin+self.cfg.evaluation.batch_size]
            labels = torch.tensor([i % self.num_classes for i in selected], device=self.device)
            with self.amp():
                latent = model.sample(labels, cfg=cfg.cfg, top_k=cfg.top_k, top_p=cfg.top_p,
                                      seed=73000+self.rank+begin*self.world, **self.sampling_options())
                images = latent.float() if self.kind=='vq' else (self.vae.fhat_to_img(latent).float()+1)*.5
            # Evaluate precisely the uint8 images exported for the official toolkit.
            pixels = images.mul(255).round().clamp(0,255).byte()
            moments.update(self.inception(pixels.float()/255))
            if values is not None:
                values[selected] = pixels.permute(0,2,3,1).cpu().numpy()
            if begin == 0 and self.rank == 0:
                path = self.media_path(f"generated-epoch{epoch:03d}.png")
                save_image(images[:self.cfg.evaluation.get('preview_samples', 32)], path,
                           nrow=int(self.cfg.evaluation.get('grid_columns', 8)))
                if self.run:
                    import wandb
                    self.run.log({"prior/samples": wandb.Image(str(path))})
        if values is not None:
            values.flush()
            del values
        result = moments.finish()
        if result[0] != count:
            raise RuntimeError("Generated sample count mismatch")
        # FeatureMoments.finish performs distributed collectives. Every rank
        # must compute its real-image shard, including nonzero ranks.
        reference_count = None
        if reference is None:
            real = FeatureMoments(self.device)
            reference_count = min(int(self.cfg.evaluation.get('fid_reference_images', count)), len(self.val))
            real_indices = np.random.default_rng(2718).permutation(len(self.val))[:reference_count][self.rank::self.world]
            for real_images, _ in self.loader(self.val, self.cfg.evaluation.batch_size, real_indices):
                real.update(self.inception(((real_images.to(self.device).float()+1)*.5).clamp(0,1)))
            real_count, ref_mu, ref_sigma = real.finish()
            if real_count != reference_count:
                raise RuntimeError('Real-image FID count mismatch')
        score = None
        if self.rank == 0:
            if reference is not None:
                ref = np.load(reference)
                ref_mu, ref_sigma = ref['mu'], ref['sigma']
            score = float(frechet_distance(result[1], result[2], ref_mu, ref_sigma))
            self.log("generation", epoch=epoch, count=count, **{f'fid_{count}': score})
            atomic_json(self.out / f"generation-epoch{epoch:03d}-{count}.json",
                        dict(epoch=epoch, count=count, pytorch_fid_diagnostic=score,
                             cfg=cfg.cfg, top_k=cfg.top_k, top_p=cfg.top_p,
                             sampling_options=self.sampling_options(),
                             real_reference_images=reference_count,
                             note="Official ADM metrics are required for published VAR comparison"))
            if official and self.cfg.evaluation.get("adm_evaluator"):
                # zip stores the .npy without loading the 9.2GB sample array into RAM.
                import zipfile
                archive = sample_path.with_suffix(".npz")
                with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
                    zf.write(sample_path, arcname="arr_0.npy")
                sample_path.unlink()
                evaluator = self.cfg.evaluation
                log_path = self.out / f"adm-epoch{epoch:03d}.log"
                with log_path.open("w") as stream:
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", TF_CPP_MIN_LOG_LEVEL="2", TF_ENABLE_ONEDNN_OPTS="0",
                               TF_NUM_INTEROP_THREADS="4", TF_NUM_INTRAOP_THREADS="16", OMP_NUM_THREADS="16")
                    subprocess.run([str(Path(evaluator.adm_python).resolve()), str(Path(evaluator.adm_evaluator).resolve()),
                                    str(Path(evaluator.adm_reference).resolve()), str(archive.resolve())],
                                   stdout=stream, stderr=subprocess.STDOUT, env=env, check=True,
                                   cwd=str(Path(evaluator.adm_evaluator).resolve().parent))
                import re
                metrics = {}
                for key, value in re.findall(r"^(FID|sFID|Inception Score|Precision|Recall):\s*([0-9.eE+-]+)", log_path.read_text(), re.M):
                    metrics[key.lower().replace(" ", "_")] = float(value)
                if "fid" not in metrics:
                    raise RuntimeError("ADM evaluator did not report FID")
                self.log("adm", epoch=epoch, count=count, **metrics)
                atomic_json(self.out / f"adm-epoch{epoch:03d}.json", dict(epoch=epoch, count=count, **metrics))
        dist.barrier()
        result = [score]
        dist.broadcast_object_list(result, src=0)
        return result[0]

    def retain_best_prior(self, score, epoch, step, count):
        if self.rank == 0:
            receipt = self.out/'best-prior.json'
            old = json.loads(receipt.read_text()) if receipt.exists() else None
            if old is not None and old['count'] != count:
                raise ValueError('Cannot rank checkpoints using different FID sample counts')
            if old is None or score < old['fid']:
                temporary = self.out/'prior-best-fid.tmp'
                if temporary.exists():
                    temporary.unlink()
                # Last is atomically replaced, so a hard link preserves this
                # completed checkpoint without copying several GB each epoch.
                try:
                    os.link(self.out/'prior-last.pt', temporary)
                except OSError:
                    shutil.copyfile(self.out/'prior-last.pt', temporary)
                temporary.replace(self.out/'prior-best-fid.pt')
                atomic_json(receipt, dict(fid=score, count=count, epoch=epoch, step=step))
                self.log('best_prior', fid=score, count=count, epoch=epoch, step=step)
        dist.barrier()

    def train_prior(self):
        cfg = self.cfg.prior
        self.vae.eval().requires_grad_(False)
        self.validate_tokenizer_roundtrip()
        if self.initialization == 'scratch':
            model = build_scratch_prior(self.vae, self.kind, self.cfg.seed+1001, depth=self.cfg.model.depth, num_classes=self.num_classes)
            if self.rank == 0:
                atomic_json(self.out/'prior-initialization.json', dict(initialization='scratch', seed=int(self.cfg.seed+1001), shared_sha256=shared_prior_digest(model)))
        else:
            model = LaserVAR(self.vae, depth=self.cfg.model.depth, num_classes=self.num_classes)
        model.to(self.device)
        optimizer = torch.optim.AdamW(optimizer_groups(model), lr=cfg.lr, betas=(.9,.95), weight_decay=cfg.weight_decay,
                                      fused=self.device.type == 'cuda')
        path = self.out / "prior-last.pt"
        state = dict(epoch=0, batch=0, step=0)
        checkpoint = None
        tokenizer_digest = file_sha256(self.out / "tokenizer-last.pt")
        if path.exists() and self.cfg.resume:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if checkpoint["tokenizer_sha256"] != tokenizer_digest:
                raise ValueError("Prior checkpoint belongs to a different tokenizer")
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            state = checkpoint["progress"]
        wrapped = DDP(model, device_ids=[self.device.index] if self.device.type == 'cuda' else None, broadcast_buffers=False)
        if checkpoint:
            restore_rng(checkpoint)
            del checkpoint
        setup = dict(parameters=sum(p.numel() for p in model.parameters()),
                     global_batch=cfg.batch_size*cfg.accumulation*self.world,
                     spatial_sequence_length=int(model.L),
                     original_var_spatial_sequence_length=sum(p*p for p in self.cfg.model.patch_nums),
                     spatial_sequence_ratio_to_original_var=1.0)
        if self.kind == 'laser':
            setup.update(sparse_depth=int(model.sparsity),
                         sparse_pairs_per_image=int(model.sparse_pairs_per_image),
                         categorical_decisions_per_image=int(model.categorical_decisions_per_image))
        self.log("prior_setup", **setup)
        for epoch in range(state["epoch"], cfg.epochs):
            self.train.epoch = epoch
            sampler = DistributedSampler(self.train, self.world, self.rank, shuffle=True, seed=self.cfg.seed, drop_last=True)
            sampler.set_epoch(epoch)
            loader = DataLoader(self.train, batch_size=cfg.batch_size, sampler=sampler,
                                num_workers=self.cfg.data.workers, pin_memory=True, drop_last=True,
                                generator=torch.Generator().manual_seed(self.cfg.seed+epoch+self.rank))
            updates = len(loader)//cfg.accumulation
            if updates == 0:
                raise ValueError("Prior effective batch exceeds the training set")
            model.train()
            optimizer.zero_grad(set_to_none=True)
            tick = time.monotonic()
            metrics = torch.zeros(3, device=self.device)
            for batch, (images, labels) in enumerate(loader):
                if batch < state["batch"]:
                    continue
                if batch >= updates*cfg.accumulation:
                    break
                if state['step'] == 0 and batch == 0:
                    self.record_first_batch('prior', images, labels, sampler)
                if batch % cfg.accumulation == 0:
                    lr_wd_annealing("lin0", optimizer, cfg.lr, cfg.weight_decay, cfg.weight_decay,
                                    state["step"], cfg.warmup_epochs*updates, cfg.epochs*updates,
                                    wp0=.005, wpe=cfg.final_lr_ratio)
                with torch.no_grad(), self.amp():
                    codes = self.vae.tokenize(images.to(self.device, non_blocking=True))
                sync = (batch+1)%cfg.accumulation == 0
                with (nullcontext() if sync else wrapped.no_sync()), self.amp():
                    loss, parts = wrapped(labels.to(self.device), codes["inputs"], codes["atoms"], codes["coefficients"])
                    (loss/cfg.accumulation).backward()
                metrics += torch.cat((loss.detach()[None], parts)) / cfg.accumulation
                if not sync:
                    continue
                gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                state = dict(epoch=epoch, batch=batch+1, step=state["step"]+1)
                if state["step"] == 1 or state["step"] % self.cfg.logging.every_steps == 0:
                    dist.all_reduce(metrics)
                    metrics /= self.world
                    self.log("prior", epoch=epoch, step=state["step"], joint_nll=metrics[0].item(),
                             atom_nll=metrics[1].item(), coefficient_nll=metrics[2].item(),
                             gradient_norm=gradient.item(), lr=optimizer.param_groups[0]["lr"],
                             images_per_second=cfg.batch_size*cfg.accumulation*self.world/(time.monotonic()-tick),
                             peak_memory_gib=torch.cuda.max_memory_allocated()/2**30 if self.device.type == 'cuda' else 0.)
                tick=time.monotonic()
                metrics.zero_()
                stop = self.stop_requested() or (self.cfg.smoke_steps and state["step"]>=self.cfg.smoke_steps)
                if state["step"]==1 or state["step"]%self.cfg.logging.checkpoint_every_steps==0 or stop:
                    save_checkpoint(path, model, optimizer, state, dict(tokenizer_sha256=tokenizer_digest))
                if stop:
                    if self.cfg.smoke_steps:
                        self.validate_prior(model, epoch)
                        self.generate(model, epoch, min(32,self.cfg.evaluation.preview_samples))
                    return
            state.update(epoch=epoch+1,batch=0)
            save_checkpoint(path, model, optimizer, state, dict(tokenizer_sha256=tokenizer_digest))
            self.validate_prior(model, epoch+1)
            if epoch==0 or (epoch+1)%self.cfg.evaluation.every_epochs==0:
                full = (epoch+1)%self.cfg.evaluation.full_every_epochs==0 or epoch+1==cfg.epochs
                # Always use the same count for the selection curve. Larger
                # export evaluations are separately named by sample count.
                count = self.cfg.evaluation.preview_samples
                same_count = count == self.cfg.evaluation.full_samples
                score = self.generate(model, epoch+1, count, official=full and same_count)
                self.retain_best_prior(score, epoch+1, state['step'], count)
                if full and not same_count:
                    self.generate(model, epoch+1, self.cfg.evaluation.full_samples, official=True)
            self.upload_checkpoint("prior", epoch+1, state["step"])
        self.log("complete", tokenizer_epochs=self.cfg.tokenizer.epochs, prior_epochs=cfg.epochs)
        if self.rank == 0:
            atomic_json(self.out/'completed.json', dict(tokenizer_epochs=self.cfg.tokenizer.epochs,
                                                       prior_epochs=cfg.epochs, time=time.time()))

    @evaluation_rng()
    @torch.no_grad()
    def validate_tokenizer_roundtrip(self):
        if self.kind != 'laser':
            return
        images, _ = next(iter(self.loader(self.val, 2, [self.rank*2, self.rank*2+1])))
        images = images.to(self.device)
        before = self.vae.quantize.coefficient_max.clone()
        with self.amp():
            codes = self.vae.tokenize(images)
            latent, inputs = self.vae.quantize.from_codes(codes['atoms'], codes['coefficients'])
            direct, _, _ = self.vae(images)
            decoded = self.vae.decoder(self.vae.post_quant_conv(latent))
        torch.testing.assert_close(latent, codes['latent'], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(inputs, codes['inputs'], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(decoded.float(), direct.float(), atol=.02, rtol=.02)
        torch.testing.assert_close(before, self.vae.quantize.coefficient_max, atol=0, rtol=0)
        if not torch.isfinite(decoded).all():
            raise FloatingPointError('Nonfinite sparse-code reconstruction')
        atomic_json(self.out/f'tokenizer-roundtrip-rank{self.rank}.json', dict(
            passed=True, max_decode_error=(decoded.float()-direct.float()).abs().max().item()))
        dist.barrier()
        self.log('tokenizer_roundtrip', passed=True, checked_images=2*self.world,
                 max_decode_error=(decoded.float()-direct.float()).abs().max().item())


def run(cfg):
    if int(cfg.data.image_size) % 16 or tuple(cfg.model.patch_nums)[-1] != int(cfg.data.image_size) // 16:
        raise ValueError('VAR scale endpoint must equal image_size / tokenizer downsampling (16)')
    os.environ.setdefault("TORCH_HOME", "/workspace/tmp/official-rqvae-eval-cache")
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
    torch.set_num_threads(4)
    cpu_smoke = cfg.get('cpu_smoke', False)
    if cpu_smoke and not cfg.smoke_steps:
        raise ValueError('CPU execution is only allowed for bounded smoke tests')
    if not cpu_smoke:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    if "RANK" not in os.environ:
        raise ValueError("Launch this backend with torchrun, including for one GPU")
    dist.init_process_group('gloo' if cpu_smoke else 'nccl', timeout=timedelta(minutes=30))
    torch.manual_seed(cfg.seed + dist.get_rank())
    np.random.seed(cfg.seed + dist.get_rank())
    random.seed(cfg.seed + dist.get_rank())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = bool(cfg.get('execution', {}).get('cudnn_benchmark', False))
    if torch.backends.cudnn.benchmark:
        torch.backends.cudnn.benchmark_limit = 8
    experiment = None
    try:
        experiment = Experiment(cfg)
        if cfg.stage == "stage1":
            completed = experiment.train_tokenizer()
            if completed and cfg.pipeline:
                torch.cuda.empty_cache()
                experiment.train_prior()
        else:
            ckpt = torch.load(Path(cfg.output_dir)/"tokenizer-last.pt", map_location="cpu", weights_only=False)
            experiment.vae.load_state_dict(ckpt["model"])
            del ckpt
            experiment.train_prior()
        if experiment.run:
            experiment.run.finish()
    except BaseException as exc:
        fatal_worker_error(exc, cfg.output_dir, dist.get_rank())
    finally:
        dist.destroy_process_group()
