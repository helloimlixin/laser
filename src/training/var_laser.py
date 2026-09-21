"""Resumable matched ImageNet VQ/LASER tokenizer -> next-scale VAR pipeline."""
from contextlib import nullcontext
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
from types import ModuleType

import numpy as np
from PIL import Image
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
from src.models.scratch_var import build_scratch_tokenizer, build_scratch_prior, state_digest, shared_prior_digest
from src.models.discriminator import NLayerDiscriminator
from src.models.rqvae.lpips import LPIPS
from src.original_rq_training import FeatureMoments, atomic_json, file_sha256
from utils.lr_control import lr_wd_annealing

ROOT = Path(__file__).resolve().parents[2]
# Load only the released FID implementation; its package initializer imports
# unrelated text/CLIP metrics and their optional dependencies.
_metrics = ModuleType("_laser_var_metrics")
_metrics.__path__ = [str(ROOT / "third_party/rq-vae-transformer/rqvae/metrics")]
sys.modules.setdefault(_metrics.__name__, _metrics)
from _laser_var_metrics.fid import get_inception_model, frechet_distance


class Images(Dataset):
    def __init__(self, root, manifest, train, seed):
        self.root, self.rows, self.seed = Path(root), manifest["samples"], seed
        self.epoch = 0
        self.transform = transforms.Compose([
            transforms.Resize(288, interpolation=InterpolationMode.LANCZOS),
            transforms.RandomCrop(256) if train else transforms.CenterCrop(256),
            transforms.ToTensor(), transforms.Normalize(.5, .5)])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        name, label = self.rows[index]
        # Fresh image augmentation each epoch, reproducible across worker counts.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed + self.epoch * 10000019 + index)
            with Image.open(self.root / name) as im:
                image = self.transform(im.convert("RGB"))
        return image, label


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
    rng = dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(),
               numpy=np.random.get_state(), python=random.getstate())
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, rng)
    if dist.get_rank() == 0:
        payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                       progress=state, rng=gathered, **(extras or {}))
        tmp = path.with_suffix(".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)
    dist.barrier()


def restore_rng(checkpoint):
    values = checkpoint["rng"][dist.get_rank()]
    torch.set_rng_state(values["torch"])
    torch.cuda.set_rng_state(values["cuda"])
    np.random.set_state(values["numpy"])
    random.setstate(values["python"])


class Experiment:
    def __init__(self, cfg):
        self.cfg = cfg
        self.initialization = cfg.model.get('initialization', 'pretrained')
        self.kind = cfg.model.get('bottleneck', 'laser')
        if self.initialization not in ('scratch', 'pretrained'):
            raise ValueError('Model initialization must be explicitly scratch or pretrained')
        if self.initialization == 'scratch' and cfg.model.pretrained_vae is not None:
            raise ValueError('Scratch training forbids pretrained_vae')
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
        self.out = Path(cfg.output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        dataset_name = str(cfg.data.get('dataset', 'imagenet')).lower()
        data_root = Path(cfg.data.root).expanduser().resolve()
        if dataset_name == 'imagenet':
            data_fingerprints = {
                f'{split}_manifest_sha256': file_sha256(Path(cfg.data.manifests)/f'{split}-manifest.json')
                for split in ('train', 'val')}
        elif dataset_name in {'stl10', 'celebahq'}:
            hf_path = data_root / 'hf'
            if hf_path.is_dir():
                files = sorted(hf_path.rglob('*.arrow'))
                if not files:
                    raise ValueError(f'No cached Arrow files found under {hf_path}')
                data_fingerprints = {f'arrow_{i:02d}_sha256': file_sha256(p) for i, p in enumerate(files)}
            else:
                if dataset_name != 'stl10':
                    raise FileNotFoundError(f'Cached CelebA-HQ dataset not found: {hf_path}')
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
                            dataset=dataset_name, data_root=str(data_root), **data_fingerprints)
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
        if self.rank == 0:
            import wandb
            self.run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project,
                                 id=cfg.wandb.id, name=cfg.wandb.id,
                                 resume="allow", mode=cfg.wandb.mode, dir=str(self.out),
                                 group=cfg.wandb.get('group'),
                                 config=OmegaConf.to_container(cfg, resolve=True))
            OmegaConf.save(cfg, self.out / "resolved-config.yaml", resolve=True)
        self.global_log_step = 0
        if dataset_name == 'imagenet':
            manifests = {s: json.loads((Path(cfg.data.manifests) / f"{s}-manifest.json").read_text()) for s in ("train", "val")}
            if manifests["train"]["classes"] != manifests["val"]["classes"]:
                raise ValueError("Train and validation label mappings differ")
            for split, expected in (("train", 1281167), ("val", 50000)):
                if len(manifests[split]["samples"]) != expected or len(manifests[split]["classes"]) != 1000:
                    raise ValueError(f"Incomplete ImageNet {split} manifest")
            self.num_classes = 1000
            self.train = Images(data_root / "train", manifests["train"], True, cfg.seed)
            self.val = Images(data_root / "val", manifests["val"], False, cfg.seed)
        elif dataset_name == 'stl10':
            self.num_classes = 10
            self.train = HFSquareImages(data_root, 'train', True, cfg.seed, cfg.data.image_size, True)
            self.val = HFSquareImages(data_root, 'test', False, cfg.seed, cfg.data.image_size, True)
        else:
            self.num_classes = 2
            self.train = HFSquareImages(data_root, 'train', True, cfg.seed,
                                        cfg.data.image_size, resize_crop=False)
            self.val = HFSquareImages(data_root, 'validation', False, cfg.seed,
                                      cfg.data.image_size, resize_crop=False)
        if self.initialization == 'scratch':
            self.vae = build_scratch_tokenizer(cfg.model, cfg.seed)
        else:
            self.vae = LaserVQVAE(pretrained=cfg.model.pretrained_vae, sparsity=cfg.model.sparsity,
                                 coefficient_bins=cfg.model.coefficient_bins, ch=cfg.model.vae_width,
                                 channels=cfg.model.channels, atoms=cfg.model.atoms,
                                 patch_nums=tuple(cfg.model.patch_nums))
        self.initialization_receipt = dict(
            initialization=self.initialization, bottleneck=self.kind, seed=int(cfg.seed),
            pretrained_checkpoint=None if self.initialization == 'scratch' else str(cfg.model.pretrained_vae),
            model_sha256=state_digest(self.vae), shared_backbone_sha256=state_digest(self.vae, shared_only=True))
        if self.rank == 0:
            revision = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
            if revision != UPSTREAM_REVISION:
                raise ValueError(f"Expected FoundationVision/VAR revision {UPSTREAM_REVISION}, got {revision}")
            files = [ROOT / "src/models/multiscale_laser_var.py", ROOT/'src/models/scratch_var.py', Path(__file__), ROOT / "src/models/dictionary_learner.py"]
            sites = sum(p*p for p in cfg.model.patch_nums)
            bits = sites*math.ceil(math.log2(cfg.model.atoms))
            if self.kind == 'laser':
                bits = sites*cfg.model.sparsity*(math.ceil(math.log2(cfg.model.atoms))+math.ceil(math.log2(cfg.model.coefficient_bins)))
            atomic_json(self.out / "provenance.json", dict(
                upstream_revision=revision, source_sha256={str(p.relative_to(ROOT)): file_sha256(p) for p in files},
                vae_sha256=file_sha256(cfg.model.pretrained_vae) if cfg.model.pretrained_vae else None,
                data_fingerprints=data_fingerprints, dataset=dataset_name,
                train_images=len(self.train), val_images=len(self.val), classes=self.num_classes,
                image_size=int(cfg.data.image_size), downsampling=16,
                spatial_sites=sum(p*p for p in cfg.model.patch_nums),
                sparse_pairs=sites*cfg.model.sparsity if self.kind=='laser' else None,
                nominal_bits_per_image=bits, **self.initialization_receipt,
                protocol="Fresh 288px resize/256px random crop every training epoch; no horizontal flip, official sorted WNIDs"))
            if self.initialization == 'scratch' and not (self.out/'initial-tokenizer.pt').exists():
                torch.save(dict(model=self.vae.state_dict(), **self.initialization_receipt), self.out/'initial-tokenizer.pt')
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
        return torch.autocast("cuda", dtype=torch.bfloat16)

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

    def stop_requested(self):
        flag = torch.tensor(int(self.stop), device=self.device)
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

    @torch.no_grad()
    def reconstruction(self, epoch, count):
        self.vae.eval()
        if self.inception is None:
            self.inception = get_inception_model().eval().requires_grad_(False).to(self.device)
        real, fake = FeatureMoments(self.device), FeatureMoments(self.device)
        baseline = baseline_moments = None
        if self.initialization != 'scratch' and (epoch == 0 or count == self.cfg.evaluation.full_reconstruction_images):
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
            real.update(self.inception(original))
            fake.update(self.inception(decoded))
            if baseline is not None:
                with self.amp():
                    baseline_recon, _, _ = baseline(images)
                baseline_moments.update(self.inception(((baseline_recon.float()+1)*.5).clamp(0,1)))
            mse = (decoded - original).square().mean((1, 2, 3))
            totals += torch.stack((mse.sum(), (-10 * mse.clamp_min(1e-12).log10()).sum(), mse.new_tensor(len(images))))
            if batch == 0 and self.rank == 0:
                path = self.out / f"reconstruction-epoch{epoch:03d}.png"
                save_image(torch.stack((original[:8], decoded[:8]), 1).flatten(0, 1), path, nrow=8)
                if self.run:
                    import wandb
                    self.run.log({"tokenizer/reconstructions": wandb.Image(str(path))})
        r, f = real.finish(), fake.finish()
        base = baseline_moments.finish() if baseline_moments else None
        dist.all_reduce(totals)
        if r[0] != count or f[0] != count:
            raise RuntimeError("Reconstruction evaluation count mismatch")
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
        path = self.out / "tokenizer-last.pt"
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.cfg.seed+101)
            disc = NLayerDiscriminator(norm="group")
        if self.rank == 0:
            atomic_json(self.out/'discriminator-initialization.json', dict(seed=int(self.cfg.seed+101), sha256=state_digest(disc)))
        disc.to(self.device)
        perceptual = LPIPS().eval().requires_grad_(False).to(self.device)
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
        elif self.initialization != 'scratch':
            self.calibrate()
        model = DDP(self.vae, device_ids=[self.device.index], broadcast_buffers=False)
        d_model = DDP(disc, device_ids=[self.device.index], broadcast_buffers=False)
        if checkpoint:
            restore_rng(checkpoint)
            del checkpoint
        if not path.exists() and not self.cfg.smoke_steps:
            self.reconstruction(0, self.cfg.evaluation.reconstruction_images)
        for epoch in range(state["epoch"], cfg.epochs):
            self.train.epoch = epoch
            sampler = DistributedSampler(self.train, self.world, self.rank, shuffle=True, seed=self.cfg.seed)
            sampler.set_epoch(epoch)
            loader = DataLoader(self.train, batch_size=cfg.batch_size, sampler=sampler,
                                num_workers=self.cfg.data.workers, pin_memory=True, drop_last=True,
                                generator=torch.Generator().manual_seed(self.cfg.seed+epoch+self.rank))
            # Drop at most accumulation-1 microbatches for an exact effective batch.
            batches = len(loader) // cfg.accumulation * cfg.accumulation
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
                images = images.to(self.device, non_blocking=True)
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
                             peak_memory_gib=torch.cuda.max_memory_allocated()/2**30)
                tick = time.monotonic()
                metrics.zero_()
                stop = self.stop_requested() or (self.cfg.smoke_steps and state["step"] >= self.cfg.smoke_steps)
                if state["step"] == 1 or state["step"] % self.cfg.logging.checkpoint_every_steps == 0 or stop:
                    save_checkpoint(path, self.vae, optimizer, state, dict(discriminator=disc.state_dict(), discriminator_optimizer=d_optimizer.state_dict(), initialization=self.initialization_receipt))
                if stop:
                    return bool(self.cfg.smoke_steps)
            state.update(epoch=epoch+1, batch=0)
            save_checkpoint(path, self.vae, optimizer, state, dict(discriminator=disc.state_dict(), discriminator_optimizer=d_optimizer.state_dict(), initialization=self.initialization_receipt))
            evaluation_every = int(cfg.get('evaluation_every_epochs', 1))
            if (epoch + 1) % evaluation_every == 0 or epoch + 1 == cfg.epochs:
                self.reconstruction(epoch+1, self.cfg.evaluation.reconstruction_images)
        quality = self.reconstruction(cfg.epochs, self.cfg.evaluation.full_reconstruction_images)
        if ((cfg.get('max_matched_rfid') is not None and quality['matched_rfid'] > cfg.max_matched_rfid) or
                (cfg.get('max_rfid_drift') is not None and quality.get('rfid_drift',0) > cfg.max_rfid_drift)):
            self.log('tokenizer_quality_rejected', **quality)
            return False
        return True

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

    @torch.no_grad()
    def generate(self, model, epoch, count, official=False):
        model.eval()
        if self.inception is None:
            self.inception = get_inception_model().eval().requires_grad_(False).to(self.device)
        moments = FeatureMoments(self.device)
        cfg = self.cfg.prior
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
                                      seed=73000+self.rank+begin*self.world)
                images = latent.float() if self.kind=='vq' else (self.vae.fhat_to_img(latent).float()+1)*.5
            # Evaluate precisely the uint8 images exported for the official toolkit.
            pixels = images.mul(255).round().clamp(0,255).byte()
            moments.update(self.inception(pixels.float()/255))
            if values is not None:
                values[selected] = pixels.permute(0,2,3,1).cpu().numpy()
            if begin == 0 and self.rank == 0:
                path = self.out / f"generated-epoch{epoch:03d}.png"
                save_image(images, path, nrow=4)
                if self.run:
                    import wandb
                    self.run.log({"prior/samples": wandb.Image(str(path))})
        if values is not None:
            values.flush()
            del values
        result = moments.finish()
        if result[0] != count:
            raise RuntimeError("Generated sample count mismatch")
        if self.rank == 0:
            if self.cfg.evaluation.get('fid_reference'):
                ref = np.load(self.cfg.evaluation.fid_reference)
                ref_mu, ref_sigma = ref['mu'], ref['sigma']
            else:
                real = FeatureMoments(self.device)
                real_indices = np.random.default_rng(2718).permutation(len(self.val))[:count][self.rank::self.world]
                for real_images, _ in self.loader(self.val, self.cfg.evaluation.batch_size, real_indices):
                    real.update(self.inception(((real_images.to(self.device).float()+1)*.5).clamp(0,1)))
                real_count, ref_mu, ref_sigma = real.finish()
                if real_count != count:
                    raise RuntimeError('Real-image FID count mismatch')
            score = float(frechet_distance(result[1], result[2], ref_mu, ref_sigma))
            self.log("generation", epoch=epoch, count=count, pytorch_fid_diagnostic=score)
            atomic_json(self.out / f"generation-epoch{epoch:03d}-{count}.json",
                        dict(epoch=epoch, count=count, pytorch_fid_diagnostic=score,
                             cfg=cfg.cfg, top_k=cfg.top_k, top_p=cfg.top_p,
                             note="Official ADM metrics are required for published VAR comparison"))
            if official:
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

    def train_prior(self):
        cfg = self.cfg.prior
        self.vae.eval().requires_grad_(False)
        if self.initialization == 'scratch':
            model = build_scratch_prior(self.vae, self.kind, self.cfg.seed+1001,
                                        depth=self.cfg.model.depth, num_classes=self.num_classes)
            if self.rank == 0:
                atomic_json(self.out/'prior-initialization.json', dict(initialization='scratch', seed=int(self.cfg.seed+1001), shared_sha256=shared_prior_digest(model)))
        else:
            model = LaserVAR(self.vae, depth=self.cfg.model.depth, num_classes=self.num_classes)
        model.to(self.device)
        optimizer = torch.optim.AdamW(optimizer_groups(model), lr=cfg.lr, betas=(.9,.95), weight_decay=cfg.weight_decay, fused=True)
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
        wrapped = DDP(model, device_ids=[self.device.index], broadcast_buffers=False)
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
            sampler = DistributedSampler(self.train, self.world, self.rank, shuffle=True, seed=self.cfg.seed)
            sampler.set_epoch(epoch)
            loader = DataLoader(self.train, batch_size=cfg.batch_size, sampler=sampler,
                                num_workers=self.cfg.data.workers, pin_memory=True, drop_last=True,
                                generator=torch.Generator().manual_seed(self.cfg.seed+epoch+self.rank))
            updates = len(loader)//cfg.accumulation
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
                             peak_memory_gib=torch.cuda.max_memory_allocated()/2**30)
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
                count = self.cfg.evaluation.full_samples if full else self.cfg.evaluation.preview_samples
                official = full and bool(self.cfg.evaluation.get('adm_reference'))
                self.generate(model, epoch+1, count, official=official)


def run(cfg):
    if int(cfg.data.image_size) % 16 or tuple(cfg.model.patch_nums)[-1] != int(cfg.data.image_size) // 16:
        raise ValueError('VAR scale endpoint must equal image_size / tokenizer downsampling (16)')
    os.environ.setdefault("TORCH_HOME", "/workspace/tmp/official-rqvae-eval-cache")
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
    torch.set_num_threads(4)
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    if "RANK" not in os.environ:
        raise ValueError("Launch this backend with torchrun, including for one GPU")
    dist.init_process_group("nccl", timeout=timedelta(hours=3))
    torch.manual_seed(cfg.seed + dist.get_rank())
    np.random.seed(cfg.seed + dist.get_rank())
    random.seed(cfg.seed + dist.get_rank())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
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
        out=Path(cfg.output_dir)
        out.mkdir(parents=True,exist_ok=True)
        atomic_json(out/f"failure-rank{dist.get_rank()}.json", dict(type=type(exc).__name__,message=str(exc),time=time.time()))
        if dist.get_rank() == 0:
            atomic_json(out/'status.json', dict(phase='failed', type=type(exc).__name__, message=str(exc),time=time.time()))
        raise
    finally:
        dist.destroy_process_group()
