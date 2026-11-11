"""Bounded learning pilot for original OMP supports and continuous coefficients.

Run through train.py. Images are encoded once for a fixed-batch learning check;
this is not a full ImageNet training run or a generation-quality benchmark.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import time
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.nn import functional as F

from src.continuous_sparse_codec import ContinuousSparseCodes, continuous_image_codes, continuous_latents
from src.models.continuous_compound_rq import build_continuous_compound_model, continuous_pair_loss


def file_digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def pilot_images(archive, classes, image_size, *, index_path=None):
    """One fit and one disjoint held-out image from evenly spaced train classes."""
    from torchvision import transforms
    transform = transforms.Compose([transforms.Resize(288, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(image_size), transforms.ToTensor(), transforms.Normalize(.5, .5)])
    fit, heldout, labels, identities = [], [], [], []
    stat = Path(archive).stat()
    identity = dict(path=str(Path(archive).resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    with tarfile.open(archive, 'r:') as outer:
        if index_path is not None and Path(index_path).is_file():
            index = json.loads(Path(index_path).read_text())
            if index['archive'] != identity:
                raise ValueError('Pilot archive changed since its class offsets were cached')
            members = []
            for row in index['classes']:
                member = tarfile.TarInfo(row['name'])
                member.size, member.offset_data = row['size'], row['offset_data']
                members.append(member)
        else:
            members = sorted([m for m in outer.getmembers() if m.isfile() and m.name.endswith('.tar')],
                             key=lambda m: m.name)
            if index_path is not None:
                write_json(index_path, dict(archive=identity, classes=[
                    dict(name=m.name, size=m.size, offset_data=m.offset_data) for m in members]))
        if len(members) != 1000 or not 1 <= classes <= 1000:
            raise ValueError('Expected the official nested ImageNet-1k training archive')
        for label in torch.linspace(0, 999, classes).round().long().tolist():
            member = members[label]
            with outer.extractfile(member) as stream, tarfile.open(fileobj=stream, mode='r:') as inner:
                # Archive order is deterministic. Do not enumerate thousands
                # of unused JPEG headers on an object-backed filesystem.
                images = []
                for member_image in inner:
                    if member_image.isfile() and member_image.name.endswith('.JPEG'):
                        images.append(member_image)
                        if len(images) == 2:
                            break
                if len(images) != 2:
                    raise ValueError('Each pilot class requires two distinct JPEGs')
                decoded = []
                for image in images:
                    with Image.open(io.BytesIO(inner.extractfile(image).read())) as pixels:
                        decoded.append(transform(pixels.convert('RGB')))
                fit.append(decoded[0]); heldout.append(decoded[1]); labels.append(label)
                identities.append(dict(label=label, wnid=member.name.removesuffix('.tar'),
                                       fit=images[0].name, heldout=images[1].name))
    return torch.stack(fit), torch.stack(heldout), torch.tensor(labels), identities


def run(cfg):
    if cfg.get('mode', 'pilot') == 'train':
        from src.training.continuous_compound_train import run as train
        return train(cfg)
    if cfg.stage != 'stage2' or cfg.pilot.steps < 1 or cfg.pilot.microbatch < 1:
        raise ValueError('A stage2 pilot needs positive steps and microbatch')
    if cfg.target_mode != 'original_omp_continuous' or cfg.auxiliary_loss_terms != 0:
        raise ValueError('This pilot uses original continuous OMP targets without auxiliary losses')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('The bounded fixed-batch pilot runs on one device; use the shared recipe for DDP')
    device = torch.device(cfg.pilot.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; override pilot.device=cpu')
    torch.set_num_threads(int(cfg.pilot.cpu_threads))
    torch.manual_seed(int(cfg.seed))
    synthetic = bool(cfg.data.synthetic)
    output = Path(cfg.output_dir)
    local = Path(cfg.runtime_base)
    if (output / 'result.json').exists():
        raise FileExistsError(f'Completed pilot already exists: {output}; choose a new output_dir')
    if (output / 'metrics.jsonl').exists() and cfg.resume is None:
        raise FileExistsError('Pilot metrics already exist; resume its checkpoint or choose a new output_dir')

    # Check real inputs before creating any run artifacts or constructing a model.
    source_hash = None
    if not synthetic:
        checkpoint = Path(cfg.tokenizer.checkpoint)
        archive = Path(cfg.data.train_archive)
        if not checkpoint.is_file() or not archive.is_file():
            raise FileNotFoundError('The original stage1 checkpoint and ImageNet training archive are required')
        source_hash = file_digest(checkpoint)
        if source_hash != cfg.tokenizer.sha256:
            raise ValueError('Source stage1 checkpoint SHA256 mismatch')

    output.mkdir(parents=True, exist_ok=True); local.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output / 'recipe.yaml', resolve=True)
    if synthetic:
        channels = int(cfg.arch.input_embed_dim)
        height, width, _ = cfg.arch.block_size
        aux = SimpleNamespace(dictionary=F.normalize(torch.randn(channels, cfg.tokenizer.num_atoms,
            device=device), dim=0), sparsity_level=int(cfg.tokenizer.depth),
            encoder=lambda x: x, quant_conv=lambda x: x)
        fit_images = torch.randn(cfg.pilot.classes, channels, height, width)
        heldout_images = torch.randn_like(fit_images)
        labels = torch.arange(len(fit_images)) % int(cfg.arch.vocab_size_cond)
        identities = []
    else:
        from src.models.laser_tokenizer_adapter import FrozenLaserTokenizer
        aux = FrozenLaserTokenizer(checkpoint, num_atoms=int(cfg.tokenizer.num_atoms),
            coeff_vocab_size=None, coeff_max=None, coeff_scales=list(cfg.tokenizer.coefficient_scales),
            sparsity_level=int(cfg.tokenizer.depth)).to(device)
        fit_images, heldout_images, labels, identities = pilot_images(
            archive, int(cfg.pilot.classes), int(cfg.data.image_size),
            index_path=local / 'archive-class-index.json')
    write_json(output / 'images.json', dict(synthetic=synthetic, images=identities,
        train_images=len(fit_images), heldout_images=len(heldout_images), overlapping_images=0,
        preprocessing='Resize288 Lanczos; CenterCrop256; no augmentation' if not synthetic else None))
    with torch.inference_mode():
        fit = continuous_image_codes(aux, fit_images.to(device),
            encoder_batch_size=int(cfg.targets.encoder_batch_size))
        heldout = continuous_image_codes(aux, heldout_images.to(device),
            encoder_batch_size=int(cfg.targets.encoder_batch_size))
    # OMP targets are constants, not inference tensors saved by trainable layers.
    fit = ContinuousSparseCodes(fit.atoms.clone(), fit.coefficients.clone())
    heldout = ContinuousSparseCodes(heldout.atoms.clone(), heldout.coefficients.clone())
    labels = labels.to(device)
    fit.validate(int(cfg.tokenizer.num_atoms), int(cfg.tokenizer.depth))
    heldout.validate(int(cfg.tokenizer.num_atoms), int(cfg.tokenizer.depth))
    target_copy = fit.coefficients.clone()
    target_atoms = fit.atoms.clone()
    dictionary_copy = aux.dictionary.detach().clone()
    write_json(output / 'targets.json', dict(source_checkpoint_sha256=source_hash,
        synthetic=synthetic, shape=list(fit.shape), original_deterministic_omp=True,
        coefficient_representation='physical_fp32', coefficient_quantization=False,
        coefficient_clipping=False, coefficient_noise=False, beam_search=False,
        coefficient_abs_max=float(fit.coefficients.abs().max())))

    torch.manual_seed(int(cfg.seed))
    model = build_continuous_compound_model(cfg.arch, num_atoms=int(cfg.tokenizer.num_atoms),
        coefficient_scales=list(cfg.tokenizer.coefficient_scales), components=int(cfg.density.components),
        min_std=float(cfg.density.min_std)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.optimizer.lr),
        betas=tuple(cfg.optimizer.betas), weight_decay=float(cfg.optimizer.weight_decay))
    use_amp = bool(cfg.pilot.amp) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    signature = dict(arch=OmegaConf.to_container(model.config, resolve=True),
        source_checkpoint_sha256=source_hash, synthetic=synthetic, images=identities,
        seed=int(cfg.seed), classes=int(cfg.pilot.classes), microbatch=int(cfg.pilot.microbatch),
        tokenizer=OmegaConf.to_container(cfg.tokenizer, resolve=True),
        density=OmegaConf.to_container(cfg.density, resolve=True), atom_weight=float(cfg.atom_ce_weight),
        optimizer=OmegaConf.to_container(cfg.optimizer, resolve=True), amp=use_amp,
        precision='native_fp16_amp' if use_amp else 'float32')
    start_step = 0
    if cfg.resume is not None:
        payload = torch.load(cfg.resume, map_location=device, weights_only=False)
        if payload['signature'] != signature:
            raise ValueError('Pilot resume requires the same model, targets, optimizer, and batch')
        model.load_state_dict(payload['state_dict'], strict=True)
        optimizer.load_state_dict(payload['optimizer'])
        scaler.load_state_dict(payload['scaler'])
        start_step = int(payload['step'])
        if start_step >= int(cfg.pilot.steps):
            raise ValueError('Resuming requires pilot.steps greater than the saved step')
        torch.set_rng_state(payload['rng_cpu'].cpu())
        if device.type == 'cuda':
            torch.cuda.set_rng_state(payload['rng_cuda'].cpu(), device)
        del payload

    def step_loss(codes, mode):
        result = dict(loss=0., atom_nll=0., coefficient_nll=0., joint_nll=0.)
        for offset in range(0, len(codes.atoms), int(cfg.pilot.microbatch)):
            end = offset + int(cfg.pilot.microbatch)
            part = ContinuousSparseCodes(codes.atoms[offset:end], codes.coefficients[offset:end])
            weight = len(part.atoms) / len(codes.atoms)
            prediction = model(part, aux, labels[offset:end], amp=use_amp)
            loss, metrics = continuous_pair_loss(prediction, part, model.coefficient_scales,
                                                  atom_weight=float(cfg.atom_ce_weight))
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite continuous compound likelihood')
            if mode == 'train':
                scaler.scale(loss * weight).backward()
            result['loss'] += float(loss.detach()) * weight
            for name, value in metrics.items():
                result[name] += float(value) * weight
        return result

    model.eval()
    with torch.no_grad():
        initial = dict(fit=step_loss(fit, 'eval'), heldout=step_loss(heldout, 'eval'))
    write_json(output / 'initial.json', initial)
    started = time.monotonic()
    for step in range(start_step + 1, int(cfg.pilot.steps) + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        rng_cpu = torch.get_rng_state()
        rng_cuda = torch.cuda.get_rng_state(device) if device.type == 'cuda' else None
        metrics = step_loss(fit, 'train')

        def replay_backward():
            nonlocal metrics
            torch.set_rng_state(rng_cpu)
            if rng_cuda is not None:
                torch.cuda.set_rng_state(rng_cuda, device)
            metrics = step_loss(fit, 'train')

        from runtime.amp_replay import recoverable_amp_step
        norm, retries = recoverable_amp_step(model.parameters(), optimizer, scaler,
            max_norm=float(cfg.optimizer.max_grad_norm), replay_backward=replay_backward)
        row = dict(step=step, **metrics, gradient_norm=float(norm), amp_retries=retries,
                   seconds=time.monotonic() - started)
        if device.type == 'cuda':
            row['peak_memory_gib'] = torch.cuda.max_memory_allocated(device) / 1024 ** 3
        with (output / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        write_json(output / 'status.json', dict(phase='pilot', **row))
        print(json.dumps(row), flush=True)

    model.eval()
    with torch.no_grad():
        final = dict(fit=step_loss(fit, 'eval'), heldout=step_loss(heldout, 'eval'))
        generator = torch.Generator(device=device).manual_seed(int(cfg.seed) + 1)
        samples = model.sample_sparse(min(4, len(labels)), aux, labels[:4], amp=use_amp,
            generator=generator, **OmegaConf.to_container(cfg.sampling))
        samples.validate(int(cfg.tokenizer.num_atoms), int(cfg.tokenizer.depth))
        torch.save(dict(atoms=samples.atoms.cpu(), coefficients=samples.coefficients.cpu()), output / 'sample-codes.pt')
        if not synthetic:
            from torchvision.utils import save_image
            latent = continuous_latents(samples, aux.dictionary).permute(0, 3, 1, 2).contiguous()
            images = aux.decoder(aux.post_quant_conv(latent)).clamp(-1, 1)
            save_image(images.mul(.5).add(.5), output / 'samples.png', nrow=4)
    assert torch.equal(fit.atoms, target_atoms) and torch.equal(fit.coefficients, target_copy)
    assert torch.equal(aux.dictionary, dictionary_copy)
    if cfg.pilot.save_checkpoint:
        checkpoint_path = local / 'last.pt'
        temporary = local / 'last.pending'
        torch.save(dict(state_dict=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
            step=int(cfg.pilot.steps), signature=signature, rng_cpu=torch.get_rng_state(),
            rng_cuda=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None), temporary)
        temporary.replace(checkpoint_path)
        write_json(output / 'checkpoint.json', dict(path=str(checkpoint_path), step=int(cfg.pilot.steps)))
    report = dict(passed=True, experiment='fixed_batch_learning_pilot', synthetic=synthetic,
        steps=int(cfg.pilot.steps), initial=initial, final=final, source_checkpoint_sha256=source_hash,
        trainable_parameters=sum(p.numel() for p in model.parameters()),
        targets_and_dictionary_unchanged=True, sampled_coefficients_finite=True,
        generation_fid_measured=False, full_imagenet_training=False,
        density_nll_comparable_to_bin_ce=False, seconds=time.monotonic() - started)
    write_json(output / 'result.json', report)
    write_json(output / 'status.json', dict(phase='completed', step=int(cfg.pilot.steps),
        seconds=report['seconds'], result=str(output / 'result.json')))
    print(json.dumps(report), flush=True)
