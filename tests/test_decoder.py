import pytest
import torch
import sys
import os
from pathlib import Path
from typing import Optional

import torchvision
import multiprocessing as mp
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.encoder import Encoder
from models.decoder import Decoder


ARTIFACTS_DIR = Path(__file__).resolve().parent / 'artifacts'
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def _dir_has_images(path: Path) -> bool:
    for ext in IMG_EXTS:
        try:
            next(path.rglob(f'*{ext}'))
            return True
        except StopIteration:
            continue
    return False


def _find_dataset_dir(env_var: str, relative_subpaths: list[str], absolute_candidates: list[Path]) -> Optional[Path]:
    candidates: list[Path] = []

    # Prefer relative project data first: ../../Data then ../../../Data
    here = Path(__file__).resolve()
    base2 = here.parents[2] / 'Data'
    base3 = here.parents[3] / 'Data'
    for sub in relative_subpaths:
        candidates.append(base2 / sub)
        candidates.append(base3 / sub)

    # Then environment variable
    env = os.environ.get(env_var)
    if env:
        candidates.append(Path(env))

    # Finally absolute defaults on this machine
    candidates.extend(absolute_candidates)

    for cand in candidates:
        if cand.exists() and cand.is_dir() and _dir_has_images(cand):
            return cand
    return None


def _find_celeba_dir() -> Optional[Path]:
    relative_subpaths = [
        'celeba/img_align_celeba',
        'celeba',
    ]
    absolute_candidates = [
        Path('/home/xl598/Data/celeba/img_align_celeba'),
        Path('/home/xl598/Data/celeba'),
    ]
    return _find_dataset_dir('CELEBA_DIR', relative_subpaths, absolute_candidates)


class _FlatImageDataset(Dataset):
    def __init__(self, root: str | Path, transform=None, max_files: int | None = None):
        self.root = Path(root)
        self.transform = transform
        paths: list[Path] = []
        for p in self.root.rglob('*'):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                paths.append(p)
        paths.sort()
        if max_files is not None:
            paths = paths[:max_files]
        if not paths:
            raise RuntimeError(f'No images found under {self.root}')
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        with Image.open(path) as img:
            img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, str(path)


def _make_activation_grid(tensor: torch.Tensor, max_channels: int = 16) -> torch.Tensor:
    assert tensor.dim() == 4, 'Expected (B, C, H, W)'
    sample = tensor[0:1]
    channels = min(sample.shape[1], max_channels)
    maps = sample[:, :channels].clone().detach()
    maps = (maps - maps.amin(dim=(2, 3), keepdim=True)) / (
        (maps.amax(dim=(2, 3), keepdim=True) - maps.amin(dim=(2, 3), keepdim=True)).clamp_min(1e-6)
    )
    maps = maps.squeeze(0)
    grid = torchvision.utils.make_grid(maps.unsqueeze(1), nrow=4)
    if grid.shape[0] == 1:
        grid = grid.repeat(3, 1, 1)
    return grid


def _find_latest_vqvae_ckpt() -> Optional[Path]:
    ckpt_root = Path(__file__).resolve().parent.parent / 'outputs' / 'checkpoints'
    if not ckpt_root.exists():
        return None
    run_dirs = [p for p in ckpt_root.iterdir() if p.is_dir() and p.name.startswith('run_')]
    if not run_dirs:
        return None
    # pick latest by name (timestamp suffix)
    run_dirs.sort()
    vq_dir = run_dirs[-1] / 'vqvae'
    if not vq_dir.exists():
        return None
    last = vq_dir / 'last.ckpt'
    if last.exists():
        return last
    ckpts = sorted(vq_dir.glob('*.ckpt'))
    return ckpts[-1] if ckpts else None


def _load_encoder_decoder_from_ckpt(device: torch.device) -> Optional[tuple[Encoder, Decoder]]:
    ckpt = _find_latest_vqvae_ckpt()
    if ckpt is None:
        return None
    data = torch.load(str(ckpt), map_location='cpu')
    state = data.get('state_dict', data)

    enc = Encoder(in_channels=3, num_hiddens=128, num_residual_blocks=2, num_residual_hiddens=32).to(device)
    dec = Decoder(in_channels=128, num_hiddens=128, num_residual_blocks=2, num_residual_hiddens=32).to(device)

    # Extract matching sub-state dicts
    enc_state = {k.split('encoder.', 1)[1]: v for k, v in state.items() if k.startswith('encoder.')}
    dec_state = {k.split('decoder.', 1)[1]: v for k, v in state.items() if k.startswith('decoder.')}

    missing_e, unexpected_e = enc.load_state_dict(enc_state, strict=False)
    missing_d, unexpected_d = dec.load_state_dict(dec_state, strict=False)
    # If nothing loaded, consider failure
    if len(enc_state) == 0 or len(dec_state) == 0:
        return None
    enc.eval()
    dec.eval()
    return enc, dec


def test_shapes_and_grads():
    torch.manual_seed(0)
    model = Decoder(in_channels=128, num_hiddens=128, num_residual_blocks=2, num_residual_hiddens=32)

    x = torch.randn(4, 128, 64, 64, requires_grad=True)
    y = model(x)

    assert y.shape == (4, 3, 256, 256)

    y.mean().backward()
    has_param_grads = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_param_grads
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_vis_random():
    torch.manual_seed(0)
    model = Decoder(in_channels=128, num_hiddens=128, num_residual_blocks=2, num_residual_hiddens=32)
    model.eval()

    z = torch.randn(1, 128, 64, 64)
    with torch.no_grad():
        y = model(z)

    # Save inputs (latent activations) and decoder outputs
    lat_grid = _make_activation_grid(z, max_channels=16)
    out_grid = torchvision.utils.make_grid(y.clamp(-1, 1), nrow=1, normalize=True, value_range=(-1, 1))

    out_dir = ARTIFACTS_DIR
    torchvision.utils.save_image(lat_grid, str(out_dir / 'decoder_latents.png'))
    torchvision.utils.save_image(out_grid, str(out_dir / 'decoder_outputs.png'))

    assert (out_dir / 'decoder_latents.png').exists()
    assert (out_dir / 'decoder_outputs.png').exists()


def test_celeba():
    celeba_path = _find_celeba_dir()
    if celeba_path is None:
        pytest.skip('CelebA not found. Set CELEBA_DIR or put images in ../../Data/celeba/img_align_celeba or /home/xl598/Data/celeba/img_align_celeba')

    if not torch.cuda.is_available():
        pytest.skip('CUDA is required for this test')

    device = torch.device('cuda')

    transform = Compose([
        Resize((256, 256)),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = _FlatImageDataset(celeba_path, transform=transform, max_files=64)
    ctx = mp.get_context('spawn')
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        multiprocessing_context=ctx,
    )

    loaded = _load_encoder_decoder_from_ckpt(device)
    if loaded is not None:
        enc, dec = loaded
    else:
        enc = Encoder(in_channels=3, num_hiddens=128, num_residual_blocks=2, num_residual_hiddens=32).to(device)
        dec = Decoder(in_channels=128, num_hiddens=128, num_residual_blocks=2, num_residual_hiddens=32).to(device)
        enc.eval()
        dec.eval()

    out_dir = ARTIFACTS_DIR / 'celeba_dec'
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            feats = enc(images)
            recons = dec(feats)

            img_grid = torchvision.utils.make_grid(images[:4].cpu(), nrow=2, normalize=True, value_range=(-1, 1))
            torchvision.utils.save_image(img_grid, str(out_dir / f'inputs_{i:04d}.png'))

            rec_grid = torchvision.utils.make_grid(recons[:4].cpu(), nrow=2, normalize=True, value_range=(-1, 1))
            torchvision.utils.save_image(rec_grid, str(out_dir / f'recons_{i:04d}.png'))

            saved += 1
            if saved >= 4:
                break

    assert (out_dir / 'inputs_0000.png').exists()
    assert (out_dir / 'recons_0000.png').exists()


# ----------------------------
# Additional datasets (ImageNet, FFHQ) - slow GPU tests
# ----------------------------

def _find_imagenet_dir() -> Optional[Path]:
    relative_subpaths = [
        'imagenet/val',
        'imagenet',
        'ILSVRC2012/val',
        'ILSVRC2012',
    ]
    absolute_candidates = [
        Path('/home/xl598/Data/imagenet/val'),
        Path('/home/xl598/Data/imagenet'),
        Path('/home/xl598/Data/ILSVRC2012/val'),
        Path('/home/xl598/Data/ILSVRC2012'),
    ]
    return _find_dataset_dir('IMAGENET_DIR', relative_subpaths, absolute_candidates)


def _find_ffhq_dir() -> Optional[Path]:
    relative_subpaths = [
        'ffhq/images1024x1024',
        'ffhq/images',
        'ffhq',
        'ffhq-dataset/images1024x1024',
        'ffhq-dataset/images',
    ]
    absolute_candidates = [
        Path('/home/xl598/Data/ffhq/images1024x1024'),
        Path('/home/xl598/Data/ffhq/images'),
        Path('/home/xl598/Data/ffhq'),
        Path('/home/xl598/Data/ffhq-dataset/images1024x1024'),
        Path('/home/xl598/Data/ffhq-dataset/images'),
    ]
    return _find_dataset_dir('FFHQ_DIR', relative_subpaths, absolute_candidates)


def _run_dir_viz(dataset_path: Path, out_subdir: str, image_size: int = 256, max_files: int = 64) -> None:
    if not torch.cuda.is_available():
        pytest.skip('CUDA is required for this test')

    device = torch.device('cuda')
    transform = Compose([
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = _FlatImageDataset(dataset_path, transform=transform, max_files=max_files)
    ctx = mp.get_context('spawn')
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        multiprocessing_context=ctx,
    )

    loaded = _load_encoder_decoder_from_ckpt(device)
    if loaded is not None:
        enc, dec = loaded
    else:
        enc = Encoder(in_channels=3, num_hiddens=128, num_residual_blocks=2, num_residual_hiddens=32).to(device)
        dec = Decoder(in_channels=128, num_hiddens=128, num_residual_blocks=2, num_residual_hiddens=32).to(device)
        enc.eval()
        dec.eval()

    out_dir = ARTIFACTS_DIR / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            feats = enc(images)
            recons = dec(feats)

            img_grid = torchvision.utils.make_grid(images[:4].cpu(), nrow=2, normalize=True, value_range=(-1, 1))
            torchvision.utils.save_image(img_grid, str(out_dir / f'inputs_{i:04d}.png'))

            rec_grid = torchvision.utils.make_grid(recons[:4].cpu(), nrow=2, normalize=True, value_range=(-1, 1))
            torchvision.utils.save_image(rec_grid, str(out_dir / f'recons_{i:04d}.png'))

            saved += 1
            if saved >= 4:
                break

    assert (out_dir / 'inputs_0000.png').exists()
    assert (out_dir / 'recons_0000.png').exists()


def test_imagenet():
    path = _find_imagenet_dir()
    if path is None:
        pytest.skip('ImageNet not found. Set IMAGENET_DIR or place under ../../Data/imagenet or /home/xl598/Data/imagenet')
    _run_dir_viz(path, out_subdir='imagenet_dec', image_size=256)


def test_ffhq():
    path = _find_ffhq_dir()
    if path is None:
        pytest.skip('FFHQ not found. Set FFHQ_DIR or place under ../../Data/ffhq or /home/xl598/Data/ffhq')
    _run_dir_viz(path, out_subdir='ffhq_dec', image_size=1024)


