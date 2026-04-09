"""
Encoder/Decoder tests with visualization.
Artifacts go to: tests/artifacts/codec/{encoder,decoder}/{random,celeba,imagenet,ffhq}/
"""
import os
import sys
from pathlib import Path

import pytest
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from tqdm import tqdm

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.encoder import Encoder
from src.models.decoder import Decoder
from src.checkpoint_io import load_lightning_module

# Output dirs
OUT = Path(__file__).resolve().parent / "artifacts" / "codec"
OUT.mkdir(parents=True, exist_ok=True)

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ============================================================================
# Helpers
# ============================================================================

def find_data(name: str) -> Path | None:
    """Find dataset by name. Checks env vars and common paths."""
    env_map = {"celeba": "CELEBA_DIR", "imagenet": "IMAGENET_DIR", "ffhq": "FFHQ_DIR"}
    paths = {
        "celeba": ["celeba/img_align_celeba", "celeba"],
        "imagenet": ["imagenet/val", "imagenet", "ILSVRC2012/val"],
        "ffhq": ["ffhq/images1024x1024", "ffhq/images", "ffhq"],
    }
    # Check env var first
    env = os.environ.get(env_map.get(name, ""))
    if env and Path(env).is_dir():
        return Path(env)
    # Check common paths
    for base in [Path("/home/xl598/Data"), ROOT.parent / "Data", ROOT.parent.parent / "Data"]:
        for sub in paths.get(name, [name]):
            p = base / sub
            if p.is_dir() and any(p.rglob("*.jpg")) or any(p.rglob("*.png")):
                return p
    return None


class ImageDataset(Dataset):
    """Simple flat image dataset."""
    def __init__(self, root: Path, transform, max_files: int = 64):
        self.transform = transform
        files = sorted([p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT])[:max_files]
        if not files:
            raise RuntimeError(f"No images in {root}")
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("RGB")
        return self.transform(img), str(self.files[i])


def make_loader(path: Path, size: int, max_files: int = 64, batch: int = 4) -> DataLoader:
    """Build a DataLoader for images at `path`."""
    tfm = Compose([Resize((size, size)), ToTensor(), Normalize((0.5,)*3, (0.5,)*3)])
    ds = ImageDataset(path, tfm, max_files)
    # Use spawn to avoid fork() deadlock warnings in multi-threaded context
    return DataLoader(ds, batch_size=batch, shuffle=False, num_workers=2, pin_memory=True, multiprocessing_context="spawn")


def to_img(x: torch.Tensor) -> torch.Tensor:
    """Normalize tensor from [-1,1] to [0,1] for saving."""
    return (x.clamp(-1, 1) + 1) / 2


def autorange(x: torch.Tensor) -> torch.Tensor:
    """Scale tensor to [0,1] using its own min/max."""
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-6)


def act_grid(feat: torch.Tensor, n: int = 16) -> torch.Tensor:
    """Make a grid of first `n` activation channels from a (B,C,H,W) tensor."""
    c = feat[0, :n].unsqueeze(1)  # (n, 1, H, W)
    c = (c - c.amin(dim=(2, 3), keepdim=True)) / (c.amax(dim=(2, 3), keepdim=True) - c.amin(dim=(2, 3), keepdim=True) + 1e-6)
    return torchvision.utils.make_grid(c, nrow=4)


def train_vqvae(model, imgs: torch.Tensor, steps: int = 25):
    """Quick overfit so recon isn't blank."""
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in tqdm(range(steps), desc="VQ-VAE quick train", leave=False):
        opt.zero_grad()
        rec, vq_loss, _ = model(imgs)
        loss = ((rec - imgs) ** 2).mean() + vq_loss
        loss.backward()
        opt.step()
    model.eval()


# ============================================================================
# Basic shape/grad tests
# ============================================================================

def test_encoder():
    """Check encoder output shape and gradients."""
    enc = Encoder(in_channels=3, num_hiddens=128, num_residual_blocks=2, num_residual_hiddens=32)
    x = torch.randn(2, 3, 256, 256, requires_grad=True)
    y = enc(x)
    assert y.shape == (2, 128, 64, 64), f"Bad shape: {y.shape}"
    y.mean().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_decoder():
    """Check decoder output shape and gradients."""
    dec = Decoder(in_channels=128, num_hiddens=128, num_residual_blocks=2, num_residual_hiddens=32)
    z = torch.randn(2, 128, 64, 64, requires_grad=True)
    y = dec(z)
    assert y.shape == (2, 3, 256, 256), f"Bad shape: {y.shape}"
    y.mean().backward()
    assert z.grad is not None and z.grad.abs().sum() > 0


# ============================================================================
# Random input visualization (no dataset needed)
# ============================================================================

def test_encoder_random():
    """Visualize encoder activations on random noise."""
    enc = Encoder(in_channels=3, num_hiddens=128, num_residual_blocks=2, num_residual_hiddens=32)
    enc.eval()
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        y = enc(x)
    out = OUT / "encoder" / "random"
    out.mkdir(parents=True, exist_ok=True)
    torchvision.utils.save_image(act_grid(y), str(out / "acts.png"))
    assert (out / "acts.png").exists()


def test_decoder_random():
    """Visualize decoder output on random latent."""
    dec = Decoder(in_channels=128, num_hiddens=128, num_residual_blocks=2, num_residual_hiddens=32)
    dec.eval()
    z = torch.randn(1, 128, 64, 64)
    with torch.no_grad():
        y = dec(z)
    out = OUT / "decoder" / "random"
    out.mkdir(parents=True, exist_ok=True)
    torchvision.utils.save_image(act_grid(z), str(out / "latents.png"))
    torchvision.utils.save_image(autorange(y), str(out / "output.png"))
    assert (out / "output.png").exists()


# ============================================================================
# Dataset visualization tests
# ============================================================================

@pytest.mark.parametrize("name,size", [("celeba", 256), ("imagenet", 256), ("ffhq", 1024)])
def test_encoder_dataset(name: str, size: int):
    """Visualize encoder activations on real images."""
    path = find_data(name)
    if path is None:
        pytest.skip(f"{name} not found")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    device = torch.device("cuda")
    enc = Encoder(in_channels=3, num_hiddens=128, num_residual_blocks=2, num_residual_hiddens=32).to(device).eval()
    loader = make_loader(path, size)

    out = OUT / "encoder" / name
    out.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for i, (imgs, _) in enumerate(loader):
            if i >= 4:
                break
            imgs = imgs.to(device)
            feats = enc(imgs)
            torchvision.utils.save_image(to_img(imgs), str(out / f"input_{i}.png"), nrow=2)
            torchvision.utils.save_image(act_grid(feats), str(out / f"acts_{i}.png"))

    assert (out / "input_0.png").exists()


@pytest.mark.parametrize("name,size", [("celeba", 256), ("imagenet", 256), ("ffhq", 1024)])
def test_decoder_dataset(name: str, size: int):
    """Visualize VQ-VAE reconstruction on real images."""
    path = find_data(name)
    if path is None:
        pytest.skip(f"{name} not found")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    device = torch.device("cuda")
    loader = make_loader(path, size)

    # Try to load checkpoint, otherwise train a tiny model
    from src.models.vqvae import VQVAE
    vqvae = None
    ckpt_dir = ROOT / "outputs" / "checkpoints"
    if ckpt_dir.exists():
        runs = sorted([d for d in ckpt_dir.iterdir() if d.name.startswith("run_")])
        if runs:
            ckpt = runs[-1] / "vqvae" / "last.ckpt"
            if ckpt.exists():
                try:
                    vqvae = load_lightning_module(
                        VQVAE,
                        ckpt,
                        map_location="cpu",
                        strict=False,
                        compute_fid=False,
                    )
                    vqvae = vqvae.to(device).eval()
                except Exception:
                    vqvae = None

    # Get first batch for potential training
    first_batch = next(iter(loader))
    imgs0, _ = first_batch
    imgs0 = imgs0.to(device)

    if vqvae is None:
        # Train a tiny VQ-VAE so recon isn't blank
        vqvae = VQVAE(
            in_channels=3, num_hiddens=128, num_embeddings=128, embedding_dim=32,
            num_residual_blocks=2, num_residual_hiddens=32, commitment_cost=0.25,
            decay=0.0, perceptual_weight=0.0, learning_rate=1e-3, beta=0.9, compute_fid=False,
        ).to(device)
        train_vqvae(vqvae, imgs0, steps=25)

    out = OUT / "decoder" / name
    out.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for i, (imgs, _) in enumerate(loader):
            if i >= 4:
                break
            imgs = imgs.to(device)
            recon, _, _ = vqvae(imgs)

            # Save input, recon (autoranged), and side-by-side compare
            torchvision.utils.save_image(to_img(imgs), str(out / f"input_{i}.png"), nrow=2)
            torchvision.utils.save_image(autorange(recon), str(out / f"recon_{i}.png"), nrow=2)

            # Side-by-side: input on top, recon on bottom
            compare = torch.cat([to_img(imgs), autorange(recon)], dim=2)
            torchvision.utils.save_image(compare, str(out / f"compare_{i}.png"), nrow=2)

            # Stats for debugging
            stats = f"min={recon.min():.3f} max={recon.max():.3f} mean={recon.mean():.3f} std={recon.std():.3f}\n"
            (out / f"stats_{i}.txt").write_text(stats)

    assert (out / "input_0.png").exists()
    assert (out / "recon_0.png").exists()
