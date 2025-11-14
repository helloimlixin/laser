import os
import sys
from pathlib import Path

import pytest
import torch
import lightning as pl
import torchvision
from omegaconf import OmegaConf

# Add the project root to sys.path so 'src.*' imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.dlvae import DLVAE
from src.data.celeba import CelebADataModule
from src.data.config import DataConfig


MODEL_CFG_PATH = Path(__file__).resolve().parents[1] / "configs" / "model" / "dlvae.yaml"
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts" / "dlvae_ckpt"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def _has_celeba_images() -> bool:
    # Try to resolve using same logic as CelebADataModule by probing common env/config
    candidates = [
        os.environ.get('CELEBA_DIR', ''),
        '/home/xl598/Data/celeba/img_align_celeba',
        '/home/xl598/Data/celeba',
    ]
    for c in candidates:
        if c and os.path.isdir(c):
            for root, _, files in os.walk(c):
                if any(f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')) for f in files):
                    return True
    return False


@pytest.mark.skipif(not _has_celeba_images(), reason="CelebA images not available (set CELEBA_DIR).")
def test_dlvae_tiny_train_on_celeba():
    torch.manual_seed(0)
    pl.seed_everything(0)

    # Minimal data config
    data_dir = os.environ.get('CELEBA_DIR', '')
    cfg = DataConfig(
        dataset='celeba',
        data_dir=data_dir,
        batch_size=8,
        num_workers=4,
        image_size=32,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        augment=False,
    )
    dm = CelebADataModule(cfg)
    dm.prepare_data()
    dm.setup()

    # Small model for speed
    model = DLVAE(
        in_channels=3,
        num_hiddens=32,
        num_embeddings=64,
        embedding_dim=16,
        sparsity_level=3,
        num_residual_blocks=1,
        num_residual_hiddens=16,
        commitment_cost=0.25,
        decay=0.99,
        perceptual_weight=0.0,
        learning_rate=1e-3,
        beta=0.9,
        compute_fid=False,
        omp_tolerance=1e-7,
        omp_debug=False,
        patch_size=2,
    )

    before_dict = model.bottleneck.dictionary.detach().clone()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    trainer = pl.Trainer(
        accelerator='cpu',
        max_epochs=1,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        limit_train_batches=2,
        limit_val_batches=1,
        deterministic=True,
    )
    trainer.fit(model, datamodule=dm)

    # Run a forward on a small batch to confirm shapes
    batch = next(iter(dm.train_dataloader()))
    x, _ = batch
    recon, dl_loss, coeffs = model(x)
    assert recon.shape == x.shape
    assert dl_loss.ndim == 0
    assert coeffs.shape[0] == model.bottleneck.num_embeddings

    after_dict = model.bottleneck.dictionary.detach().clone()
    # Ensure that at least some training occurred (weights changed)
    assert not torch.allclose(before_dict, after_dict)



def _load_model_cfg():
    if MODEL_CFG_PATH.exists():
        return OmegaConf.load(str(MODEL_CFG_PATH))
    return None


def _resolve_ckpt_candidate(path: str) -> str:
    if not path:
        return ""
    path = os.path.expanduser(path)
    if os.path.isdir(path):
        preferred = [
            "last.ckpt",
            "model.ckpt",
            "model.pth",
            "model.pt",
            "state_dict.ckpt",
            "state_dict.pt",
            "weights.pt",
            "weights.pth",
            "mp_rank_00_model_states.pt",
        ]
        for name in preferred:
            candidate = os.path.join(path, name)
            if os.path.isfile(candidate):
                return candidate
        for root, _, files in os.walk(path):
            for fname in files:
                if fname.endswith((".ckpt", ".pt", ".pth", ".bin")):
                    return os.path.join(root, fname)
    return path


def _extract_state_dict(payload):
    if isinstance(payload, dict):
        if isinstance(payload.get("state_dict"), dict):
            return payload["state_dict"]
        module_blob = payload.get("module")
        if isinstance(module_blob, dict):
            return module_blob.get("state_dict", module_blob)
        for key in ("model", "ema", "model_state_dict", "net", "generator"):
            maybe = payload.get(key)
            if isinstance(maybe, dict):
                return maybe
    return payload


def _load_module_state(module, ckpt_path: str, label: str):
    resolved = _resolve_ckpt_candidate(ckpt_path)
    if not resolved or not os.path.exists(resolved):
        raise FileNotFoundError(f"{label} checkpoint not found at {ckpt_path}")
    state = torch.load(resolved, map_location="cpu")
    state_dict = _extract_state_dict(state)
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"{label} checkpoint invalid: no state_dict")
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    return missing, unexpected


def _build_eval_dlvae(cfg_model):
    return DLVAE(
        in_channels=cfg_model.in_channels,
        num_hiddens=cfg_model.num_hiddens,
        num_embeddings=cfg_model.num_embeddings,
        embedding_dim=cfg_model.embedding_dim,
        sparsity_level=cfg_model.sparsity_level,
        patch_size=getattr(cfg_model, "patch_size", 1),
        num_residual_blocks=cfg_model.num_residual_blocks,
        num_residual_hiddens=cfg_model.num_residual_hiddens,
        commitment_cost=cfg_model.commitment_cost,
        decay=cfg_model.decay,
        perceptual_weight=cfg_model.perceptual_weight,
        learning_rate=1e-3,
        beta=0.9,
        compute_fid=False,
        omp_tolerance=getattr(cfg_model, "omp_tolerance", 1e-7),
        omp_debug=getattr(cfg_model, "omp_debug", False),
    )


def _resolve_dlvae_ckpt(cfg_model):
    return os.environ.get("DLVAE_CKPT", cfg_model.get("dlvae_ckpt", ""))


def _unnormalize(tensor, mean, std):
    device = tensor.device
    dtype = tensor.dtype
    mean_t = torch.tensor(mean, device=device, dtype=dtype).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=device, dtype=dtype).view(1, -1, 1, 1)
    return (tensor * std_t + mean_t).clamp(0.0, 1.0)


def _save_recon_grid(original, recon, mean, std, path: Path):
    orig_vis = _unnormalize(original, mean, std).cpu()
    recon_vis = _unnormalize(recon, mean, std).cpu()
    tiles = []
    for o, r in zip(orig_vis, recon_vis):
        tiles.extend([o, r])
    if not tiles:
        return
    grid = torch.stack(tiles, dim=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    torchvision.utils.save_image(grid, str(path), nrow=2)





@pytest.mark.skipif(not _has_celeba_images(), reason="CelebA images not available (set CELEBA_DIR).")
def test_dlvae_checkpoint_reconstruction_visuals():
    cfg_model = _load_model_cfg()
    if cfg_model is None:
        pytest.skip("dlvae config is missing.")

    torch.manual_seed(0)
    pl.seed_everything(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dlvae = _build_eval_dlvae(cfg_model).to(device)
    dlvae.eval()

    data_dir = os.environ.get('CELEBA_DIR', '')
    data_cfg = DataConfig(
        dataset='celeba',
        data_dir=data_dir,
        batch_size=8,
        num_workers=4,
        image_size=cfg_model.get('image_size', 128),
        mean=(0.5, 0.5, 0.5),
        std=(1.0, 1.0, 1.0),
        augment=False,
    )
    dm = CelebADataModule(data_cfg)
    dm.prepare_data()
    dm.setup()
    val_loader = dm.val_dataloader()
    batch = next(iter(val_loader))
    x = batch[0][:4].to(device)

    dlvae_ckpt = _resolve_dlvae_ckpt(cfg_model)
    if not dlvae_ckpt or not os.path.exists(_resolve_ckpt_candidate(dlvae_ckpt)):
        pytest.skip("DLVAE checkpoint is not available.")
    _load_module_state(dlvae, dlvae_ckpt, "DLVAE")
    recon_source = "dlvae"

    with torch.no_grad():
        recon, _, _ = dlvae(x)

    mse = torch.mean((recon - x) ** 2).item()
    assert torch.isfinite(torch.tensor(mse, device=device))

    artifact_path = ARTIFACT_DIR / f"celeba_recon_{recon_source}.png"
    _save_recon_grid(x.detach(), recon.detach(), data_cfg.mean, data_cfg.std, artifact_path)
    assert artifact_path.exists(), f"Failed to write recon grid to {artifact_path}"
