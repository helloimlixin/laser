#!/usr/bin/env python3
"""Distributed reconstruction FID for an upstream LASER checkpoint."""

from __future__ import annotations
import argparse, importlib.util, json, logging, os, sys
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset
from torchvision import datasets
from torchmetrics.image.fid import FrechetInceptionDistance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_official_rqtransformer_laser_stage2 import LaserAux, val_image_transform


class FlatImages(Dataset):
    """Recursively load an image directory without requiring class folders."""

    def __init__(self, root: Path, transform):
        self.files = sorted(
            path
            for path in root.rglob("*")
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )
        if not self.files:
            raise ValueError(f"no images found below {root}")
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with Image.open(self.files[index]) as image:
            return self.transform(image.convert("RGB")), 0


class IndexedSubset(Dataset):
    """Return source images together with their row-aligned cache index."""

    def __init__(self, dataset: Dataset, count: int):
        self.dataset = dataset
        self.count = int(count)

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        return image, index


def native_fid_model(device):
    # Import the repository's FID Inception module directly. Importing the
    # rqvae.metrics package eagerly loads unrelated CLIP/tokenizer metrics.
    inception_path = (ROOT / "third_party" / "rq-vae-transformer" /
                      "rqvae" / "metrics" / "inception.py")
    spec = importlib.util.spec_from_file_location("laser_native_fid_inception", inception_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    class InceptionWrapper(module.InceptionV3):
        def forward(self, inp):
            pred = super().forward(inp)[0]
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = torch.nn.functional.adaptive_avg_pool2d(pred, (1, 1))
            return pred.reshape(pred.shape[0], -1)

    block = module.InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    return InceptionWrapper([block]).to(device).eval()


def native_frechet(real_sum, real_cross, fake_sum, fake_cross, count):
    import numpy as np
    from scipy import linalg
    n = float(count)
    mu_r = (real_sum / n).cpu().numpy()
    mu_f = (fake_sum / n).cpu().numpy()
    cov_r = ((real_cross - torch.outer(real_sum, real_sum) / n) / (n - 1)).cpu().numpy()
    cov_f = ((fake_cross - torch.outer(fake_sum, fake_sum) / n) / (n - 1)).cpu().numpy()
    diff = mu_r - mu_f
    covmean, _ = linalg.sqrtm(cov_r.dot(cov_f), disp=False)
    if not np.isfinite(covmean).all():
        logging.warning("FID covariance product is singular; adding 1e-6 to diagonals")
        offset = np.eye(cov_r.shape[0]) * 1e-6
        covmean = linalg.sqrtm((cov_r + offset).dot(cov_f + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f"Imaginary component {np.max(np.abs(covmean.imag))}")
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(cov_r) + np.trace(cov_f) - 2 * np.trace(covmean))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--token-cache", type=Path, default=None,
        help="Decode row-aligned quantized cache entries instead of continuous coefficients",
    )
    p.add_argument(
        "--dataset", choices=("imagenet", "celebahq", "ffhq"), default="imagenet"
    )
    p.add_argument("--num-images", type=int, default=50_000)
    p.add_argument("--num-atoms", type=int, default=16_384)
    p.add_argument("--coeff-vocab-size", type=int, default=2_048)
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--backend", choices=("torchmetrics", "native"), default="torchmetrics")
    p.add_argument("--wandb-entity", default="helloimlixin-rutgers")
    p.add_argument("--wandb-project", default="laser")
    p.add_argument("--wandb-id", default=None)
    p.add_argument("--wandb-name", default=None)
    p.add_argument(
        "--wandb-mode", choices=("online", "disabled"), default="disabled"
    )
    args = p.parse_args()
    if args.num_images <= 0:
        p.error("--num-images must be positive")
    if args.wandb_mode == "online" and (not args.wandb_id or not args.wandb_name):
        p.error("--wandb-id and --wandb-name are required in online mode")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=45))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if args.dataset in {"celebahq", "ffhq"}:
        full_dataset = FlatImages(args.data, transform=val_image_transform())
        split_name = f"{args.dataset}_full_{args.num_images}"
    else:
        full_dataset = datasets.ImageFolder(
            args.data / "val", transform=val_image_transform()
        )
        split_name = "imagenet_val"
    if len(full_dataset) < args.num_images:
        raise ValueError(
            f"{args.dataset} has {len(full_dataset):,} images, fewer than requested "
            f"{args.num_images:,}"
        )
    cache_payload = None
    cache_meta = None
    if args.token_cache is not None:
        cache_payload = torch.load(
            args.token_cache, map_location="cpu", weights_only=True, mmap=True
        )
        cache_meta = dict(cache_payload["meta"])
        if cache_meta.get("dataset") != args.dataset:
            raise ValueError(
                f"token cache dataset mismatch: {cache_meta.get('dataset')!r} != {args.dataset!r}"
            )
        if len(cache_payload["atoms"]) < args.num_images:
            raise ValueError(
                f"token cache has {len(cache_payload['atoms']):,} rows, fewer than "
                f"the requested {args.num_images:,}"
            )
        dataset = IndexedSubset(full_dataset, args.num_images)
    else:
        dataset = Subset(full_dataset, range(args.num_images))
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False) if world > 1 else None
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, shuffle=False,
                        num_workers=8, pin_memory=True, persistent_workers=True)
    # Native DictionaryLearning.forward() leaves its OMP coefficients
    # unbounded. Disable the stage-2 tokenization clamp for reconstruction FID.
    aux = LaserAux(
        args.checkpoint,
        int(cache_meta["num_atoms"]) if cache_meta is not None else args.num_atoms,
        int(cache_meta["coeff_vocab_size"]) if cache_meta is not None else args.coeff_vocab_size,
        float(cache_meta["coeff_max"]) if cache_meta is not None else 20.0,
        float(cache_meta.get("coeff_scale", 1.0)) if cache_meta is not None else 1.0,
        attn_resolutions=((16,) if args.dataset in {"celebahq", "ffhq"} else (8,)),
        coeff_scales=(cache_meta.get("coeff_scales") if cache_meta is not None else None),
        clamp_coeffs=cache_meta is not None,
        coeff_bin_centers=(
            cache_meta.get("coeff_bin_centers") if cache_meta is not None else None
        ),
    ).to(device).eval()
    metric = None
    inception = None
    if args.backend == "torchmetrics":
        metric = FrechetInceptionDistance(feature=2048, normalize=True,
                                          sync_on_compute=world > 1).to(device)
    else:
        inception = native_fid_model(device)
        real_sum = torch.zeros(2048, device=device, dtype=torch.float64)
        fake_sum = torch.zeros_like(real_sum)
        real_cross = torch.zeros(2048, 2048, device=device, dtype=torch.float64)
        fake_cross = torch.zeros_like(real_cross)
    seen = 0
    with torch.no_grad():
        for batch_idx, (images, row_indices) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            if cache_payload is None:
                atoms, coeffs = aux.encode_sparse_components(images)
                vectors = aux.dictionary.t()[atoms.long()]
                # encode_sparse_components returns stage-2-normalized coefficients;
                # restore the physical LASER values for native stage-1 reconstruction.
                physical_coeffs = coeffs * aux.coeff_scales.view(1, 1, 1, 2)
                z_q = (vectors * physical_coeffs[..., None]).sum(dim=-2)
                z_q = aux.post_quant_conv(z_q.permute(0, 3, 1, 2).contiguous())
                recon = ((aux.decoder(z_q).float() + 1.0) * 0.5).clamp(0, 1)
            else:
                indices = row_indices.long()
                atoms = cache_payload["atoms"][indices].to(device, dtype=torch.long)
                coeffs = cache_payload["coeffs"][indices].to(device, dtype=torch.float32)
                coeff_ids, _ = aux.compound_coeff_ids(
                    coeffs, stochastic=False, hard=True
                )
                recon = ((aux.decode_compound(atoms, coeff_ids).float() + 1.0) * 0.5).clamp(0, 1)
            real = ((images.float() + 1.0) * 0.5).clamp(0, 1)
            if metric is not None:
                metric.update(real, real=True)
                metric.update(recon, real=False)
            else:
                fr = inception(real).float()
                ff = inception(recon).float()
                real_sum += fr.sum(0, dtype=torch.float64)
                fake_sum += ff.sum(0, dtype=torch.float64)
                # The repo's native backend passes activations to numpy.cov,
                # whose accumulation is fp64. Match that numerical behavior.
                fr64, ff64 = fr.double(), ff.double()
                real_cross += fr64.t() @ fr64
                fake_cross += ff64.t() @ ff64
            seen += images.size(0)
            if (not dist.is_initialized() or dist.get_rank() == 0) and batch_idx % 100 == 0:
                global_seen = min(args.num_images, seen * world)
                print(
                    f"rFID encoded {global_seen:,}/{args.num_images:,} images",
                    flush=True,
                )
    if metric is not None:
        value = float(metric.compute().item())
    else:
        count = torch.tensor(float(seen), device=device, dtype=torch.float64)
        for tensor in (real_sum, fake_sum, real_cross, fake_cross, count):
            if dist.is_initialized():
                dist.all_reduce(tensor)
        value = native_frechet(real_sum, real_cross, fake_sum, fake_cross, int(count.item()))
    if not dist.is_initialized() or dist.get_rank() == 0:
        payload = {"checkpoint": str(args.checkpoint),
                   "split": split_name, "dataset": args.dataset,
                   "num_images": args.num_images,
                   "fid_backend": args.backend, "rfid": value,
                   "token_cache": str(args.token_cache) if args.token_cache is not None else None,
                   "coeff_quantization": (
                       cache_meta.get("coeff_quantization", "uniform")
                       if cache_meta is not None else "continuous"
                   )}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2))
        os.replace(temporary, args.output)
        if args.wandb_mode == "online":
            import wandb

            run = wandb.init(
                entity=args.wandb_entity,
                project=args.wandb_project,
                id=args.wandb_id,
                name=args.wandb_name,
                resume="allow",
            )
            metric_name = (
                "diagnostics/quantized_reconstruction_rfid"
                if cache_meta is not None
                else "diagnostics/continuous_reconstruction_rfid"
            )
            run.config.update(
                {"diagnostic_coefficient_quantizer": payload["coeff_quantization"]},
                allow_val_change=True,
            )
            run.log({
                metric_name: value,
                "diagnostics/reconstruction_rfid_num_images": args.num_images,
            })
            run.finish()
        print(
            f"Full {args.dataset.upper()} {args.num_images:,}-image rFID: {value:.6f}",
            flush=True,
        )
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
