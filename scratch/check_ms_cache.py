"""
Decode a cached spatial scale-VAR token pyramid back into images.

This sanity check compares the original images, direct AE reconstructions from
the finest latent, and reconstructions obtained from the cached per-scale
token targets.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

try:
    from scale_var import (
        _decode_batch_images,
        _decode_latents_in_chunks,
        _decode_partial_tokens_to_latent,
        _multiscale_cache_uses_current_formulation,
    )
except ModuleNotFoundError:
    try:
        from ms_omp_var import (
            _decode_batch_images,
            _decode_latents_in_chunks,
            _decode_partial_tokens_to_latent,
            _multiscale_cache_uses_current_formulation,
        )
    except ModuleNotFoundError:
        from laser_multiscale_omp_var import (
            _decode_batch_images,
            _decode_latents_in_chunks,
            _decode_partial_tokens_to_latent,
            _multiscale_cache_uses_current_formulation,
        )
try:
    from omp_var import _build_dataset, _build_laser, _parse_cli_bool, _safe_torch_load, save_image_grid
except ModuleNotFoundError:
    from laser_omp_var import _build_dataset, _build_laser, _parse_cli_bool, _safe_torch_load, save_image_grid
from proto import _load_module_checkpoint


def _reconstruct_scale_latents(
    bottleneck,
    cache: Dict[str, torch.Tensor],
    scales: Tuple[int, ...],
    start_index: int,
    num_samples: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    token_depth = int(cache["token_depth"])
    latents_by_scale: Dict[str, torch.Tensor] = {}
    for scale in scales:
        tokens = cache[f"tokens_s{int(scale)}"][start_index:start_index + num_samples].to(
            device=device,
            dtype=torch.long,
            non_blocking=(device.type == "cuda"),
        )
        tokens_grid = tokens.view(num_samples, int(scale), int(scale), token_depth)
        latents_by_scale[str(scale)] = _decode_partial_tokens_to_latent(bottleneck, tokens_grid)
    return latents_by_scale


def _stack_comparison_rows(
    originals: torch.Tensor,
    ae_recons: torch.Tensor,
    ms_recons: torch.Tensor,
) -> torch.Tensor:
    return torch.stack([originals.cpu(), ae_recons.cpu(), ms_recons.cpu()], dim=1).flatten(0, 1)


def main():
    parser = argparse.ArgumentParser(description="Decode a cached scale-VAR token pyramid.")
    parser.add_argument("--dataset", type=str, default="celeba", choices=["celeba", "cifar10"])
    parser.add_argument("--data_dir", type=str, default="/home/xl598/Projects/data/celeba")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu_threads", type=int, default=4)

    parser.add_argument("--cache_path", type=str, required=True)
    parser.add_argument("--stage1_source_ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="runs/laser_scale_var/cache_sanity")
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--decode_batch_size", type=int, default=8)

    parser.add_argument("--num_hiddens", type=int, default=128)
    parser.add_argument("--ae_num_downsamples", type=int, default=4)
    parser.add_argument("--num_res_layers", type=int, default=2)
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--num_atoms", type=int, default=1024)
    parser.add_argument("--sparsity_level", type=int, default=8)
    parser.add_argument("--n_bins", type=int, default=256)
    parser.add_argument("--coef_max", type=float, default=3.0)
    parser.add_argument("--coef_quantization", type=str, default="uniform", choices=["uniform", "mu_law"])
    parser.add_argument("--coef_mu", type=float, default=0.0)
    parser.add_argument("--commitment_cost", type=float, default=0.25)
    parser.add_argument("--patch_based", type=_parse_cli_bool, nargs="?", const=True, default=False)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--patch_stride", type=int, default=2)
    parser.add_argument("--patch_reconstruction", type=str, default="center_crop", choices=["center_crop", "hann"])
    args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(int(args.cpu_threads))
    os.environ["MKL_NUM_THREADS"] = str(int(args.cpu_threads))
    torch.set_num_threads(int(args.cpu_threads))
    try:
        torch.set_num_interop_threads(max(1, min(4, int(args.cpu_threads))))
    except RuntimeError:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    cache_path = Path(args.cache_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cache = _safe_torch_load(cache_path)
    if not _multiscale_cache_uses_current_formulation(cache):
        raise RuntimeError(
            f"Cache at {cache_path} uses a legacy multiscale formulation. "
            "Rebuild it with the current scale_var.py first."
        )
    scales = tuple(int(v) for v in torch.as_tensor(cache["scales"]).tolist())
    num_items = int(cache["num_items"])
    start_index = max(0, int(args.start_index))
    if start_index >= num_items:
        raise ValueError(f"start_index={start_index} is outside cache with num_items={num_items}")
    num_samples = min(int(args.num_samples), num_items - start_index)
    if num_samples <= 0:
        raise ValueError("num_samples must be > 0 after clamping to the cache size")

    dataset, _ = _build_dataset(args.dataset, args.data_dir, args.image_size, args.seed)
    if start_index + num_samples > len(dataset):
        raise ValueError(
            f"Requested dataset slice [{start_index}, {start_index + num_samples}) exceeds dataset size {len(dataset)}"
        )

    ae = _build_laser(args).to(device)
    ckpt_path = Path(args.stage1_source_ckpt).expanduser().resolve()
    _load_module_checkpoint(ae, ckpt_path)
    if hasattr(ae.bottleneck, "canonicalize_sparse_slots"):
        ae.bottleneck.canonicalize_sparse_slots = False
    ae.eval()
    ae.requires_grad_(False)

    originals = torch.stack(
        [_decode_batch_images(dataset[idx]) for idx in range(start_index, start_index + num_samples)],
        dim=0,
    )

    with torch.no_grad():
        originals_device = originals.to(device=device, non_blocking=(device.type == "cuda"))
        z_full = ae.encoder(originals_device)
        ae_recons = _decode_latents_in_chunks(ae, z_full, decode_batch_size=int(args.decode_batch_size))

        latents_by_scale = _reconstruct_scale_latents(
            bottleneck=ae.bottleneck,
            cache=cache,
            scales=scales,
            start_index=start_index,
            num_samples=num_samples,
            device=device,
        )
        final_latent = latents_by_scale[str(scales[-1])]
        ms_recons = _decode_latents_in_chunks(ae, final_latent, decode_batch_size=int(args.decode_batch_size))

    comparison = _stack_comparison_rows(originals, ae_recons, ms_recons)
    comparison_path = output_dir / "cache_compare_input_ae_multiscale.png"
    save_image_grid(comparison, str(comparison_path), nrow=3)

    latent_mse = F.mse_loss(final_latent, z_full).item()
    ae_image_mse = F.mse_loss(ae_recons, originals.cpu()).item()
    ms_image_mse = F.mse_loss(ms_recons, originals.cpu()).item()
    ms_vs_ae_image_mse = F.mse_loss(ms_recons, ae_recons).item()

    for scale in scales:
        latent_scale = latents_by_scale[str(scale)]
        full_latent = latent_scale
        if int(scale) != int(scales[-1]):
            full_latent = F.interpolate(
                latent_scale,
                size=(int(scales[-1]), int(scales[-1])),
                mode="bilinear",
                align_corners=False,
            )
        scale_imgs = _decode_latents_in_chunks(ae, full_latent, decode_batch_size=int(args.decode_batch_size))
        scale_path = output_dir / f"cache_scale_s{int(scale)}.png"
        save_image_grid(scale_imgs, str(scale_path))

    print(f"[ScaleVAR sanity] device={device} cache={cache_path}")
    print(f"[ScaleVAR sanity] scales={scales} samples={num_samples} start_index={start_index}")
    print(f"[ScaleVAR sanity] comparison_grid={comparison_path}")
    print(f"[ScaleVAR sanity] latent_mse(finest_scale, z_full)={latent_mse:.6f}")
    print(f"[ScaleVAR sanity] image_mse(ae_recon, input)={ae_image_mse:.6f}")
    print(f"[ScaleVAR sanity] image_mse(scale_var_recon, input)={ms_image_mse:.6f}")
    print(f"[ScaleVAR sanity] image_mse(scale_var_recon, ae_recon)={ms_vs_ae_image_mse:.6f}")
    print(f"[ScaleVAR sanity] outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
