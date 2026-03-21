#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

import laser
import laser_transformer
import proto


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a one-batch anomaly-detection repro for the patch recon_mse stage-2 loss."
    )
    parser.add_argument("--stage1-ckpt", required=True)
    parser.add_argument("--stage2-ckpt", required=True)
    parser.add_argument("--token-cache", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--ae-num-downsamples", type=int, default=2)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--num-atoms", type=int, default=4096)
    parser.add_argument("--sparsity-level", type=int, default=16)
    parser.add_argument("--coef-max", type=float, default=8.0)
    parser.add_argument("--num-hiddens", type=int, default=64)
    parser.add_argument("--num-res-layers", type=int, default=1)
    parser.add_argument("--max-ch-mult", type=int, default=1)
    parser.add_argument("--decoder-extra-residual-layers", type=int, default=0)
    parser.add_argument("--tf-d-model", type=int, default=512)
    parser.add_argument("--tf-heads", type=int, default=8)
    parser.add_argument("--tf-layers", type=int, default=12)
    parser.add_argument("--tf-ff", type=int, default=1024)
    parser.add_argument("--tf-dropout", type=float, default=0.1)
    parser.add_argument("--tf-global-tokens", type=int, default=0)
    parser.add_argument("--coeff-loss-weight", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        dist.init_process_group(backend="nccl" if args.device.startswith("cuda") else "gloo")
    if args.device.startswith("cuda"):
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device(args.device)
    torch.manual_seed(0)
    torch.autograd.set_detect_anomaly(True)

    token_cache = torch.load(Path(args.token_cache), map_location="cpu")
    tokens_flat = token_cache["tokens_flat"]
    coeffs_flat = token_cache["coeffs_flat"]
    H, W, D = token_cache["shape"]
    sample_index = int(args.sample_index)
    if rank == 0:
        print(
            f"[debug] token cache loaded: tokens={tuple(tokens_flat.shape)} "
            f"coeffs={tuple(coeffs_flat.shape)} shape={(H, W, D)} sample_index={sample_index}"
        )

    if sample_index < 0 or sample_index >= int(tokens_flat.size(0)):
        raise IndexError(f"sample_index {sample_index} out of range for {tokens_flat.size(0)} cached items")

    ae = laser.LASER(
        in_channels=3,
        resolution=args.image_size,
        num_hiddens=args.num_hiddens,
        num_downsamples=args.ae_num_downsamples,
        num_residual_layers=args.num_res_layers,
        max_ch_mult=args.max_ch_mult,
        decoder_extra_residual_layers=args.decoder_extra_residual_layers,
        use_mid_attention=False,
        embedding_dim=args.embedding_dim,
        num_embeddings=args.num_atoms,
        sparsity_level=args.sparsity_level,
        commitment_cost=1.0,
        n_bins=256,
        coef_max=args.coef_max,
        coef_quantization="uniform",
        coef_mu=0.0,
        quantize_sparse_coeffs=False,
        patch_based=True,
        patch_size=8,
        patch_stride=4,
        patch_reconstruction="hann",
    ).to(device)
    proto._load_module_checkpoint(ae, Path(args.stage1_ckpt))
    ae.eval()
    latent_hw = args.image_size // (2 ** args.ae_num_downsamples)
    ae._remember_latent_hw(torch.empty(1, args.embedding_dim, latent_hw, latent_hw))

    cfg = laser_transformer.build_spatial_depth_prior_config(
        ae.bottleneck,
        H=H,
        W=W,
        D=D,
        d_model=args.tf_d_model,
        n_heads=args.tf_heads,
        n_spatial_layers=args.tf_layers,
        n_depth_layers=max(1, args.tf_layers // 2),
        d_ff=args.tf_ff,
        dropout=args.tf_dropout,
        n_global_spatial_tokens=args.tf_global_tokens,
        real_valued_coeffs=True,
        coeff_max_fallback=args.coef_max,
        autoregressive_coeffs=True,
    )
    transformer = laser_transformer.SpatialDepthPrior(cfg).to(device)
    proto._load_module_checkpoint(transformer, Path(args.stage2_ckpt))
    transformer.train()
    if distributed:
        transformer = DDP(transformer, device_ids=[local_rank], output_device=local_rank)
    transformer_module = transformer.module if isinstance(transformer, DDP) else transformer

    tok_grid = tokens_flat[sample_index:sample_index + 1].view(1, H * W, D).to(
        device=device,
        dtype=torch.long,
    )
    coeff_grid = coeffs_flat[sample_index:sample_index + 1].view(1, H * W, D).to(
        device=device,
        dtype=torch.float32,
    )

    atom_logits, _, _ = transformer(
        tok_grid,
        coeff_grid,
        mask_tokens=tok_grid,
        return_features=True,
    )
    ce_loss = F.cross_entropy(atom_logits.reshape(-1, cfg.vocab_size), tok_grid.reshape(-1))
    target_coeff = ae.clamp_sparse_coeffs(coeff_grid)

    rollout_context_tok_grid = tok_grid.detach().clone()
    rollout_context_coeff_grid = coeff_grid.detach().clone()
    rollout_mask_tok_grid = rollout_context_tok_grid.detach().clone()
    rollout_atom_logits, _, rollout_depth_h = transformer(
        rollout_context_tok_grid,
        rollout_context_coeff_grid,
        mask_tokens=rollout_mask_tok_grid,
        return_features=True,
    )

    pred_atoms = rollout_atom_logits.argmax(dim=-1)
    pred_atoms_for_coeff = pred_atoms.detach().clone()
    pred_atoms_for_recon = pred_atoms.detach().clone()
    pred_coeff = ae.clamp_sparse_coeffs(
        transformer_module.predict_coeffs_for_atoms(rollout_depth_h, pred_atoms_for_coeff)
    )
    pred_latent = proto._reconstruct_stage2_sparse_latent(
        ae,
        pred_atoms_for_recon.view(1, H, W, D),
        pred_coeff.view(1, H, W, D),
    )
    with torch.no_grad():
        target_latent = proto._reconstruct_stage2_sparse_latent(
            ae,
            tok_grid.view(1, H, W, D),
            target_coeff.view(1, H, W, D),
        )
    coeff_reg_loss = F.mse_loss(pred_latent, target_latent)
    loss = ce_loss + float(args.coeff_loss_weight) * coeff_reg_loss
    if rank == 0:
        print(
            "[debug] forward complete: "
            f"ce={float(ce_loss.item()):.6f} "
            f"recon={float(coeff_reg_loss.item()):.6f} "
            f"loss={float(loss.item()):.6f}"
        )
    loss.backward()
    if rank == 0:
        print("[debug] backward complete")
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
