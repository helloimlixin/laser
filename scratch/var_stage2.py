import argparse
import math
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from tqdm import tqdm

from proto import (
    DEFAULT_CELEBA_DIR,
    FlatImageDataset,
    LASER,
    _barrier,
    _cleanup_distributed,
    _compute_quantized_rq_losses,
    _compute_stage2_sample_reference_stats,
    _decode_stage2_candidates_in_chunks,
    _distributed_mean,
    _init_distributed,
    _init_wandb,
    _load_module_checkpoint,
    _load_token_cache,
    _log_wandb,
    _log_wandb_image,
    _parse_cli_bool,
    _select_best_stage2_samples,
    _unlink_if_exists,
    _unwrap_module,
    _wait_for_file_signal,
    _write_atomic_text,
    precompute_tokens,
    save_image_grid,
)
from var import SparsityLevelVAR, build_sparsity_var_config, force_greedy_omp_slot_order


def _build_var_cache_expected_meta(args, stage2_source_set, token_subset: Optional[int], ae: LASER, stage1_ckpt: Path) -> dict:
    effective_items = len(stage2_source_set) if token_subset is None else int(token_subset)
    st = stage1_ckpt.stat()
    return {
        "version": 2,
        "dataset": str(args.dataset),
        "image_size": int(args.image_size),
        "seed": int(args.seed),
        "source_items": int(len(stage2_source_set)),
        "effective_items": int(effective_items),
        "quantize_sparse_coeffs": bool(ae.bottleneck.quantize_sparse_coeffs),
        "ae_num_downsamples": int(args.ae_num_downsamples),
        "embedding_dim": int(args.embedding_dim),
        "num_atoms": int(args.num_atoms),
        "sparsity_level": int(args.sparsity_level),
        "max_ch_mult": int(getattr(ae, "max_ch_mult", 2)),
        "decoder_extra_residual_layers": int(getattr(ae, "decoder_extra_residual_layers", 1)),
        "use_mid_attention": bool(getattr(ae, "use_mid_attention", True)),
        "patch_based": bool(args.patch_based),
        "patch_size": int(args.patch_size),
        "patch_stride": int(args.patch_stride),
        "patch_reconstruction": str(args.patch_reconstruction),
        "canonicalize_sparse_slots": bool(getattr(ae.bottleneck, "canonicalize_sparse_slots", True)),
        "omp_slot_order": "greedy" if not bool(getattr(ae.bottleneck, "canonicalize_sparse_slots", True)) else "canonicalized",
        "stage1_ckpt_path": str(stage1_ckpt.resolve()),
        "stage1_ckpt_size": int(st.st_size),
        "stage1_ckpt_mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))),
    }


def _var_cache_is_compatible(cache, expected_meta: dict) -> Tuple[bool, str]:
    if not isinstance(cache, dict):
        return False, "cache payload is not a dict"

    tokens_flat = cache.get("tokens_flat")
    coeffs_flat = cache.get("coeffs_flat", None)
    shape = cache.get("shape")
    if not torch.is_tensor(tokens_flat) or tokens_flat.ndim != 2:
        return False, "tokens_flat must be a rank-2 tensor"
    if not isinstance(shape, (tuple, list)) or len(shape) != 3:
        return False, "shape must be a 3-tuple/list"

    H, W, D = (int(shape[0]), int(shape[1]), int(shape[2]))
    if tokens_flat.size(1) != H * W * D:
        return False, "tokens_flat width does not match cached shape metadata"

    expect_real_valued = not bool(expected_meta["quantize_sparse_coeffs"])
    if expect_real_valued != (coeffs_flat is not None):
        mode = "real-valued coefficients" if expect_real_valued else "quantized coefficients"
        return False, "cache coefficient mode does not match current run ({})".format(mode)

    if coeffs_flat is not None:
        if not torch.is_tensor(coeffs_flat) or coeffs_flat.ndim != 2:
            return False, "coeffs_flat must be a rank-2 tensor when present"
        if coeffs_flat.size(0) != tokens_flat.size(0):
            return False, "coeffs_flat row count does not match tokens_flat"
        if coeffs_flat.size(1) != H * W * D:
            return False, "coeffs_flat width does not match cached shape metadata"

    required_items = int(expected_meta["effective_items"])
    if tokens_flat.size(0) < required_items:
        return False, "cache has {} items but run needs {}".format(tokens_flat.size(0), required_items)

    cache_meta = cache.get("meta")
    if cache_meta is None:
        return False, "cache meta is missing"
    for key, expected_value in expected_meta.items():
        if cache_meta.get(key) != expected_value:
            return False, "meta mismatch for {}: cache={!r}, expected={!r}".format(
                key,
                cache_meta.get(key),
                expected_value,
            )
    return True, "ok"


def _build_laser(args) -> LASER:
    return LASER(
        in_channels=3,
        num_hiddens=args.num_hiddens,
        num_downsamples=args.ae_num_downsamples,
        num_residual_layers=args.num_res_layers,
        resolution=args.image_size,
        embedding_dim=args.embedding_dim,
        num_embeddings=args.num_atoms,
        sparsity_level=args.sparsity_level,
        commitment_cost=args.commitment_cost,
        n_bins=args.n_bins,
        coef_max=args.coef_max,
        quantize_sparse_coeffs=args.quantize_sparse_coeffs,
        coef_quantization=args.coef_quantization,
        coef_mu=args.coef_mu,
        out_tanh=True,
        patch_based=args.patch_based,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        patch_reconstruction=args.patch_reconstruction,
    )


def _build_stage2_source_set(args):
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    eval_tfm = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        normalize,
    ])
    if args.dataset == "cifar10":
        return datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=eval_tfm)
    if args.dataset == "celeba":
        token_full = FlatImageDataset(root=args.data_dir, transform=eval_tfm)
        val_size = max(1, int(0.05 * len(token_full)))
        train_size = len(token_full) - val_size
        indices = torch.randperm(len(token_full), generator=torch.Generator().manual_seed(args.seed)).tolist()
        train_indices = indices[:train_size]
        return Subset(token_full, train_indices)
    raise ValueError("Unsupported dataset: {}".format(args.dataset))


def train_stage2_var(
    model: nn.Module,
    token_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    rq_atom_loss_weight: float,
    rq_coeff_loss_weight: float,
    coeff_loss_weight: float,
    coeff_loss_type: str,
    coeff_huber_delta: float,
    coeff_depth_weighting: str,
    coeff_focal_gamma: float,
    stage2_amp: bool,
    stage2_amp_dtype: str,
    out_dir: str,
    ae_for_decode: LASER,
    H: int,
    W: int,
    D: int,
    sample_every_steps: int,
    sample_batch_size: int,
    sample_candidate_factor: int,
    sample_temperature: float,
    sample_temperature_end: Optional[float],
    sample_top_k: Optional[int],
    sample_top_p: Optional[float],
    sample_image_size: Optional[int],
    sample_reference_stats: Optional[dict],
    token_sampler: Optional[DistributedSampler],
    is_main_process: bool,
    wandb_run: Optional[object],
    warmup_steps: int,
    min_lr_ratio: float,
    weight_decay: float,
):
    model_module = _unwrap_module(model)
    ae_decode = _unwrap_module(ae_for_decode)
    ae_decode.eval()
    ae_decode.requires_grad_(False)

    if weight_decay > 0.0:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)

    total_steps = max(1, epochs * max(1, len(token_loader)))
    warmup_steps = max(0, int(warmup_steps))
    min_lr_ratio = float(max(0.0, min(float(min_lr_ratio), 1.0)))
    coeff_loss_type = str(coeff_loss_type).lower()
    if coeff_loss_type not in {"huber", "mse", "recon_mse", "gt_atom_recon_mse"}:
        raise ValueError("Unsupported coeff_loss_type: {!r}".format(coeff_loss_type))

    amp_dtype_name = str(stage2_amp_dtype).strip().lower()
    if amp_dtype_name not in {"auto", "float16", "bfloat16"}:
        raise ValueError("Unsupported stage2_amp_dtype: {!r}".format(stage2_amp_dtype))
    amp_enabled = bool(stage2_amp) and device.type == "cuda"
    if amp_dtype_name == "auto":
        amp_dtype = torch.bfloat16 if (amp_enabled and torch.cuda.is_bf16_supported()) else torch.float16
    elif amp_dtype_name == "bfloat16":
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16
    scaler_enabled = amp_enabled and amp_dtype == torch.float16
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler(device="cuda", enabled=scaler_enabled)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    global_step = 0
    best_epoch_loss = float("inf")
    sample_top_k = None if sample_top_k is None or int(sample_top_k) <= 0 else int(sample_top_k)
    sample_top_p = None if sample_top_p is None or float(sample_top_p) <= 0.0 else float(sample_top_p)
    sample_candidate_factor = max(1, int(sample_candidate_factor))
    real_valued = bool(model_module.real_valued_coeffs)

    for epoch in range(1, epochs + 1):
        if token_sampler is not None:
            token_sampler.set_epoch(epoch)
        model.train()
        pbar = tqdm(token_loader, desc="[VAR] epoch {}/{}".format(epoch, epochs), disable=(not is_main_process))
        running = 0.0
        steps = 0

        for batch in pbar:
            if real_valued:
                tok_flat = batch[0].to(device).long()
                coeff_flat = batch[1].to(device).float()
            else:
                tok_flat = batch[0] if isinstance(batch, (tuple, list)) else batch
                tok_flat = tok_flat.to(device).long()
                coeff_flat = None
            B = tok_flat.size(0)

            opt.zero_grad(set_to_none=True)
            atom_ce_loss = None
            coeff_ce_loss = None
            coeff_reg_loss = None
            coeff_ce_raw = None
            ce_loss = None
            autocast_ctx = (
                torch.autocast(device_type=device.type, dtype=amp_dtype)
                if amp_enabled else nullcontext()
            )
            with autocast_ctx:
                if real_valued:
                    tok_grid = tok_flat.view(B, H * W, D)
                    coeff_grid = coeff_flat.view(B, H * W, D)
                    atom_logits, coeff_pred = model(
                        tok_grid,
                        coeff_grid,
                        bottleneck=ae_decode.bottleneck,
                    )
                    ce_loss = F.cross_entropy(
                        atom_logits.reshape(-1, model_module.atom_vocab_size),
                        tok_grid.reshape(-1),
                    )
                    atom_ce_loss = ce_loss
                    pred_coeff = ae_decode.clamp_sparse_coeffs(coeff_pred)
                    coef_max = float(getattr(ae_decode.bottleneck, "coef_max", float("inf")))
                    target_coeff = coeff_grid.clamp(-coef_max, coef_max)
                    if coeff_loss_type == "mse":
                        coeff_reg_loss = F.mse_loss(pred_coeff, target_coeff)
                    elif coeff_loss_type == "huber":
                        coeff_reg_loss = F.huber_loss(pred_coeff, target_coeff, delta=coeff_huber_delta)
                    elif coeff_loss_type == "recon_mse":
                        pred_atoms = atom_logits.argmax(dim=-1)
                        pred_latent = ae_decode.bottleneck._reconstruct_sparse(
                            pred_atoms.view(B, H, W, D),
                            pred_coeff.view(B, H, W, D),
                        )
                        with torch.no_grad():
                            target_latent = ae_decode.bottleneck._reconstruct_sparse(
                                tok_grid.view(B, H, W, D),
                                target_coeff.view(B, H, W, D),
                            )
                        coeff_reg_loss = F.mse_loss(pred_latent, target_latent)
                    else:
                        pred_latent = ae_decode.bottleneck._reconstruct_sparse(
                            tok_grid.view(B, H, W, D),
                            pred_coeff.view(B, H, W, D),
                        )
                        with torch.no_grad():
                            target_latent = ae_decode.bottleneck._reconstruct_sparse(
                                tok_grid.view(B, H, W, D),
                                target_coeff.view(B, H, W, D),
                            )
                        coeff_reg_loss = F.mse_loss(pred_latent, target_latent)
                    loss = ce_loss + float(coeff_loss_weight) * coeff_reg_loss
                else:
                    tok_grid = tok_flat.view(B, H * W, D)
                    atom_targets = tok_grid[..., 0::2]
                    coeff_targets = tok_grid[..., 1::2] - model_module.atom_vocab_size
                    atom_logits, coeff_logits = model(
                        tok_grid,
                        bottleneck=ae_decode.bottleneck,
                    )
                    atom_ce_per = F.cross_entropy(
                        atom_logits.reshape(-1, model_module.atom_vocab_size),
                        atom_targets.reshape(-1),
                        reduction="none",
                    ).view(B, H * W, model_module.num_stages)
                    coeff_ce_per = F.cross_entropy(
                        coeff_logits.reshape(-1, model_module.coeff_vocab_size),
                        coeff_targets.reshape(-1),
                        reduction="none",
                    ).view(B, H * W, model_module.num_stages)
                    coeff_ce_raw = coeff_ce_per.mean()
                    per_token_ce = torch.empty(
                        B,
                        H * W,
                        D,
                        device=tok_grid.device,
                        dtype=atom_ce_per.dtype,
                    )
                    per_token_ce[..., 0::2] = atom_ce_per
                    per_token_ce[..., 1::2] = coeff_ce_per
                    ce_loss, atom_ce_loss, coeff_ce_loss, loss = _compute_quantized_rq_losses(
                        per_token_ce,
                        atom_loss_weight=rq_atom_loss_weight,
                        coeff_loss_weight=rq_coeff_loss_weight,
                        coeff_depth_weighting=coeff_depth_weighting,
                        coeff_focal_gamma=coeff_focal_gamma,
                        coeff_logits=coeff_logits,
                    )

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            global_step += 1
            if warmup_steps > 0 or min_lr_ratio < 1.0:
                if global_step <= warmup_steps:
                    scale = max(0.01, global_step / max(1, warmup_steps))
                else:
                    progress = (global_step - warmup_steps) / max(1, total_steps - warmup_steps)
                    progress = min(progress, 1.0)
                    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                    scale = min_lr_ratio + (1.0 - min_lr_ratio) * cosine
                for pg in opt.param_groups:
                    pg["lr"] = lr * scale

            loss_log = _distributed_mean(loss)
            ce_log = _distributed_mean(ce_loss.detach()) if ce_loss is not None else None
            atom_ce_log = _distributed_mean(atom_ce_loss.detach()) if atom_ce_loss is not None else None
            coeff_ce_log = _distributed_mean(coeff_ce_loss.detach()) if coeff_ce_loss is not None else None
            coeff_ce_raw_log = _distributed_mean(coeff_ce_raw.detach()) if coeff_ce_raw is not None else None
            coeff_reg_log = _distributed_mean(coeff_reg_loss.detach()) if coeff_reg_loss is not None else None
            running += float(loss_log.item())
            steps += 1

            if is_main_process:
                postfix = {"loss": float(loss_log.item())}
                if ce_log is not None:
                    postfix["ce"] = float(ce_log.item())
                if atom_ce_log is not None:
                    postfix["atom_ce"] = float(atom_ce_log.item())
                if coeff_ce_log is not None:
                    postfix["coeff_ce"] = float(coeff_ce_log.item())
                if coeff_reg_log is not None:
                    postfix[coeff_loss_type] = float(coeff_reg_log.item())
                pbar.set_postfix(**postfix)

                log_payload = {
                    "stage2/train_loss": float(loss_log.item()),
                    "stage2/epoch": epoch,
                    "stage2/lr": float(opt.param_groups[0]["lr"]),
                }
                if ce_log is not None:
                    log_payload["stage2/ce_loss"] = float(ce_log.item())
                if atom_ce_log is not None:
                    log_payload["stage2/atom_ce_loss"] = float(atom_ce_log.item())
                if coeff_ce_log is not None:
                    log_payload["stage2/coeff_ce_loss"] = float(coeff_ce_log.item())
                if coeff_ce_raw_log is not None:
                    log_payload["stage2/coeff_ce_raw"] = float(coeff_ce_raw_log.item())
                if coeff_reg_log is not None:
                    log_payload["stage2/coeff_reg_loss"] = float(coeff_reg_log.item())
                    log_payload["stage2/coeff_loss_type"] = coeff_loss_type
                _log_wandb(
                    wandb_run,
                    log_payload,
                    step_metric="stage2/step",
                    step_value=global_step,
                )

            if sample_every_steps > 0 and (global_step % sample_every_steps == 0):
                opt.zero_grad(set_to_none=True)
                _barrier()
                if is_main_process:
                    model.eval()
                    print("[VAR] sampling at step {}...".format(global_step))
                    with torch.no_grad():
                        candidate_batch = max(int(sample_batch_size), int(sample_batch_size) * sample_candidate_factor)
                        if real_valued:
                            atoms_gen, coeffs_gen = model_module.generate(
                                batch_size=candidate_batch,
                                temperature=sample_temperature,
                                temperature_end=sample_temperature_end,
                                top_k=sample_top_k,
                                top_p=sample_top_p,
                                bottleneck=ae_decode.bottleneck,
                                show_progress=True,
                                progress_desc="[VAR] sample step {}".format(global_step),
                            )
                            coeffs_gen = ae_decode.clamp_sparse_coeffs(coeffs_gen)
                            imgs = _decode_stage2_candidates_in_chunks(
                                ae_decode,
                                atoms_gen.view(-1, H, W, D),
                                coeffs=coeffs_gen.view(-1, H, W, D),
                                decode_batch_size=max(1, min(8, int(sample_batch_size))),
                            )
                        else:
                            tokens_gen = model_module.generate(
                                batch_size=candidate_batch,
                                temperature=sample_temperature,
                                temperature_end=sample_temperature_end,
                                top_k=sample_top_k,
                                top_p=sample_top_p,
                                bottleneck=ae_decode.bottleneck,
                                show_progress=True,
                                progress_desc="[VAR] sample step {}".format(global_step),
                            )
                            imgs = _decode_stage2_candidates_in_chunks(
                                ae_decode,
                                tokens_gen.view(-1, H, W, D),
                                decode_batch_size=max(1, min(8, int(sample_batch_size))),
                            )
                        imgs_raw = imgs[: min(int(sample_batch_size), int(imgs.size(0)))].clone()
                        imgs = _select_best_stage2_samples(
                            imgs,
                            keep=sample_batch_size,
                            reference_stats=sample_reference_stats,
                        )
                        if sample_image_size is not None and int(sample_image_size) > 0:
                            if imgs.size(-2) != int(sample_image_size) or imgs.size(-1) != int(sample_image_size):
                                imgs = F.interpolate(
                                    imgs,
                                    size=(int(sample_image_size), int(sample_image_size)),
                                    mode="bilinear",
                                    align_corners=False,
                                )
                    save_image_grid(imgs_raw, os.path.join(out_dir, "var_step{:06d}_raw_samples.png".format(global_step)))
                    save_image_grid(imgs, os.path.join(out_dir, "var_step{:06d}_samples.png".format(global_step)))
                    _log_wandb_image(
                        wandb_run,
                        "stage2/raw_samples",
                        imgs_raw,
                        step_metric="stage2/step",
                        step_value=global_step,
                        caption="step={} raw".format(global_step),
                    )
                    _log_wandb_image(
                        wandb_run,
                        "stage2/samples",
                        imgs,
                        step_metric="stage2/step",
                        step_value=global_step,
                        caption="step={}".format(global_step),
                    )
                    print("[VAR] sampling done at step {}".format(global_step))
                _barrier()
                model.train()

        epoch_loss = running / max(1, steps)
        if is_main_process:
            print("[VAR] epoch {} train_loss={:.6f}".format(epoch, epoch_loss))
            _log_wandb(
                wandb_run,
                {
                    "stage2/epoch_loss": float(epoch_loss),
                    "stage2/epoch": epoch,
                },
                step_metric="stage2/step",
                step_value=global_step,
            )

        _barrier()
        if is_main_process:
            os.makedirs(out_dir, exist_ok=True)
            torch.save(model_module.state_dict(), os.path.join(out_dir, "var_last.pt"))
            if epoch_loss < best_epoch_loss:
                best_epoch_loss = epoch_loss
                torch.save(model_module.state_dict(), os.path.join(out_dir, "var_best.pt"))
        _barrier()


def main():
    parser = argparse.ArgumentParser(description="Train stage-2 SparsityLevelVAR on top of a LASER checkpoint.")
    parser.add_argument("--run_dir", type=str, required=True, help="A completed proto.py run directory containing stage1/ae_best.pt")
    parser.add_argument("--dataset", type=str, default="celeba", choices=["cifar10", "celeba"])
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_CELEBA_DIR))
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dist_timeout_minutes", type=int, default=180)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--token_num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--stage2_batch_size", type=int, default=32)
    parser.add_argument("--stage2_epochs", type=int, default=50)
    parser.add_argument("--stage2_lr", type=float, default=1e-3)
    parser.add_argument("--stage2_rq_atom_loss_weight", type=float, default=1.0)
    parser.add_argument("--stage2_rq_coeff_loss_weight", type=float, default=1.0)
    parser.add_argument("--stage2_coeff_loss_weight", type=float, default=0.1)
    parser.add_argument("--stage2_coeff_loss_type", type=str, default="gt_atom_recon_mse", choices=["huber", "mse", "recon_mse", "gt_atom_recon_mse"])
    parser.add_argument("--stage2_coeff_huber_delta", type=float, default=1.0)
    parser.add_argument("--coeff_depth_weighting", type=str, default="none", choices=["none", "linear", "inverse_rank"])
    parser.add_argument("--coeff_focal_gamma", type=float, default=0.0)
    parser.add_argument("--stage2_warmup_steps", type=int, default=500)
    parser.add_argument("--stage2_min_lr_ratio", type=float, default=0.01)
    parser.add_argument("--stage2_weight_decay", type=float, default=0.01)
    parser.add_argument("--stage2_amp", type=_parse_cli_bool, nargs="?", const=True, default=True)
    parser.add_argument("--stage2_amp_dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16"])
    parser.add_argument("--stage2_sample_every_steps", type=int, default=2000)
    parser.add_argument("--stage2_sample_batch_size", type=int, default=32)
    parser.add_argument("--stage2_sample_candidate_factor", type=int, default=4)
    parser.add_argument("--stage2_sample_temperature", type=float, default=0.9)
    parser.add_argument("--stage2_sample_temperature_end", type=float, default=1.0)
    parser.add_argument("--stage2_sample_top_k", type=int, default=256)
    parser.add_argument("--stage2_sample_top_p", type=float, default=0.95)
    parser.add_argument("--stage2_sample_image_size", type=int, default=128)
    parser.add_argument("--token_subset", type=int, default=0)
    parser.add_argument("--rebuild_token_cache", action="store_true")

    parser.add_argument("--wandb", dest="wandb", action="store_true", default=True)
    parser.add_argument("--no_wandb", dest="wandb", action="store_false")
    parser.add_argument("--wandb_project", type=str, default="laser")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default="laser_var_stage2")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_dir", type=str, default="./wandb")

    parser.add_argument("--num_hiddens", type=int, default=128)
    parser.add_argument("--ae_num_downsamples", type=int, default=4)
    parser.add_argument("--num_res_layers", type=int, default=2)
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--num_atoms", type=int, default=1024)
    parser.add_argument("--sparsity_level", type=int, default=8)
    parser.add_argument("--n_bins", type=int, default=256)
    parser.add_argument("--coef_max", type=float, default=3.0)
    parser.add_argument("--quantize_sparse_coeffs", type=_parse_cli_bool, nargs="?", const=True, default=True)
    parser.add_argument("--coef_quantization", type=str, default="uniform", choices=["uniform", "mu_law"])
    parser.add_argument("--coef_mu", type=float, default=0.0)
    parser.add_argument("--commitment_cost", type=float, default=0.25)
    parser.add_argument("--patch_based", dest="patch_based", action="store_true", default=False)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--patch_stride", type=int, default=2)
    parser.add_argument("--patch_reconstruction", type=str, default="center_crop", choices=["center_crop", "hann"])

    parser.add_argument("--var_d_model", type=int, default=512)
    parser.add_argument("--var_heads", type=int, default=8)
    parser.add_argument("--var_layers", type=int, default=12)
    parser.add_argument("--var_ff", type=int, default=1024)
    parser.add_argument("--var_dropout", type=float, default=0.1)
    parser.add_argument("--var_global_tokens", type=int, default=0)

    args = parser.parse_args()
    if args.stage2_sample_temperature <= 0.0:
        raise ValueError("stage2_sample_temperature must be > 0")
    if args.stage2_sample_temperature_end <= 0.0:
        raise ValueError("stage2_sample_temperature_end must be > 0")
    if not 0.0 <= args.stage2_sample_top_p <= 1.0:
        raise ValueError("stage2_sample_top_p must be in [0, 1]")
    if args.token_subset < 0:
        args.token_subset = 0
    run_dir = Path(args.run_dir).expanduser().resolve()
    stage1_ckpt = run_dir / "stage1" / "ae_best.pt"
    if not stage1_ckpt.exists():
        raise FileNotFoundError("Stage-1 checkpoint not found: {}".format(stage1_ckpt))

    distributed, rank, local_rank, world_size = _init_distributed(args.dist_timeout_minutes)
    is_main_process = (rank == 0)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("medium")

    if distributed:
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    stage2_dir = run_dir / "stage2_var"
    if is_main_process:
        os.makedirs(stage2_dir, exist_ok=True)

    laser = _build_laser(args).to(device)
    _load_module_checkpoint(laser, stage1_ckpt)
    force_greedy_omp_slot_order(laser)
    laser.eval()

    if is_main_process:
        print("[VAR] stage1 checkpoint: {}".format(stage1_ckpt))
        print("[VAR] using greedy OMP slot order for stage2 tokenization")

    stage2_source_set = _build_stage2_source_set(args)
    token_subset = None if args.token_subset <= 0 else min(args.token_subset, len(stage2_source_set))
    token_cache_path = stage2_dir / "tokens_cache_greedy.pt"
    token_cache_ready_path = stage2_dir / "tokens_cache_greedy.ready"
    token_cache_error_path = stage2_dir / "tokens_cache_greedy.failed"
    expected_meta = _build_var_cache_expected_meta(args, stage2_source_set, token_subset, laser, stage1_ckpt)

    _barrier()
    if is_main_process:
        _unlink_if_exists(token_cache_ready_path)
        _unlink_if_exists(token_cache_error_path)
        try:
            cache_ready = False
            if token_cache_path.exists() and not args.rebuild_token_cache:
                token_cache = _load_token_cache(str(token_cache_path))
                compatible, reason = _var_cache_is_compatible(token_cache, expected_meta)
                if compatible:
                    tokens_flat = token_cache["tokens_flat"]
                    H, W, D = token_cache["shape"]
                    print("[VAR] reusing token cache: {} (H={}, W={}, D={})".format(tuple(tokens_flat.shape), H, W, D))
                    cache_ready = True
                else:
                    print("[VAR] rebuilding token cache ({})".format(reason))
            else:
                if args.rebuild_token_cache:
                    print("[VAR] rebuilding token cache (--rebuild_token_cache)")
            if not cache_ready:
                token_loader = DataLoader(
                    stage2_source_set,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.token_num_workers,
                    pin_memory=pin_memory,
                    persistent_workers=(args.token_num_workers > 0),
                )
                tokens_flat, coeffs_flat, H, W, D = precompute_tokens(
                    laser,
                    token_loader,
                    device,
                    max_items=token_subset,
                )
                cache = {
                    "tokens_flat": tokens_flat,
                    "shape": (H, W, D),
                    "meta": expected_meta,
                }
                if coeffs_flat is not None:
                    cache["coeffs_flat"] = coeffs_flat
                torch.save(cache, str(token_cache_path))
                print("[VAR] token dataset: {} (H={}, W={}, D={})".format(tuple(tokens_flat.shape), H, W, D))
            _write_atomic_text(token_cache_ready_path, "ready\n")
        except Exception as exc:
            _write_atomic_text(token_cache_error_path, "[VAR] token cache preparation failed: {}: {}\n".format(type(exc).__name__, exc))
            raise
    else:
        _wait_for_file_signal(
            token_cache_ready_path,
            token_cache_error_path,
            timeout_seconds=max(60.0, float(args.dist_timeout_minutes) * 60.0),
            description="var token cache at {}".format(token_cache_path),
        )
    token_cache = _load_token_cache(str(token_cache_path))
    tokens_flat = token_cache["tokens_flat"]
    coeffs_flat = token_cache.get("coeffs_flat", None)
    H, W, D = token_cache["shape"]
    expected_items = int(expected_meta["effective_items"])
    if tokens_flat.size(0) > expected_items:
        tokens_flat = tokens_flat[:expected_items]
        if coeffs_flat is not None:
            coeffs_flat = coeffs_flat[:expected_items]
    real_valued = coeffs_flat is not None

    sample_reference_stats = _compute_stage2_sample_reference_stats(
        laser,
        tokens_flat,
        coeffs_flat,
        H,
        W,
        D,
        device,
    )
    if real_valued:
        token_dataset = TensorDataset(tokens_flat, coeffs_flat)
    else:
        token_dataset = tokens_flat
    token_sampler = DistributedSampler(token_dataset, shuffle=True) if distributed else None
    token_loader = DataLoader(
        token_dataset,
        batch_size=args.stage2_batch_size,
        shuffle=(token_sampler is None),
        sampler=token_sampler,
        num_workers=0,
        pin_memory=pin_memory,
        drop_last=(len(token_dataset) >= args.stage2_batch_size),
    )

    var_model = SparsityLevelVAR(
        build_sparsity_var_config(
            laser.bottleneck,
            H=H,
            W=W,
            d_model=args.var_d_model,
            n_heads=args.var_heads,
            n_layers=args.var_layers,
            d_ff=args.var_ff,
            dropout=args.var_dropout,
            n_global_tokens=args.var_global_tokens,
            use_reconstruction_conditioning=True,
            real_valued_coeffs=real_valued,
            coeff_max_fallback=args.coef_max,
        )
    ).to(device)
    if is_main_process:
        coeff_mode = "real-valued coeffs" if real_valued else "quantized sparse coeffs"
        print(
            "[VAR] using SparsityLevelVAR ({}, d_model={}, layers={}, global_tokens={})".format(
                coeff_mode,
                args.var_d_model,
                args.var_layers,
                args.var_global_tokens,
            )
        )

    wandb_run = None
    if is_main_process:
        args.out_dir = str(stage2_dir)
        args.out_root = str(run_dir)
        args.launch_timestamp = run_dir.name
        wandb_run = _init_wandb(args)
    _barrier()

    if args.stage2_epochs <= 0:
        if is_main_process and wandb_run is not None:
            wandb_run.finish()
        _cleanup_distributed()
        return

    var_model_stage2 = DDP(var_model, device_ids=[local_rank], output_device=local_rank) if distributed else var_model
    train_stage2_var(
        model=var_model_stage2,
        token_loader=token_loader,
        device=device,
        epochs=args.stage2_epochs,
        lr=args.stage2_lr,
        rq_atom_loss_weight=args.stage2_rq_atom_loss_weight,
        rq_coeff_loss_weight=args.stage2_rq_coeff_loss_weight,
        coeff_loss_weight=args.stage2_coeff_loss_weight,
        coeff_loss_type=args.stage2_coeff_loss_type,
        coeff_huber_delta=args.stage2_coeff_huber_delta,
        coeff_depth_weighting=args.coeff_depth_weighting,
        coeff_focal_gamma=args.coeff_focal_gamma,
        stage2_amp=args.stage2_amp,
        stage2_amp_dtype=args.stage2_amp_dtype,
        out_dir=str(stage2_dir),
        ae_for_decode=laser,
        H=H,
        W=W,
        D=D,
        sample_every_steps=args.stage2_sample_every_steps,
        sample_batch_size=args.stage2_sample_batch_size,
        sample_candidate_factor=args.stage2_sample_candidate_factor,
        sample_temperature=args.stage2_sample_temperature,
        sample_temperature_end=(None if args.stage2_sample_temperature_end <= 0 else args.stage2_sample_temperature_end),
        sample_top_k=(None if args.stage2_sample_top_k <= 0 else args.stage2_sample_top_k),
        sample_top_p=(None if args.stage2_sample_top_p <= 0 else args.stage2_sample_top_p),
        sample_image_size=args.stage2_sample_image_size,
        sample_reference_stats=sample_reference_stats,
        token_sampler=token_sampler,
        is_main_process=is_main_process,
        wandb_run=wandb_run,
        warmup_steps=args.stage2_warmup_steps,
        min_lr_ratio=args.stage2_min_lr_ratio,
        weight_decay=args.stage2_weight_decay,
    )

    if is_main_process:
        if wandb_run is not None:
            wandb_run.finish()
        print("Outputs saved to: {}".format(stage2_dir))
    _barrier()
    _cleanup_distributed()


if __name__ == "__main__":
    main()
