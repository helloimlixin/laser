"""Short stage-2 sampling helpers."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
from torchvision.utils import save_image

from src.stage2_compat import (
    Stage1DecodeBundle,
    decode_stage2_outputs,
    ensure_stage2_cache_metadata,
    extract_state_dict,
    load_stage1_decoder_bundle,
    load_torch_payload,
)
from src.stage2_paths import infer_latest_stage2_checkpoint, infer_latest_token_cache


S1_META_KEYS = {
    "stage1_checkpoint",
    "patch_based",
    "patch_size",
    "patch_stride",
    "patch_reconstruction",
    "variational_coeffs",
    "variational_coeff_prior_std",
    "variational_coeff_min_std",
    "image_size",
    "latent_hw",
    "coeff_vocab_size",
    "n_bins",
    "coeff_bin_values",
    "coeff_max",
    "coef_max",
    "coef_quantization",
    "coef_mu",
    "quantize_sparse_coeffs",
}

load_token_cache = None


@dataclass
class Run:
    ckpt: Path
    cache_pt: Path
    cache: dict
    hps: dict
    net: object
    s1: Stage1DecodeBundle
    shape: Tuple[int, int, int]
    dev: torch.device


@dataclass
class Batch:
    imgs: torch.Tensor
    toks: Optional[torch.Tensor] = None
    atoms: Optional[torch.Tensor] = None
    coeffs: Optional[torch.Tensor] = None


def norm_arch(raw, *, allow_auto: bool = True) -> str:
    text = str(raw or "").strip().lower()
    if text in {"", "auto"}:
        return "auto" if allow_auto else ""
    if text in {"sparse_spatial_depth", "spatial_depth"}:
        return "spatial_depth"
    if text in {"gpt", "mingpt"}:
        return "gpt"
    raise ValueError(f"Unsupported stage-2 architecture: {raw!r}")


def pick_dev(arg) -> torch.device:
    if isinstance(arg, torch.device):
        return arg
    text = str(arg).strip().lower()
    if text in {"", "auto", "gpu"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(arg))


def pick_nrow(n: int, nrow: Optional[int] = None) -> int:
    if nrow is not None and int(nrow) > 0:
        return int(nrow)
    root = int(math.sqrt(int(n)))
    if root * root == int(n):
        return max(1, root)
    return max(1, int(math.ceil(math.sqrt(int(n)))))


def sample_dir(ckpt) -> Path:
    ckpt = Path(ckpt).expanduser().resolve()
    return ckpt.parent / f"{ckpt.stem}_samples"


def gen_dir(ckpt, *, ar_dir="outputs/ar") -> Path:
    ckpt = Path(ckpt).expanduser().resolve()
    return Path(ar_dir).expanduser().resolve() / "generated" / ckpt.parent.name


def _hps(payload) -> dict:
    raw = payload.get("hyper_parameters") if isinstance(payload, dict) else None
    return dict(raw or {}) if isinstance(raw, dict) else {}


def _state(payload) -> dict:
    state = extract_state_dict(payload)
    if not isinstance(state, dict):
        raise RuntimeError("Expected a checkpoint with a state_dict")
    return state


def _count_ix(state: dict, prefix: str) -> int:
    idx = set()
    for key in state:
        if not key.startswith(prefix):
            continue
        tail = key[len(prefix):].split(".", 1)[0]
        if tail.isdigit():
            idx.add(int(tail))
    return len(idx)


def _find_key(state: dict, prefix: str, suffix: str) -> str:
    for key in state:
        if key.startswith(prefix) and key.endswith(suffix):
            return key
    raise KeyError(f"Missing state key for prefix={prefix!r} suffix={suffix!r}")


def _pick_arch(state: dict, hps: dict, arch: str) -> str:
    picked = norm_arch(arch, allow_auto=True)
    if picked != "auto":
        return picked
    saved = norm_arch(hps.get("prior_architecture"), allow_auto=False)
    if saved:
        return saved
    if any(key.startswith("prior.spatial_blocks.") for key in state):
        return "spatial_depth"
    if "prior.pos_emb" in state:
        return "gpt"
    raise ValueError("Could not infer prior architecture from the checkpoint.")


def _coeff_vals(
    n_bins: int,
    cache: dict,
    hps: dict,
    *,
    cmax=None,
    cquant=None,
    cmu=None,
) -> torch.Tensor:
    meta = cache.get("meta", {}) if isinstance(cache, dict) else {}
    for box in (meta, cache, hps):
        raw = box.get("coeff_bin_values") if isinstance(box, dict) else None
        if raw is not None and cmax is None and cquant is None and cmu is None:
            vals = torch.as_tensor(raw, dtype=torch.float32).reshape(-1)
            if vals.numel() == int(n_bins):
                return vals

    coeff_max = float(
        cmax
        if cmax is not None
        else meta.get("coef_max", hps.get("prior_coeff_max"))
    )
    quant = str(cquant if cquant is not None else meta.get("coef_quantization", "uniform")).strip().lower()
    mu = float(cmu if cmu is not None else meta.get("coef_mu", 0.0))

    if quant == "uniform":
        return torch.linspace(-coeff_max, coeff_max, steps=int(n_bins), dtype=torch.float32)
    if quant != "mu_law":
        raise ValueError(f"Unsupported coeff quantization: {quant!r}")
    if mu <= 0.0:
        raise ValueError(f"coeff mu must be > 0, got {mu}")
    if int(n_bins) == 1:
        return torch.zeros(1, dtype=torch.float32)
    z = torch.linspace(-1.0, 1.0, steps=int(n_bins), dtype=torch.float32)
    vals = torch.sign(z) * (torch.expm1(z.abs() * math.log1p(mu)) / mu)
    return vals * coeff_max


def _patch_codec(cache: dict, hps: dict, *, cmax=None, cquant=None, cmu=None) -> dict:
    if cache.get("coeffs_flat") is not None:
        return cache

    out = dict(cache)
    meta = dict(cache.get("meta", {}) or {})
    raw = meta.get("coeff_bin_values")
    if raw is None and cache.get("coeff_bin_values") is not None:
        raw = cache["coeff_bin_values"]
    need = raw is None or cmax is not None or cquant is not None or cmu is not None
    if not need:
        out["meta"] = meta
        return out

    n_bins = int(
        meta.get("coeff_vocab_size")
        or meta.get("n_bins")
        or hps.get("resolved_coeff_vocab_size")
        or hps.get("prior_coeff_vocab_size")
        or 0
    )
    if n_bins <= 0 and raw is not None:
        n_bins = int(torch.as_tensor(raw).numel())
    if n_bins > 0:
        meta["coeff_bin_values"] = _coeff_vals(n_bins, cache, hps, cmax=cmax, cquant=cquant, cmu=cmu)
        meta["coeff_vocab_size"] = n_bins
        meta["n_bins"] = n_bins
        if cmax is not None:
            meta["coeff_max"] = float(cmax)
            meta["coef_max"] = float(cmax)
    out["meta"] = meta
    return out


def _swap_s1(cache: dict, s1_ckpt) -> dict:
    out = dict(cache)
    meta = dict(cache.get("meta", {}) or {})
    for key in S1_META_KEYS:
        meta.pop(key, None)
    meta["stage1_checkpoint"] = str(Path(s1_ckpt).expanduser().resolve())
    out["meta"] = meta
    return out


def _pick_cache(cache_pt, hps: dict, ar_dir) -> Path:
    if cache_pt is not None:
        picked = Path(cache_pt).expanduser().resolve()
        if not picked.exists():
            raise FileNotFoundError(f"Token cache not found: {picked}")
        return picked

    saved = hps.get("token_cache_path")
    if saved is not None:
        picked = Path(str(saved)).expanduser().resolve()
        if picked.exists():
            return picked

    picked = infer_latest_token_cache(ar_output_dir=ar_dir)
    if picked is not None:
        return Path(picked).expanduser().resolve()

    if saved is not None:
        raise FileNotFoundError(f"Saved token cache path is stale: {saved}")
    raise FileNotFoundError("Could not infer a token cache. Pass --token-cache explicitly.")


def _legacy_hps(payload, cache: dict, *, arch="auto", heads=None, drop=None, cmax=None) -> dict:
    state = _state(payload)
    hps = _hps(payload)
    arch = _pick_arch(state, hps, arch)
    meta = cache.get("meta", {}) if isinstance(cache, dict) else {}
    out = dict(hps)

    out["prior_architecture"] = arch
    out["prior_d_model"] = int(state["prior.token_emb.weight"].shape[1])
    out["prior_n_heads"] = int(heads or hps.get("prior_n_heads") or 8)
    out["prior_dropout"] = float(drop if drop is not None else hps.get("prior_dropout", 0.0))
    out["resolved_total_vocab_size"] = int(
        hps.get("resolved_total_vocab_size") or state["prior.token_head.weight"].shape[0]
    )

    if arch == "spatial_depth":
        out["prior_d_ff"] = int(
            hps.get("prior_d_ff")
            or state[_find_key(state, "prior.spatial_blocks.", ".ffn.0.weight")].shape[0]
        )
        out["prior_n_spatial_layers"] = int(
            hps.get("prior_n_spatial_layers") or _count_ix(state, "prior.spatial_blocks.")
        )
        out["prior_n_depth_layers"] = int(
            hps.get("prior_n_depth_layers") or _count_ix(state, "prior.depth_blocks.")
        )
        out["prior_n_global_spatial_tokens"] = int(
            hps.get("prior_n_global_spatial_tokens")
            or (state["prior.global_spatial_tokens"].shape[1] if "prior.global_spatial_tokens" in state else 0)
        )
    else:
        out["prior_d_ff"] = int(
            hps.get("prior_d_ff")
            or state[_find_key(state, "prior.blocks.", ".ffn.0.weight")].shape[0]
        )
        out["prior_n_layers"] = int(hps.get("prior_n_layers") or _count_ix(state, "prior.blocks."))
        out["prior_n_global_spatial_tokens"] = int(
            hps.get("prior_n_global_spatial_tokens")
            or (state["prior.global_spatial_tokens"].shape[1] if "prior.global_spatial_tokens" in state else 0)
        )

    if cache.get("coeffs_flat") is not None:
        atom = int(hps.get("resolved_atom_vocab_size") or hps.get("prior_atom_vocab_size") or out["resolved_total_vocab_size"])
        out["prior_real_valued_coeffs"] = True
        out["prior_gaussian_coeffs"] = bool(hps.get("prior_gaussian_coeffs", meta.get("variational_coeffs", False)))
        out["prior_autoregressive_coeffs"] = bool(hps.get("prior_autoregressive_coeffs", True))
        out["prior_atom_vocab_size"] = atom
        out["resolved_atom_vocab_size"] = atom
        out["prior_coeff_vocab_size"] = 0
        out["resolved_coeff_vocab_size"] = 0
        out["prior_coeff_max"] = float(
            cmax if cmax is not None else hps.get("prior_coeff_max", meta.get("coeff_max", meta.get("coef_max", 24.0)))
        )
        out["prior_coeff_prior_std"] = float(
            hps.get("prior_coeff_prior_std", meta.get("variational_coeff_prior_std", 0.25))
        )
        out["prior_coeff_min_std"] = float(
            hps.get("prior_coeff_min_std", meta.get("variational_coeff_min_std", 0.01))
        )
    else:
        atom = hps.get("resolved_atom_vocab_size") or hps.get("prior_atom_vocab_size") or meta.get("num_atoms")
        coeff = hps.get("resolved_coeff_vocab_size") or hps.get("prior_coeff_vocab_size") or meta.get("coeff_vocab_size") or meta.get("n_bins")
        if atom is not None:
            out["prior_atom_vocab_size"] = int(atom)
            out["resolved_atom_vocab_size"] = int(atom)
        if coeff is not None:
            out["prior_coeff_vocab_size"] = int(coeff)
            out["resolved_coeff_vocab_size"] = int(coeff)
        coeff_max = cmax if cmax is not None else hps.get("prior_coeff_max", meta.get("coeff_max", meta.get("coef_max")))
        if coeff_max is not None:
            out["prior_coeff_max"] = float(coeff_max)
        out["prior_real_valued_coeffs"] = False

    return out


def _make_net(
    ckpt: Path,
    cache: dict,
    *,
    payload=None,
    arch="auto",
    heads=None,
    drop=None,
    cmax=None,
    cquant=None,
    cmu=None,
):
    from src.models.sparse_token_prior import SparseTokenPriorModule, build_sparse_prior_from_hparams

    payload = load_torch_payload(ckpt) if payload is None else payload
    state = _state(payload)
    hps = _hps(payload)
    cache = _patch_codec(cache, hps, cmax=cmax, cquant=cquant, cmu=cmu)

    merged = dict(hps)
    picked_arch = norm_arch(arch, allow_auto=True)
    if picked_arch != "auto":
        merged["prior_architecture"] = picked_arch
    if heads is not None:
        merged["prior_n_heads"] = int(heads)
    if drop is not None:
        merged["prior_dropout"] = float(drop)
    if cmax is not None:
        merged["prior_coeff_max"] = float(cmax)

    try:
        prior = build_sparse_prior_from_hparams(cache, hparams=merged)
    except Exception:
        merged = _legacy_hps(payload, cache, arch=picked_arch, heads=heads, drop=drop, cmax=cmax)
        prior = build_sparse_prior_from_hparams(cache, hparams=merged)

    net = SparseTokenPriorModule(
        prior=prior,
        learning_rate=float(merged.get("learning_rate", 3e-4)),
        weight_decay=float(merged.get("weight_decay", 0.01)),
        warmup_steps=int(merged.get("warmup_steps", 1000)),
        min_lr_ratio=float(merged.get("min_lr_ratio", 0.01)),
        atom_loss_weight=float(merged.get("atom_loss_weight", 1.0)),
        coeff_loss_weight=float(merged.get("coeff_loss_weight", 1.0)),
        coeff_depth_weighting=str(merged.get("coeff_depth_weighting", "none")),
        coeff_focal_gamma=float(merged.get("coeff_focal_gamma", 0.0)),
        coeff_loss_type=merged.get("coeff_loss_type", "auto"),
        coeff_huber_delta=float(merged.get("coeff_huber_delta", 0.5)),
        sample_coeff_temperature=merged.get("sample_coeff_temperature"),
        sample_coeff_mode=str(merged.get("sample_coeff_mode") or "gaussian"),
    )
    net.load_state_dict(state)
    net.save_hyperparameters(merged)
    return net, merged, cache


def load_run(
    *,
    ckpt=None,
    cache_pt=None,
    s1_ckpt=None,
    dev="auto",
    out_root="outputs",
    ar_dir="outputs/ar",
    arch="auto",
    heads=None,
    drop=None,
    cmax=None,
    cquant=None,
    cmu=None,
) -> Run:
    dev = pick_dev(dev)

    if ckpt is None:
        ckpt = infer_latest_stage2_checkpoint(ar_output_dir=ar_dir)
        if ckpt is None:
            raise FileNotFoundError("Could not infer a stage-2 checkpoint.")
    ckpt = Path(ckpt).expanduser().resolve()

    payload = load_torch_payload(ckpt)
    hps = _hps(payload)
    cache_pt = _pick_cache(cache_pt, hps, ar_dir)

    loader = load_token_cache
    if loader is None:
        from src.data.token_cache import load_token_cache as loader

    raw = loader(cache_pt)
    if s1_ckpt is not None:
        s1_ckpt = Path(s1_ckpt).expanduser().resolve()
        if not s1_ckpt.exists():
            raise FileNotFoundError(f"Stage-1 checkpoint not found: {s1_ckpt}")
        raw = _swap_s1(raw, s1_ckpt)

    cache = ensure_stage2_cache_metadata(
        raw,
        token_cache_path=cache_pt,
        output_root=out_root,
    )
    net, hps, cache = _make_net(
        ckpt,
        cache,
        payload=payload,
        arch=arch,
        heads=heads,
        drop=drop,
        cmax=cmax,
        cquant=cquant,
        cmu=cmu,
    )
    net = net.eval().to(dev)

    s1 = load_stage1_decoder_bundle(
        cache,
        token_cache_path=cache_pt,
        device=dev,
        output_root=out_root,
    )
    prior_cfg = getattr(getattr(net, "prior", None), "cfg", None)
    if prior_cfg is not None and all(hasattr(prior_cfg, key) for key in ("H", "W", "D")):
        shape = (int(prior_cfg.H), int(prior_cfg.W), int(prior_cfg.D))
    else:
        shape = tuple(int(v) for v in cache["shape"])

    return Run(
        ckpt=ckpt,
        cache_pt=cache_pt,
        cache=cache,
        hps=hps,
        net=net,
        s1=s1,
        shape=shape,
        dev=dev,
    )


@torch.no_grad()
def sample(
    net,
    s1: Stage1DecodeBundle,
    shape: Tuple[int, int, int],
    *,
    n: int,
    bs: Optional[int] = None,
    temp: float = 1.0,
    top_k: Optional[int] = None,
    ctemp: Optional[float] = None,
    cmode: Optional[str] = None,
    dev=None,
) -> Batch:
    h, w, d = shape
    bs = max(1, int(bs or n))
    left = int(n)
    top_k = None if top_k is None or int(top_k) <= 0 else int(top_k)
    dev = pick_dev(dev or getattr(net, "device", "cpu"))

    imgs = []
    toks = []
    atoms = []
    coeffs = []

    while left > 0:
        cur = min(bs, left)
        out = net.generate_sparse_codes(
            cur,
            temperature=float(temp),
            top_k=top_k,
            coeff_temperature=ctemp,
            coeff_sample_mode=cmode,
        )
        if bool(getattr(net.prior, "real_valued_coeffs", False)):
            atom_ids, vals = out
            atom_grid = atom_ids.view(cur, h, w, d)
            coeff_grid = vals.view(cur, h, w, d)
            img = decode_stage2_outputs(
                s1,
                atom_grid,
                coeff_grid,
                device=dev,
            ).detach().cpu()
            atoms.append(atom_grid.detach().cpu())
            coeffs.append(coeff_grid.detach().cpu())
        else:
            tok_grid = out.view(cur, h, w, d)
            img = decode_stage2_outputs(
                s1,
                tok_grid,
                device=dev,
            ).detach().cpu()
            toks.append(tok_grid.detach().cpu())
        imgs.append(img)
        left -= cur

    return Batch(
        imgs=torch.cat(imgs, dim=0),
        toks=(torch.cat(toks, dim=0) if toks else None),
        atoms=(torch.cat(atoms, dim=0) if atoms else None),
        coeffs=(torch.cat(coeffs, dim=0) if coeffs else None),
    )


def _window_origin(pos: int, total: int, window: int) -> int:
    if window >= total:
        return 0
    return min(max(int(pos) - int(window) + 1, 0), int(total) - int(window))


@torch.no_grad()
def sample_slide(
    net,
    s1: Stage1DecodeBundle,
    shape: Tuple[int, int, int],
    *,
    out_h: int,
    out_w: int,
    n: int,
    bs: Optional[int] = None,
    temp: float = 1.0,
    top_k: Optional[int] = None,
    dev=None,
) -> Batch:
    crop_h, crop_w, d = (int(shape[0]), int(shape[1]), int(shape[2]))
    out_h = int(out_h)
    out_w = int(out_w)
    if out_h < crop_h or out_w < crop_w:
        raise ValueError(
            f"Sliding-window sampling expects target latent shape >= training crop; "
            f"got target {(out_h, out_w)} and crop {(crop_h, crop_w)}"
        )
    prior = getattr(net, "prior", None)
    if prior is None or bool(getattr(prior, "real_valued_coeffs", False)):
        raise ValueError("Sliding-window high-resolution sampling is currently supported only for quantized GPT priors.")
    if "gpt" not in type(prior).__name__.lower():
        raise ValueError("Sliding-window high-resolution sampling currently supports only gpt stage-2 checkpoints.")

    bs = max(1, int(bs or n))
    left = int(n)
    top_k = None if top_k is None or int(top_k) <= 0 else int(top_k)
    dev = pick_dev(dev or getattr(net, "device", "cpu"))

    imgs = []
    toks = []
    while left > 0:
        cur = min(bs, left)
        full = torch.zeros(cur, out_h, out_w, d, dtype=torch.long, device=dev)
        for r in range(out_h):
            for c in range(out_w):
                top = _window_origin(r, out_h, crop_h)
                left_col = _window_origin(c, out_w, crop_w)
                local_r = int(r - top)
                local_c = int(c - left_col)

                # Re-run the crop-trained prior on a local prompt window and keep
                # only the current site, matching the paper's sliding-window idea.
                prompt_tokens = full[:, top : top + crop_h, left_col : left_col + crop_w, :].clone()
                rr = torch.arange(top, top + crop_h, device=dev).view(1, crop_h, 1, 1)
                cc = torch.arange(left_col, left_col + crop_w, device=dev).view(1, 1, crop_w, 1)
                prompt_mask = ((rr < r) | ((rr == r) & (cc < c))).expand(cur, crop_h, crop_w, d)

                window_tokens = prior.generate(
                    batch_size=cur,
                    temperature=float(temp),
                    top_k=top_k,
                    prompt_tokens=prompt_tokens.view(cur, crop_h * crop_w, d),
                    prompt_mask=prompt_mask.reshape(cur, crop_h * crop_w, d),
                    show_progress=False,
                )
                full[:, r, c, :] = window_tokens[:, local_r * crop_w + local_c, :]

        img = decode_stage2_outputs(
            s1,
            full,
            device=dev,
        ).detach().cpu()
        imgs.append(img)
        toks.append(full.detach().cpu())
        left -= cur

    return Batch(
        imgs=torch.cat(imgs, dim=0),
        toks=torch.cat(toks, dim=0),
        atoms=None,
        coeffs=None,
    )


def save_grid(imgs: torch.Tensor, out_dir, *, stem="samples", nrow: Optional[int] = None) -> Tuple[Path, Path]:
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    nrow = pick_nrow(int(imgs.size(0)), nrow)
    raw = out_dir / f"{stem}.png"
    auto = out_dir / f"{stem}_auto.png"
    save_image(imgs, raw, nrow=nrow, normalize=True, value_range=(-1.0, 1.0))
    save_image(imgs, auto, nrow=nrow, normalize=True, scale_each=True)
    return raw, auto


def pack_dump(batch: Batch, run: Run) -> dict:
    data = {
        "shape": run.shape,
        "latent_hw": run.s1.latent_hw,
        "stage1_checkpoint": str(run.s1.checkpoint_path),
        "stage2_checkpoint": str(run.ckpt),
        "token_cache": str(run.cache_pt),
    }
    if batch.atoms is not None and batch.coeffs is not None:
        data["atom_ids"] = batch.atoms
        data["coeffs"] = batch.coeffs
    elif batch.toks is not None:
        data["tokens"] = batch.toks
    return data


def save_dump(batch: Batch, run: Run, path) -> Path:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pack_dump(batch, run), path)
    return path


# Compatibility aliases for the earlier refactor.
Stage2Runtime = Run
Stage2SampleBatch = Batch
resolve_stage2_device = pick_dev
resolve_stage2_nrow = pick_nrow
default_stage2_sample_dir = sample_dir
default_generate_output_dir = gen_dir
load_stage2_runtime = load_run
generate_stage2_samples = sample
save_stage2_sample_grids = save_grid
build_stage2_sample_payload = pack_dump
save_stage2_sample_payload = save_dump
