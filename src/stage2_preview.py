"""Stage-2 sample previews: decode generated tokens and save/log media."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Mapping, Optional

import lightning as pl
import numpy as np
import torch
from PIL import Image, ImageDraw

from src.audio_logging import (
    audio_config_from_source,
    build_generated_audio_log_payload,
    normalize_generated_waveform_for_preview,
)
from src.data.imagenet_labels import class_names_for_dataset, imagenet_synsets_from_names, ordered_class_names
from src.data.token_cache import load_token_cache
from src.s2 import pick_nrow, sample as sample_s2, sample_slide, save_grid
from src.stage2_compat import (
    ensure_stage2_cache_metadata as add_cache_meta,
    load_stage1_decoder_bundle as load_s1,
)
from src.text_conditioning import encode_prompts_from_cache
from src.wandb_media import log_wandb_images, log_wandb_payload


def _format_variant_number(value: float) -> str:
    return f"{float(value):.3g}".replace("-", "m").replace(".", "p")


def _sanitize_sample_variant_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(name).strip())
    return cleaned.strip("_") or "sample"


def _parse_variant_float(value) -> float:
    return float(str(value).replace("p", "."))


def _sample_variant_from_spec(spec, *, fallback_ctemp, fallback_cmode) -> dict[str, object]:
    if isinstance(spec, Mapping):
        temp = spec.get("temperature", spec.get("temp", spec.get("t")))
        top_k = spec.get("top_k", spec.get("topk", spec.get("k")))
        if temp is None or top_k is None:
            raise ValueError(f"Sample variant mappings need temperature/temp and top_k/topk fields: {spec!r}")
        ctemp = spec.get("coeff_temperature", spec.get("ctemp", fallback_ctemp))
        cmode = spec.get("coeff_mode", spec.get("cmode", fallback_cmode))
        name = spec.get("name")
    else:
        raw = str(spec).strip()
        match_temp = re.search(r"(?:temperature|temp|t)[_=\-]?([0-9]+(?:[p.][0-9]+)?)", raw, re.I)
        match_top_k = re.search(r"(?:top[_\-]?k|topk|k)[_=\-]?([0-9]+)", raw, re.I)
        if match_temp is None or match_top_k is None:
            raise ValueError(
                "Sample variant strings must look like 't0p60_k32' or "
                f"'temp0.60_topk32', got {raw!r}"
            )
        temp = _parse_variant_float(match_temp.group(1))
        top_k = int(match_top_k.group(1))
        ctemp = fallback_ctemp
        cmode = fallback_cmode
        name = raw

    temp_f = float(temp)
    top_k_i = int(top_k)
    if name is None or str(name).strip() == "":
        name = f"t{_format_variant_number(temp_f)}_k{top_k_i}"
    return {
        "name": _sanitize_sample_variant_name(str(name)),
        "temp": temp_f,
        "top_k": top_k_i,
        "ctemp": None if ctemp is None else float(ctemp),
        "cmode": None if cmode is None else str(cmode).strip().lower(),
    }


def _resolve_sample_variants(
    specs,
    *,
    temp: float,
    top_k: int,
    ctemp,
    cmode,
) -> list[dict[str, object]]:
    if specs is None or specs == "":
        return [
            {
                "name": "",
                "temp": float(temp),
                "top_k": int(top_k),
                "ctemp": None if ctemp is None else float(ctemp),
                "cmode": None if cmode is None else str(cmode).strip().lower(),
            }
        ]
    if isinstance(specs, str):
        raw_specs = [item.strip() for item in specs.split(",") if item.strip()]
    elif isinstance(specs, Mapping):
        raw_specs = [specs]
    else:
        raw_specs = list(specs)
    variants = [
        _sample_variant_from_spec(item, fallback_ctemp=ctemp, fallback_cmode=cmode)
        for item in raw_specs
    ]
    if not variants:
        raise ValueError("sample_variants was provided but no valid variants were found")
    seen: set[str] = set()
    for variant in variants:
        name = str(variant["name"])
        if name in seen:
            raise ValueError(f"Duplicate sample variant name: {name}")
        seen.add(name)
    return variants


def _prior_token_shape(mod: pl.LightningModule, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    prior_cfg = getattr(getattr(mod, "prior", None), "cfg", None)
    if prior_cfg is not None and all(hasattr(prior_cfg, key) for key in ("H", "W", "D")):
        return tuple(int(getattr(prior_cfg, key)) for key in ("H", "W", "D"))
    return tuple(int(v) for v in fallback)


def _sample_for_preview(
    mod: pl.LightningModule,
    s1,
    *,
    prior_shape: tuple[int, int, int],
    full_shape: tuple[int, int, int],
    n: int,
    temp: float,
    top_k: int,
    ctemp,
    cmode,
    class_labels: Optional[torch.Tensor] = None,
    text_tokens: Optional[torch.Tensor] = None,
    text_mask: Optional[torch.Tensor] = None,
    dev: torch.device,
):
    """Sample full-size previews when a GPT prior was trained on latent crops."""
    prior = getattr(mod, "prior", None)
    prior_name = type(prior).__name__.lower() if prior is not None else ""
    full_shape = tuple(int(v) for v in full_shape)
    prior_shape = tuple(int(v) for v in prior_shape)
    if (
        prior is not None
        and "gpt" in prior_name
        and prior_shape[:2] != full_shape[:2]
        and prior_shape[2] == full_shape[2]
    ):
        return sample_slide(
            mod,
            s1,
            prior_shape,
            out_h=full_shape[0],
            out_w=full_shape[1],
            n=n,
            temp=temp,
            top_k=top_k,
            dev=dev,
        )
    return sample_s2(
        mod,
        s1,
        prior_shape,
        n=n,
        temp=temp,
        top_k=top_k,
        ctemp=ctemp,
        cmode=cmode,
        class_labels=class_labels,
        text_tokens=text_tokens,
        text_mask=text_mask,
        dev=dev,
    )


def _is_waveform_samples(samples: torch.Tensor) -> bool:
    return torch.is_tensor(samples) and samples.ndim == 3 and int(samples.size(1)) == 1


def _grid_uint8(images: list[np.ndarray], *, nrow: int) -> np.ndarray:
    if not images:
        raise ValueError("images must be non-empty")
    nrow = max(1, int(nrow))
    height, width, channels = images[0].shape
    rows = int(math.ceil(len(images) / float(nrow)))
    grid = np.full((rows * height, nrow * width, channels), 255, dtype=np.uint8)
    for idx, image in enumerate(images):
        row = idx // nrow
        col = idx % nrow
        grid[row * height: (row + 1) * height, col * width: (col + 1) * width] = image
    return grid


_DEFAULT_CLASS_NAMES = {
    "cifar10": [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ],
    "stl10": [
        "airplane",
        "bird",
        "car",
        "cat",
        "deer",
        "dog",
        "horse",
        "monkey",
        "ship",
        "truck",
    ],
}


def _class_names_from_cache(cache: Optional[dict]) -> list[str]:
    meta = cache.get("meta", {}) if isinstance(cache, dict) else {}
    raw = meta.get("class_names")
    dataset = str(meta.get("dataset", "")).strip().lower()
    names = class_names_for_dataset(dataset, raw)
    if names:
        return names
    return list(_DEFAULT_CLASS_NAMES.get(dataset, []))


def _class_synsets_from_cache(cache: Optional[dict]) -> list[str]:
    meta = cache.get("meta", {}) if isinstance(cache, dict) else {}
    raw = meta.get("class_synsets")
    synsets = ordered_class_names(raw)
    if synsets:
        return synsets
    dataset = str(meta.get("dataset", "")).strip().lower()
    return imagenet_synsets_from_names(dataset, meta.get("class_names"))


def _class_name_for_label(cache: Optional[dict], label: int) -> str:
    names = _class_names_from_cache(cache)
    if 0 <= int(label) < len(names):
        return str(names[int(label)])
    return ""


def _class_label_display(cache: Optional[dict], label: int) -> str:
    name = _class_name_for_label(cache, label)
    class_label_name = ""
    if isinstance(cache, dict):
        meta = cache.get("meta", {}) or {}
        class_label_name = str(meta.get("class_label_name") or "").strip().lower()
    if name:
        if "speaker" in class_label_name:
            return f"{name} (speaker {int(label)})"
        return f"{name} (class {int(label)})"
    if "speaker" in class_label_name:
        return f"speaker {int(label)}"
    return f"class {int(label)}"


def _class_label_texts(cache: Optional[dict], labels: Optional[torch.Tensor]) -> list[str]:
    if labels is None:
        return []
    label_list = [int(label) for label in labels.detach().cpu().reshape(-1).tolist()]
    return [_class_label_display(cache, label) for label in label_list]


def _class_label_caption(cache: Optional[dict], label: int) -> str:
    return _class_label_display(cache, label)


def _conditioning_captions(
    cache: Optional[dict],
    labels: Optional[torch.Tensor],
    prompts: Optional[list[str]] = None,
    *,
    n: int,
) -> list[str]:
    prompts = list(prompts or [])
    label_list = [] if labels is None else [int(label) for label in labels.detach().cpu().reshape(-1).tolist()]
    captions: list[str] = []
    for idx in range(max(0, int(n))):
        parts = [f"generated audio {idx}"]
        if idx < len(label_list):
            parts.append(f"conditioned on {_class_label_caption(cache, label_list[idx])}")
        if idx < len(prompts) and str(prompts[idx]).strip():
            parts.append(f"text={str(prompts[idx]).strip()}")
        captions.append(" | ".join(parts))
    return captions


def _to_uint8_pil(image: torch.Tensor) -> Image.Image:
    image = image.detach().cpu().to(torch.float32)
    if image.ndim != 3:
        raise ValueError(f"Expected CHW image tensor, got {tuple(image.shape)}")
    image = image.clamp(-1.0, 1.0).add(1.0).mul(127.5).round().to(torch.uint8)
    array = image.permute(1, 2, 0).contiguous().numpy()
    if array.shape[-1] == 1:
        return Image.fromarray(array[..., 0], mode="L").convert("RGB")
    return Image.fromarray(array, mode="RGB")


def _text_width(draw: ImageDraw.ImageDraw, text: str) -> int:
    try:
        left, _, right, _ = draw.textbbox((0, 0), text)
        return int(right - left)
    except Exception:
        return int(draw.textlength(text))


def _fit_text(draw: ImageDraw.ImageDraw, text: str, max_width: int) -> str:
    text = str(text)
    if _text_width(draw, text) <= max_width:
        return text
    suffix = "..."
    keep = max(1, len(text) - 1)
    while keep > 1:
        candidate = text[:keep].rstrip() + suffix
        if _text_width(draw, candidate) <= max_width:
            return candidate
        keep -= 1
    return suffix


def _save_labeled_image_grid(
    imgs: torch.Tensor,
    labels: torch.Tensor,
    cache: Optional[dict],
    out_dir: Path,
    *,
    stem: str,
    nrow: Optional[int] = None,
) -> Optional[Path]:
    label_texts = _class_label_texts(cache, labels)
    if not label_texts:
        return None
    n = min(int(imgs.size(0)), len(label_texts))
    if n <= 0:
        return None
    panels: list[Image.Image] = []
    for image, label_text in zip(imgs[:n], label_texts[:n]):
        pil = _to_uint8_pil(image).convert("RGB")
        width, height = pil.size
        label_height = 28
        panel = Image.new("RGB", (width, height + label_height), "white")
        draw = ImageDraw.Draw(panel)
        fitted = _fit_text(draw, label_text, max(1, width - 6))
        draw.text((3, 3), fitted, fill="black")
        panel.paste(pil, (0, label_height))
        panels.append(panel)
    if not panels:
        return None
    nrow = pick_nrow(len(panels), nrow)
    rows = int(math.ceil(len(panels) / float(max(1, nrow))))
    cell_w, cell_h = panels[0].size
    grid = Image.new("RGB", (nrow * cell_w, rows * cell_h), "white")
    for idx, panel in enumerate(panels):
        row = idx // nrow
        col = idx % nrow
        grid.paste(panel, (col * cell_w, row * cell_h))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}_labeled.png"
    grid.save(path)
    return path


def _write_class_label_manifest(path: Path, labels: Optional[torch.Tensor], cache: Optional[dict]) -> None:
    label_texts = _class_label_texts(cache, labels)
    if labels is None or not label_texts:
        return
    label_list = [int(label) for label in labels.detach().cpu().reshape(-1).tolist()]
    synsets = _class_synsets_from_cache(cache)
    wav_files = []
    if path.is_dir():
        wav_files = sorted(item.name for item in path.glob("*.wav"))
    entries = [
        {
            "index": int(idx),
            "file": wav_files[idx] if idx < len(wav_files) else "",
            "class_id": int(label),
            "class_name": _class_name_for_label(cache, int(label)),
            "class_synset": synsets[int(label)] if 0 <= int(label) < len(synsets) else "",
            "label": text,
            "caption": _class_label_caption(cache, int(label)),
        }
        for idx, (label, text) in enumerate(zip(label_list, label_texts))
    ]
    if path.is_dir():
        txt_path = path / "class_labels.txt"
        tsv_path = path / "class_labels.tsv"
        json_path = path / "class_labels.json"
    else:
        txt_path = path.with_suffix(".class_labels.txt")
        tsv_path = path.with_suffix(".class_labels.tsv")
        json_path = path.with_suffix(".class_labels.json")
    txt_path.write_text("\n".join(label_texts) + "\n", encoding="utf-8")
    tsv_path.write_text(
        "index\tfile\tclass_id\tclass_name\tclass_synset\tlabel\tcaption\n"
        + "\n".join(
            "\t".join(
                str(entry[key])
                for key in ("index", "file", "class_id", "class_name", "class_synset", "label", "caption")
            )
            for entry in entries
        )
        + "\n",
        encoding="utf-8",
    )
    json_path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")


def _write_text_prompt_manifest(path: Path, *, stem: str, prompts: list[str]) -> None:
    if not prompts:
        return
    root = path if path.is_dir() else path.parent
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    for idx, prompt in enumerate(prompts):
        expected = root / f"{stem}_{idx:02d}.wav"
        file_name = expected.name if expected.exists() else ""
        entries.append({"index": int(idx), "file": file_name, "text": str(prompt)})
    (root / "prompts.txt").write_text("\n".join(str(prompt) for prompt in prompts) + "\n", encoding="utf-8")
    (root / "sample_texts.tsv").write_text(
        "index\tfile\ttext\n"
        + "\n".join(f"{entry['index']}\t{entry['file']}\t{entry['text']}" for entry in entries)
        + "\n",
        encoding="utf-8",
    )
    (root / "conditioning.json").write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")


def _colorize_scalar_maps(
    maps: torch.Tensor,
    *,
    cmap_name: str,
    per_map: bool,
    value_min: Optional[float] = None,
    value_max: Optional[float] = None,
) -> list[np.ndarray]:
    maps = torch.nan_to_num(maps.detach().cpu().to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
    try:
        import matplotlib

        cmap = matplotlib.colormaps.get_cmap(cmap_name)
    except Exception:
        cmap = None

    out: list[np.ndarray] = []
    if maps.numel() == 0:
        return out
    if not per_map:
        global_min = float(maps.min().item()) if value_min is None else float(value_min)
        global_max = float(maps.max().item()) if value_max is None else float(value_max)
    for scalar_map in maps:
        if per_map:
            lo = float(scalar_map.min().item()) if value_min is None else float(value_min)
            hi = float(scalar_map.max().item()) if value_max is None else float(value_max)
        else:
            lo, hi = global_min, global_max
        if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
            norm = torch.zeros_like(scalar_map)
        else:
            norm = ((scalar_map - lo) / (hi - lo)).clamp(0.0, 1.0)
        arr = norm.numpy()
        if cmap is not None:
            rgb = (cmap(arr)[..., :3] * 255.0).round().astype(np.uint8)
        else:
            gray = (arr * 255.0).round().astype(np.uint8)
            rgb = np.stack([gray, gray, gray], axis=-1)
        out.append(rgb)
    return out


def _save_scalar_map_grid(
    values: torch.Tensor,
    out_dir: Path,
    *,
    stem: str,
    max_items: int,
    max_depths: int,
    cmap_name: str,
    per_map: bool,
    value_min: Optional[float] = None,
    value_max: Optional[float] = None,
) -> Optional[Path]:
    if not torch.is_tensor(values) or values.ndim != 4:
        return None
    B, H, W, D = values.shape
    if B <= 0 or H <= 0 or W <= 0 or D <= 0:
        return None
    keep_b = min(int(max_items), int(B))
    keep_d = min(int(max_depths), int(D))
    maps = (
        values[:keep_b, :, :, :keep_d]
        .permute(0, 3, 1, 2)
        .reshape(keep_b * keep_d, H, W)
    )
    colored = _colorize_scalar_maps(
        maps,
        cmap_name=cmap_name,
        per_map=per_map,
        value_min=value_min,
        value_max=value_max,
    )
    if not colored:
        return None
    max_side = max(int(colored[0].shape[0]), int(colored[0].shape[1]))
    scale = max(1, int(math.ceil(96.0 / float(max(1, max_side)))))
    if scale > 1:
        resampling = getattr(getattr(Image, "Resampling", Image), "NEAREST", Image.NEAREST)
        colored = [
            np.asarray(
                Image.fromarray(image).resize(
                    (int(image.shape[1]) * scale, int(image.shape[0]) * scale),
                    resampling,
                )
            )
            for image in colored
        ]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.png"
    Image.fromarray(_grid_uint8(colored, nrow=keep_d)).save(path)
    return path


def _sparse_visual_tensors(
    batch,
    cache: Optional[dict],
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    atoms = getattr(batch, "atoms", None)
    coeffs = getattr(batch, "coeffs", None)
    coeff_bins = None
    if atoms is not None:
        return atoms, coeffs, coeff_bins

    tokens = getattr(batch, "toks", None)
    if tokens is None:
        return None, None, None
    tokens = tokens.detach()
    if tokens.ndim != 4:
        return None, None, None
    if int(tokens.size(-1)) >= 2 and int(tokens.size(-1)) % 2 == 0:
        atoms = tokens[..., 0::2]
        meta = cache.get("meta", {}) if isinstance(cache, dict) else {}
        atom_vocab = int(meta.get("num_atoms") or meta.get("atom_vocab_size") or 0)
        coeff_bins = tokens[..., 1::2]
        if atom_vocab > 0:
            coeff_bins = coeff_bins - atom_vocab
        return atoms, None, coeff_bins
    return tokens, None, None


def _build_sparse_visuals(batch, cache: Optional[dict], out_dir: Path, *, stem: str) -> dict[str, Path]:
    atoms, coeffs, coeff_bins = _sparse_visual_tensors(batch, cache)
    paths: dict[str, Path] = {}
    meta = cache.get("meta", {}) if isinstance(cache, dict) else {}
    atom_vocab = int(meta.get("num_atoms") or meta.get("atom_vocab_size") or 0)
    if atoms is not None:
        path = _save_scalar_map_grid(
            atoms,
            out_dir,
            stem=f"{stem}_atom_ids",
            max_items=4,
            max_depths=8,
            cmap_name="turbo",
            per_map=False,
            value_min=0.0,
            value_max=float(max(atom_vocab - 1, 1)) if atom_vocab > 0 else None,
        )
        if path is not None:
            paths["atom_id_maps"] = path
    if coeffs is not None:
        coeff_abs = torch.nan_to_num(coeffs.detach().cpu().to(torch.float32).abs())
        coeff_limit_value = float(coeff_abs.max().item()) if coeff_abs.numel() > 0 else 0.0
        path = _save_scalar_map_grid(
            coeffs,
            out_dir,
            stem=f"{stem}_coeff_values",
            max_items=4,
            max_depths=8,
            cmap_name="coolwarm",
            per_map=False,
            value_min=-coeff_limit_value,
            value_max=coeff_limit_value,
        )
        if path is not None:
            paths["coeff_value_maps"] = path
        path = _save_scalar_map_grid(
            coeffs.abs(),
            out_dir,
            stem=f"{stem}_coeff_abs",
            max_items=4,
            max_depths=8,
            cmap_name="magma",
            per_map=True,
        )
        if path is not None:
            paths["coeff_abs_maps"] = path
    if coeff_bins is not None:
        coeff_vocab = int(meta.get("n_bins") or meta.get("coeff_vocab_size") or 0)
        coeff_bin_values = meta.get("coeff_bin_values")
        if coeff_vocab <= 0 and coeff_bin_values is not None:
            coeff_vocab = int(torch.as_tensor(coeff_bin_values).numel())
        path = _save_scalar_map_grid(
            coeff_bins,
            out_dir,
            stem=f"{stem}_coeff_bins",
            max_items=4,
            max_depths=8,
            cmap_name="viridis",
            per_map=False,
            value_min=0.0 if coeff_vocab > 0 else None,
            value_max=float(max(coeff_vocab - 1, 1)) if coeff_vocab > 0 else None,
        )
        if path is not None:
            paths["coeff_bin_maps"] = path
    return paths


def _sparse_generation_stats(batch, cache: Optional[dict]) -> dict[str, object]:
    atoms, coeffs, coeff_bins = _sparse_visual_tensors(batch, cache)
    stats: dict[str, object] = {}
    if atoms is not None:
        atoms_flat = atoms.detach().cpu().to(torch.float32).reshape(-1)
        if atoms_flat.numel() == 0:
            return stats
        stats["generation/atom_id_min"] = float(atoms_flat.min().item())
        stats["generation/atom_id_max"] = float(atoms_flat.max().item())
        unique_atoms = int(torch.unique(atoms_flat.to(torch.long)).numel())
        stats["generation/unique_atoms"] = unique_atoms
        meta = cache.get("meta", {}) if isinstance(cache, dict) else {}
        atom_vocab = int(meta.get("num_atoms") or meta.get("atom_vocab_size") or 0)
        if atom_vocab > 0:
            stats["generation/unique_atom_frac"] = float(unique_atoms) / float(atom_vocab)
        try:
            import wandb

            stats["generation/atom_id_hist"] = wandb.Histogram(atoms_flat.numpy())
        except Exception:
            pass
    if coeffs is not None:
        coeffs_flat = torch.nan_to_num(coeffs.detach().cpu().to(torch.float32).reshape(-1))
        if coeffs_flat.numel() > 0:
            stats["generation/coeff_mean"] = float(coeffs_flat.mean().item())
            stats["generation/coeff_std"] = float(coeffs_flat.std(unbiased=False).item())
            stats["generation/coeff_abs_mean"] = float(coeffs_flat.abs().mean().item())
            stats["generation/coeff_abs_max"] = float(coeffs_flat.abs().max().item())
            try:
                import wandb

                stats["generation/coeff_hist"] = wandb.Histogram(coeffs_flat.numpy())
            except Exception:
                pass
    if coeff_bins is not None:
        bins_flat = coeff_bins.detach().cpu().to(torch.float32).reshape(-1)
        if bins_flat.numel() > 0:
            stats["generation/coeff_bin_min"] = float(bins_flat.min().item())
            stats["generation/coeff_bin_max"] = float(bins_flat.max().item())
            try:
                import wandb

                stats["generation/coeff_bin_hist"] = wandb.Histogram(bins_flat.numpy())
            except Exception:
                pass
    return stats


def _load_image_for_wandb(path: Path):
    try:
        with Image.open(path) as img:
            return np.asarray(img.convert("RGB"))
    except Exception:
        return str(path)


def _log_sparse_visuals(
    logger,
    visual_paths: dict[str, Path],
    *,
    step: int,
    caption: str,
    variant_name: str = "",
) -> None:
    suffix = f"/{variant_name}" if variant_name else ""
    for name, path in visual_paths.items():
        image = _load_image_for_wandb(path)
        log_wandb_images(
            logger,
            f"generation/{name}{suffix}",
            [image],
            step=step,
            captions=[caption],
        )
        log_wandb_images(
            logger,
            f"s2/{name}{suffix}",
            [image],
            step=step,
            captions=[caption],
        )


def _save_waveform_samples(samples: torch.Tensor, out_dir, *, stem: str, sample_rate: int, audio_config=None) -> Path:
    out_root = Path(out_dir).expanduser().resolve() / "audio_samples" / stem
    out_root.mkdir(parents=True, exist_ok=True)
    waveforms = samples.detach().cpu().to(torch.float32)[:, 0].clamp(-1.0, 1.0)
    if audio_config is not None:
        waveforms = torch.stack(
            [normalize_generated_waveform_for_preview(waveform, audio_config) for waveform in waveforms],
            dim=0,
        )
    try:
        import numpy as np
        from scipy.io import wavfile

        for idx, waveform in enumerate(waveforms):
            pcm = waveform.numpy()
            pcm = np.clip(pcm, -1.0, 1.0)
            pcm = np.round(pcm * 32767.0).astype(np.int16)
            wavfile.write(out_root / f"{stem}_{idx:02d}.wav", int(sample_rate), pcm)
    except Exception:
        torch.save(waveforms, out_root / f"{stem}.pt")
    return out_root


class Stage2SamplePreviewCallback(pl.Callback):
    """Save decoded samples during stage-2 training without bloating the train script."""

    def __init__(
        self,
        *,
        cache_pt: str,
        out_dir: str,
        step_every: int = 0,
        epoch_every: int = 0,
        n: int = 4,
        temp: float = 1.0,
        top_k: int = 0,
        ctemp=None,
        cmode: Optional[str] = None,
        sample_variants=None,
        s1_root: str = "outputs",
        use_wandb: bool = False,
        text_prompts: Optional[list[str]] = None,
        class_labels: Optional[list[int]] = None,
    ):
        super().__init__()
        self.cache_pt = str(cache_pt)
        self.out_dir = Path(out_dir).expanduser().resolve()
        self.step_every = max(0, int(step_every))
        self.epoch_every = max(0, int(epoch_every))
        self.n = max(1, int(n))
        self.temp = float(temp)
        self.top_k = int(top_k)
        self.ctemp = None if ctemp is None else float(ctemp)
        self.cmode = None if cmode is None else str(cmode).strip().lower()
        self.sample_variants = _resolve_sample_variants(
            sample_variants,
            temp=self.temp,
            top_k=self.top_k,
            ctemp=self.ctemp,
            cmode=self.cmode,
        )
        self.s1_root = str(s1_root)
        self.use_wandb = bool(use_wandb)
        self.text_prompts = list(text_prompts or [])
        self.class_labels = [int(label) for label in (class_labels or [])]
        self._last_step = -1
        self._last_epoch = -1
        self._cache = None
        self._s1 = None
        self._shape = None
        self._disabled_reason = None

    def _text_condition(self, dev: torch.device, n: int):
        cache = self._cache if isinstance(self._cache, dict) else {}
        prior_needs_text = bool(self.text_prompts) or bool((cache.get("meta", {}) or {}).get("has_text_conditioning"))
        if not prior_needs_text:
            return None, None, []
        prompts = self.text_prompts
        if not prompts:
            prompts = list(cache.get("text", []) or [])
        if not prompts:
            prompts = [""]
        tokens, mask, normalized = encode_prompts_from_cache(prompts, cache=cache, n=n, device=dev)
        return tokens, mask, normalized

    def _class_condition(self, dev: torch.device, n: int, model: Optional[pl.LightningModule] = None):
        if not self.class_labels:
            prior = getattr(model, "prior", None)
            is_class_conditional = bool(getattr(prior, "class_conditional", False))
            if not is_class_conditional and getattr(prior, "class_emb", None) is None:
                return None
            cache_meta = self._cache.get("meta", {}) if isinstance(self._cache, dict) else {}
            num_classes = int(getattr(prior, "num_classes", 0) or cache_meta.get("num_classes", 0) or 0)
            if num_classes <= 0:
                return None
            return torch.randint(0, num_classes, (int(n),), device=dev, dtype=torch.long)
        labels = torch.as_tensor(self.class_labels, device=dev, dtype=torch.long).reshape(-1)
        if int(labels.numel()) <= 0:
            return None
        if int(labels.numel()) < int(n):
            reps = int(math.ceil(float(n) / float(max(1, int(labels.numel())))))
            labels = labels.repeat(reps)
        return labels[: int(n)]

    @staticmethod
    def _barrier_if_needed(trainer: pl.Trainer):
        if getattr(trainer, "world_size", 1) <= 1:
            return
        strategy = getattr(trainer, "strategy", None)
        barrier = getattr(strategy, "barrier", None)
        if callable(barrier):
            barrier()

    def _ready(self, dev: torch.device):
        if self._cache is None:
            # Cache metadata tells us both the token shape and how to recover the
            # matching stage-1 decoder for visual/audio previews.
            raw = load_token_cache(self.cache_pt)
            self._cache = add_cache_meta(
                raw,
                token_cache_path=self.cache_pt,
                output_root=self.s1_root,
            )
            self._shape = tuple(int(v) for v in self._cache["shape"])
        if self._s1 is None:
            self._s1 = load_s1(
                self._cache,
                token_cache_path=self.cache_pt,
                device=dev,
                output_root=self.s1_root,
            )
        else:
            self._s1.model = self._s1.model.to(dev)

    def _log(
        self,
        trainer: pl.Trainer,
        *,
        step: int,
        epoch: Optional[int],
        direct: Path,
        batch=None,
        text_prompts: Optional[list[str]] = None,
        variant_name: str = "",
    ):
        if not self.use_wandb:
            return
        logger = getattr(trainer, "logger", None)
        if logger is None:
            return
        if epoch is not None:
            step = int(step) + 1
        step = int(step)

        bits = [f"step={step}"]
        if epoch is not None:
            bits.append(f"epoch={epoch}")
        if variant_name:
            bits.append(f"variant={variant_name}")
        cap = " ".join(bits)
        log_images = direct.is_file() and direct.suffix.lower() in {".png", ".jpg", ".jpeg"}
        suffix = f"/{variant_name}" if variant_name else ""
        if log_images:
            log_wandb_images(
                logger,
                f"generation/samples{suffix}",
                [_load_image_for_wandb(direct)],
                step=step,
                captions=[cap],
            )
            log_wandb_images(
                logger,
                f"s2/samples{suffix}",
                [_load_image_for_wandb(direct)],
                step=step,
                captions=[cap],
            )
        payload = {
            "generation/epoch": epoch,
            "s2/epoch": epoch,
        }
        if batch is not None and self._cache is not None:
            cache_meta = self._cache.get("meta", {}) if isinstance(self._cache, dict) else {}
            audio_meta = self._cache.get("audio_meta") if isinstance(self._cache, dict) else None
            payload.update(
                build_generated_audio_log_payload(
                    batch.imgs,
                    audio_source=cache_meta,
                    audio_meta=audio_meta,
                    split="generation",
                    max_items=min(4, int(batch.imgs.size(0))),
                    artifact_dir=self.out_dir,
                    captions=list(text_prompts or []),
                )
            )
            payload.update(_sparse_generation_stats(batch, self._cache))
        log_wandb_payload(logger, payload, step=step)

    @torch.no_grad()
    def _save(self, trainer: pl.Trainer, mod: pl.LightningModule, *, step: int, epoch: Optional[int] = None):
        if self._disabled_reason is not None:
            return
        dev = mod.device
        try:
            self._ready(dev)
        except FileNotFoundError as err:
            self._disabled_reason = str(err)
            print(f"Warning: disabling s2 sample previews; stage-1 decoder could not be loaded ({err})")
            return
        shape = _prior_token_shape(mod, self._shape)
        full_shape = tuple(int(v) for v in self._shape)
        was_training = bool(mod.training)
        mod.eval()
        stem = f"s{step:07d}" if epoch is None else f"e{epoch:03d}_s{step:07d}"
        saved_paths: list[Path] = []
        try:
            class_labels = self._class_condition(dev, self.n, model=mod)
            text_tokens, text_mask, text_prompts = self._text_condition(dev, self.n)
            for variant in self.sample_variants:
                variant_name = str(variant["name"])
                variant_stem = f"{stem}_{variant_name}" if variant_name else stem
                batch = _sample_for_preview(
                    mod,
                    self._s1,
                    prior_shape=shape,
                    full_shape=full_shape,
                    n=self.n,
                    temp=float(variant["temp"]),
                    top_k=int(variant["top_k"]),
                    ctemp=variant["ctemp"],
                    cmode=variant["cmode"],
                    class_labels=class_labels,
                    text_tokens=text_tokens,
                    text_mask=text_mask,
                    dev=dev,
                )
                log_direct = None
                if _is_waveform_samples(batch.imgs):
                    cache_meta = self._cache.get("meta", {}) if isinstance(self._cache, dict) else {}
                    audio_config = audio_config_from_source(cache_meta)
                    sample_rate = int(audio_config["sample_rate"])
                    direct = _save_waveform_samples(
                        batch.imgs,
                        self.out_dir,
                        stem=variant_stem,
                        sample_rate=sample_rate,
                        audio_config=audio_config,
                    )
                    if text_prompts:
                        _write_text_prompt_manifest(direct, stem=variant_stem, prompts=text_prompts)
                    if class_labels is not None:
                        _write_class_label_manifest(direct, class_labels, self._cache)
                else:
                    nrow = pick_nrow(int(batch.imgs.size(0)), None)
                    direct = save_grid(batch.imgs, self.out_dir, stem=variant_stem)
                    if class_labels is not None:
                        log_direct = _save_labeled_image_grid(
                            batch.imgs,
                            class_labels,
                            self._cache,
                            self.out_dir,
                            stem=variant_stem,
                            nrow=nrow,
                        )
                        _write_class_label_manifest(direct, class_labels, self._cache)
                    visual_paths = _build_sparse_visuals(batch, self._cache, self.out_dir, stem=variant_stem)
                    if self.use_wandb and trainer.is_global_zero:
                        logger = getattr(trainer, "logger", None)
                        log_step = int(step) + 1 if epoch is not None else int(step)
                        cap = (
                            f"step={int(step)}"
                            if epoch is None
                            else f"step={log_step} epoch={int(epoch)}"
                        )
                        if variant_name:
                            cap = f"{cap} variant={variant_name}"
                        _log_sparse_visuals(
                            logger,
                            visual_paths,
                            step=log_step,
                            caption=cap,
                            variant_name=variant_name,
                        )
                self._log(
                    trainer,
                    step=step,
                    epoch=epoch,
                    direct=log_direct or direct,
                    batch=batch,
                    text_prompts=_conditioning_captions(
                        self._cache,
                        class_labels,
                        text_prompts,
                        n=min(int(batch.imgs.size(0)), self.n),
                    ),
                    variant_name=variant_name,
                )
                saved_paths.append(direct)
        finally:
            if was_training:
                mod.train()
        joined = ", ".join(str(path) for path in saved_paths)
        if epoch is None:
            print(f"Saved s2 samples at step {step}: {joined}")
        else:
            print(f"Saved s2 samples at epoch {epoch}, step {step}: {joined}")

    def on_train_batch_end(self, trainer, mod, outputs, batch, batch_idx):
        if self.step_every <= 0:
            return
        step = int(trainer.global_step)
        if step <= 0 or step == self._last_step or (step % self.step_every) != 0:
            return
        if trainer.is_global_zero:
            self._save(trainer, mod, step=step)
        self._barrier_if_needed(trainer)
        self._last_step = step

    def on_train_epoch_end(self, trainer, mod):
        if self.epoch_every <= 0:
            return
        epoch = int(trainer.current_epoch) + 1
        step = int(trainer.global_step)
        if epoch <= 0 or epoch == self._last_epoch or (epoch % self.epoch_every) != 0:
            return
        if step <= 0 or step == self._last_step:
            return
        if trainer.is_global_zero:
            self._save(trainer, mod, step=step, epoch=epoch)
        self._barrier_if_needed(trainer)
        self._last_epoch = epoch
        self._last_step = step

@torch.no_grad()
def save_final_generation_preview(
    *,
    trainer: pl.Trainer,
    mod: pl.LightningModule,
    cache_pt: str,
    out_dir: str,
    n: int,
    temp: float,
    top_k: int,
    ctemp,
    cmode,
    s1_root: str,
    use_wandb: bool,
    text_prompts: Optional[list[str]] = None,
    class_labels: Optional[list[int]] = None,
    return_batch: bool = False,
):
    saver = Stage2SamplePreviewCallback(
        cache_pt=cache_pt,
        out_dir=out_dir,
        n=n,
        temp=temp,
        top_k=top_k,
        ctemp=ctemp,
        cmode=cmode,
        s1_root=s1_root,
        use_wandb=use_wandb,
        text_prompts=text_prompts,
        class_labels=class_labels,
    )
    saver._ready(mod.device)
    shape = _prior_token_shape(mod, saver._shape)
    full_shape = tuple(int(v) for v in saver._shape)
    was_training = bool(mod.training)
    mod.eval()
    try:
        labels = saver._class_condition(mod.device, saver.n, model=mod)
        text_tokens, text_mask, normalized_prompts = saver._text_condition(mod.device, saver.n)
        batch = _sample_for_preview(
            mod,
            saver._s1,
            prior_shape=shape,
            full_shape=full_shape,
            n=saver.n,
            temp=saver.temp,
            top_k=saver.top_k,
            ctemp=saver.ctemp,
            cmode=saver.cmode,
            class_labels=labels,
            text_tokens=text_tokens,
            text_mask=text_mask,
            dev=mod.device,
        )
    finally:
        if was_training:
            mod.train()
    step = max(1, int(getattr(trainer, "global_step", 0) or 0))
    stem = f"final_s{step:07d}"
    log_direct = None
    if _is_waveform_samples(batch.imgs):
        cache_meta = saver._cache.get("meta", {}) if isinstance(saver._cache, dict) else {}
        audio_config = audio_config_from_source(cache_meta)
        sample_rate = int(audio_config["sample_rate"])
        direct = _save_waveform_samples(
            batch.imgs,
            saver.out_dir,
            stem=stem,
            sample_rate=sample_rate,
            audio_config=audio_config,
        )
        if normalized_prompts:
            _write_text_prompt_manifest(Path(direct), stem=stem, prompts=normalized_prompts)
        if labels is not None:
            _write_class_label_manifest(Path(direct), labels, saver._cache)
    else:
        nrow = pick_nrow(int(batch.imgs.size(0)), None)
        direct = save_grid(batch.imgs, saver.out_dir, stem=stem)
        if labels is not None:
            log_direct = _save_labeled_image_grid(
                batch.imgs,
                labels,
                saver._cache,
                saver.out_dir,
                stem=stem,
                nrow=nrow,
            )
            _write_class_label_manifest(Path(direct), labels, saver._cache)
        visual_paths = _build_sparse_visuals(batch, saver._cache, saver.out_dir, stem=stem)
        if use_wandb:
            _log_sparse_visuals(
                getattr(trainer, "logger", None),
                visual_paths,
                step=step,
                caption=f"step={step}",
            )
    saver._log(
        trainer,
        step=step,
        epoch=None,
        direct=log_direct or direct,
        batch=batch,
        text_prompts=_conditioning_captions(
            saver._cache,
            labels,
            normalized_prompts,
            n=min(int(batch.imgs.size(0)), saver.n),
        ),
    )
    if return_batch:
        return direct, batch, saver._cache
    return direct
