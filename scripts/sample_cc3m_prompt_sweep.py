#!/usr/bin/env python3
"""Sample a Figure-17-style text grid sweep from a CC3M checkpoint."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys

import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "rq-vae-transformer"))

from scripts.train_official_rqtransformer_laser_stage2 import (  # noqa: E402
    LaserAux,
    build_model,
    encode_cc3m_prompts,
    release_cuda_memory,
    save_paper_style_text_grid,
)


PROMPTS = (
    "a cheeseburger in front of a mountain range covered with snow",
    "a cherry blossom tree on the blue ocean",
    "a river has burst its banks and has spread out onto arable farmland alongside",
    "a photograph of crowd of people under cherry blossom trees",
    "a small house in the wilderness",
    "a small house on the shore",
    "sunset over the skyline of a city",
    "night landscape of the skyline of a city",
    "an illustration of a cathedral",
    "the fountain in the square",
)

# These are the repository's established --sample-grid-sweep presets. Keeping
# them here makes the standalone sweep independent of training CLI defaults.
SAMPLING_SETTINGS = (
    {
        "name": "at1_k250_p1__ct1_k250_p1",
        "atom_temperature": 1.0,
        "atom_top_k": 250,
        "atom_top_p": 1.0,
        "coeff_temperature": 1.0,
        "coeff_top_k": 250,
        "coeff_top_p": 1.0,
    },
    {
        "name": "at0.85_k250_p1__ct0.85_k250_p1",
        "atom_temperature": 0.85,
        "atom_top_k": 250,
        "atom_top_p": 1.0,
        "coeff_temperature": 0.85,
        "coeff_top_k": 250,
        "coeff_top_p": 1.0,
    },
    {
        "name": "at0.9_k0_p0.92__ct1_k0_p0.85",
        "atom_temperature": 0.9,
        "atom_top_k": 0,
        "atom_top_p": 0.92,
        "coeff_temperature": 1.0,
        "coeff_top_k": 0,
        "coeff_top_p": 0.85,
    },
    {
        "name": "at0.95_k250_p0.95__ct0.9_k250_p0.95",
        "atom_temperature": 0.95,
        "atom_top_k": 250,
        "atom_top_p": 0.95,
        "coeff_temperature": 0.9,
        "coeff_top_k": 250,
        "coeff_top_p": 0.95,
    },
)


def atomic_json(payload: object, target: Path) -> None:
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, target)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def image_path(output: Path, setting_name: str, prompt_index: int, sample_index: int) -> Path:
    return (
        output
        / "images"
        / setting_name
        / f"prompt_{prompt_index + 1:02d}"
        / f"sample_{sample_index + 1:02d}.png"
    )


def save_sample(image: torch.Tensor, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.stem + ".tmp.png")
    transforms.functional.to_pil_image(image.clamp(0, 1)).save(temporary, optimize=True)
    os.replace(temporary, target)


def load_completed_setting(output: Path, setting_name: str, samples_per_prompt: int):
    grid_path = output / "grids" / f"{setting_name}.jpg"
    paths = [
        image_path(output, setting_name, prompt_index, sample_index)
        for prompt_index in range(len(PROMPTS))
        for sample_index in range(samples_per_prompt)
    ]
    if not grid_path.is_file() or not all(path.is_file() for path in paths):
        return None
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(transforms.functional.to_tensor(image.convert("RGB")))
    return grid_path, torch.stack(images)


@torch.no_grad()
def generate_setting(
    model,
    aux: LaserAux,
    encoded_prompts: torch.Tensor,
    setting: dict,
    output: Path,
    *,
    samples_per_prompt: int,
    base_seed: int,
    setting_index: int,
):
    completed = load_completed_setting(output, setting["name"], samples_per_prompt)
    if completed is not None:
        print(f"reuse completed setting={setting['name']}", flush=True)
        return completed

    device = next(model.parameters()).device
    generated_rows = []
    for prompt_index, prompt in enumerate(PROMPTS):
        prompt_seed = int(base_seed) + setting_index * 100_000 + prompt_index * 1_000
        set_seed(prompt_seed)
        conditions = encoded_prompts[prompt_index : prompt_index + 1].to(device)
        conditions = conditions.repeat(samples_per_prompt, 1)
        print(
            f"setting={setting['name']} prompt={prompt_index + 1:02d}/{len(PROMPTS)} "
            f"seed={prompt_seed} text={prompt}",
            flush=True,
        )
        atoms, coeff_ids = model.sample_compound(
            samples_per_prompt,
            aux,
            cond=conditions,
            atom_temperature=float(setting["atom_temperature"]),
            atom_top_k=int(setting["atom_top_k"]) or aux.num_atoms,
            atom_top_p=float(setting["atom_top_p"]),
            coeff_temperature=float(setting["coeff_temperature"]),
            coeff_top_k=int(setting["coeff_top_k"]) or aux.coeff_vocab_size,
            coeff_top_p=float(setting["coeff_top_p"]),
            amp=True,
        )
        images = ((aux.decode_compound(atoms, coeff_ids).float().cpu() + 1.0) * 0.5).clamp(0, 1)
        if tuple(images.shape) != (samples_per_prompt, 3, 256, 256):
            raise RuntimeError(f"Unexpected decoded image shape: {tuple(images.shape)}")
        for sample_index, image in enumerate(images):
            save_sample(
                image,
                image_path(output, setting["name"], prompt_index, sample_index),
            )
        generated_rows.append(images)
        del atoms, coeff_ids, conditions, images
        release_cuda_memory(model)

    all_images = torch.cat(generated_rows, dim=0)
    grid_path = output / "grids" / f"{setting['name']}.jpg"
    save_paper_style_text_grid(all_images, PROMPTS, grid_path)
    return grid_path, all_images


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--stage2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-run", default="helloimlixin-rutgers/laser/cc3mcmp0808124922")
    parser.add_argument("--samples-per-prompt", type=int, default=8)
    parser.add_argument("--seed", type=int, default=220301941)
    parser.add_argument("--expected-epoch", type=int, default=44)
    parser.add_argument("--expected-step", type=int, default=63000)
    parser.add_argument("--num-atoms", type=int, default=16384)
    parser.add_argument("--coeff-vocab-size", type=int, default=2048)
    parser.add_argument("--coeff-max", type=float, default=20.0)
    parser.add_argument("--coeff-scale", type=float, default=6.4)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-entity", default="helloimlixin-rutgers")
    parser.add_argument("--wandb-project", default="laser")
    parser.add_argument("--wandb-id", default="cc3mcmp0808124922-sweep63000")
    parser.add_argument("--wandb-name", default="cc3mcmp0808124922 prompt sweep step 63000")
    args = parser.parse_args()

    if args.samples_per_prompt != 8:
        raise ValueError("This Figure-17 sweep requires exactly eight images per prompt")
    if not args.stage1.is_file():
        raise FileNotFoundError(args.stage1)
    if not args.stage2.is_file():
        raise FileNotFoundError(args.stage2)

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "grids").mkdir(exist_ok=True)

    print(f"hashing pinned checkpoint={args.stage2}", flush=True)
    checkpoint_sha256 = sha256_file(args.stage2)
    payload = torch.load(args.stage2, map_location="cpu", weights_only=False, mmap=True)
    checkpoint_epoch = int(payload.get("epoch", -1))
    checkpoint_step = int(payload.get("global_step", -1))
    if checkpoint_epoch != args.expected_epoch or checkpoint_step != args.expected_step:
        raise RuntimeError(
            f"Expected epoch/step {args.expected_epoch}/{args.expected_step}, "
            f"found {checkpoint_epoch}/{checkpoint_step}"
        )

    aux = LaserAux(
        args.stage1,
        args.num_atoms,
        args.coeff_vocab_size,
        args.coeff_max,
        args.coeff_scale,
        attn_resolutions=(8,),
        sparsity_level=2,
    ).to(device).eval()
    model = build_model(
        args.num_atoms + args.coeff_vocab_size,
        args.num_atoms,
        compound=True,
        coeff_vocab_size=args.coeff_vocab_size,
        compound_micro_transformer_layers=2,
        compound_depth_specific_coeff_heads=True,
        sparsity_level=2,
        model_preset="cc3m-650m",
        condition_length=32,
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    del payload
    gc.collect()
    model = model.to(device).eval()
    for module in (aux, model):
        module.requires_grad_(False)

    encoded_prompts = encode_cc3m_prompts(PROMPTS)
    completed_settings = []
    for setting_index, setting in enumerate(SAMPLING_SETTINGS):
        grid_path, images = generate_setting(
            model,
            aux,
            encoded_prompts,
            setting,
            args.output,
            samples_per_prompt=args.samples_per_prompt,
            base_seed=args.seed,
            setting_index=setting_index,
        )
        record = {
            **setting,
            "grid": str(grid_path),
            "grid_sha256": sha256_file(grid_path),
            "images": len(PROMPTS) * args.samples_per_prompt,
        }
        completed_settings.append(record)
        atomic_json(
            {
                "status": "running",
                "completed_settings": completed_settings,
            },
            args.output / "progress.json",
        )
        del images
        release_cuda_memory(model)

    manifest = {
        "status": "complete",
        "source_run": args.source_run,
        "checkpoint": str(args.stage2.resolve()),
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_step": checkpoint_step,
        "checkpoint_sha256": checkpoint_sha256,
        "stage1": str(args.stage1.resolve()),
        "prompts": list(PROMPTS),
        "samples_per_prompt": args.samples_per_prompt,
        "settings": completed_settings,
        "seed_scheme": "base + setting_index*100000 + prompt_index*1000",
        "base_seed": args.seed,
        "total_images": len(PROMPTS) * args.samples_per_prompt * len(SAMPLING_SETTINGS),
        "grid_layout": {
            "rows": len(PROMPTS),
            "image_columns": args.samples_per_prompt,
            "caption_column": "left",
        },
    }
    manifest_path = args.output / "manifest.json"
    atomic_json(manifest, manifest_path)
    atomic_json(manifest, args.output / "progress.json")

    if args.wandb:
        import wandb

        source_run_id = args.source_run.rsplit("/", 1)[-1]
        wb = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            id=args.wandb_id,
            resume="allow",
            name=args.wandb_name,
            group=f"{source_run_id}-sampling-sweeps",
            job_type="text_prompt_sampling_sweep",
            notes=f"Sampling sweep derived from {args.source_run} at step {checkpoint_step}.",
            config={
                "source_run": args.source_run,
                "checkpoint_epoch": checkpoint_epoch,
                "checkpoint_step": checkpoint_step,
                "checkpoint_sha256": checkpoint_sha256,
                "prompts": list(PROMPTS),
                "samples_per_prompt": args.samples_per_prompt,
                "sampling_settings": list(SAMPLING_SETTINGS),
                "base_seed": args.seed,
            },
        )
        wb.log({
            **{
                f"sampling_sweep/{setting['name']}": wandb.Image(
                    setting["grid"],
                    caption=f"10 prompts × 8 samples; {setting['name']}",
                )
                for setting in completed_settings
            },
            "sampling/checkpoint_epoch": checkpoint_epoch,
            "sampling/checkpoint_step": checkpoint_step,
            "sampling/prompts": len(PROMPTS),
            "sampling/images_per_prompt": args.samples_per_prompt,
            "sampling/settings": len(completed_settings),
            "sampling/total_images": manifest["total_images"],
        })
        wb.save(str(manifest_path), base_path=str(args.output), policy="now")
        wb.finish()

    print(
        f"completed settings={len(completed_settings)} total_images={manifest['total_images']} "
        f"manifest={manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
