"""Compare atom filters on one frozen checkpoint and full ImageNet val reference."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import time


POLICIES = [
    dict(name="current_k250", atom_temperature=1., atom_top_k=250, atom_top_p=1.),
    dict(name="cooled_k250", atom_temperature=.9, atom_top_k=250, atom_top_p=1.),
    dict(name="nucleus092", atom_temperature=1., atom_top_k=0, atom_top_p=.92),
    dict(name="nucleus085", atom_temperature=1., atom_top_k=0, atom_top_p=.85),
    dict(name="nucleus090_t090", atom_temperature=.9, atom_top_k=0, atom_top_p=.9),
    dict(name="nucleus092_t095", atom_temperature=.95, atom_top_k=0, atom_top_p=.92),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--policies", default=",".join(p["name"] for p in POLICIES))
    parser.add_argument("--seed", type=int, default=261002)
    args = parser.parse_args()
    sys.path.insert(0, str(args.runtime.resolve()))
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader, DistributedSampler
    from torchvision import datasets
    from omegaconf import OmegaConf
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore
    from src.training.rqtransformer import build_model, LaserAux, val_image_transform, save_unlabeled_grid

    torch.set_num_threads(8)
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    assert world == 5 and args.samples % 1000 == 0
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    dist.init_process_group("nccl", timeout=timedelta(minutes=45))
    args.output.mkdir(parents=True, exist_ok=True)
    options = OmegaConf.load(args.recipe).options
    payload = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    with torch.device("meta"):
        model = build_model(18432, 16384, sparsity_level=4, physical_pair_context=True)
    model.load_state_dict(payload["state_dict"], strict=True, assign=True)
    model = model.to(device).eval().requires_grad_(False)
    step = int(payload["global_step"])
    del payload
    aux = LaserAux(Path(options.checkpoint), 16384, 2048, 3., coeff_scale=6.4,
                   coeff_scales=list(options.coeff_scales), sparsity_level=4,
                   soft_target_physical=False, clamp_coeffs=False).to(device).eval().requires_grad_(False)
    metric = FrechetInceptionDistance(feature=2048, normalize=True,
        reset_real_features=False, sync_on_compute=True).to(device)
    inception = InceptionScore(normalize=True, splits=10, sync_on_compute=True).to(device) if args.samples == 50000 else None
    dataset = datasets.ImageFolder(Path(options.data) / "val", transform=val_image_transform())
    assert len(dataset) == 50000 and len(dataset.classes) == 1000
    loader = DataLoader(dataset, batch_size=128,
        sampler=DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=False),
        num_workers=8, pin_memory=True)
    real_count = 0
    with torch.inference_mode():
        for images, _ in loader:
            pixels = ((images.to(device, non_blocking=True).float() + 1) * .5).clamp(0, 1)
            metric.update(pixels, real=True)
            real_count += len(images)
        count = torch.tensor(real_count, device=device, dtype=torch.long)
        dist.all_reduce(count)
        assert int(count) == 50000
        if rank == 0:
            print(json.dumps(dict(phase="real_reference_ready", real_samples=int(count),
                                  checkpoint_step=step, generated_samples=args.samples)), flush=True)
        selected = [next(p for p in POLICIES if p["name"] == name) for name in args.policies.split(",")]
        local_samples = args.samples // world
        results = []
        for policy in selected:
            metric.reset()
            if inception is not None:
                inception.reset()
            torch.manual_seed(args.seed + rank)
            torch.cuda.manual_seed_all(args.seed + rank)
            start = time.monotonic()
            generated = 0
            while generated < local_samples:
                size = min(512, local_samples - generated)
                labels = (torch.arange(generated, generated + size, device=device) * world + rank) % 1000
                settings = {k: v for k, v in policy.items() if k != "name"}
                tokens = model.sample_sparse(size, aux, cond=labels, **settings,
                    coeff_temperature=float(options.coeff_temperature), coeff_top_k=0,
                    coeff_top_p=float(options.coeff_top_p), amp=True)
                support = tokens[..., 0::2].sort(-1).values
                assert (support.diff(dim=-1) > 0).all()
                for offset in range(0, size, 64):
                    decoded = aux.decode_tokens(tokens[offset:offset + 64])
                    assert torch.isfinite(decoded).all()
                    pixels = ((decoded.float() + 1) * .5).clamp(0, 1)
                    metric.update(pixels, real=False)
                    if inception is not None:
                        inception.update(pixels)
                    if generated == 0 and offset == 0 and rank == 0:
                        save_unlabeled_grid(pixels.cpu(), args.output / (policy["name"] + ".png"), nrow=8)
                    del decoded, pixels
                generated += size
                if rank == 0:
                    print(json.dumps(dict(phase="sampler_progress", policy=policy["name"],
                        generated_on_rank=generated, samples_on_rank=local_samples,
                        elapsed_seconds=time.monotonic()-start)), flush=True)
                del tokens, support
            result = dict(**policy, checkpoint_step=step, seed=args.seed,
                          generated_samples=args.samples, real_samples=50000,
                          coeff_temperature=float(options.coeff_temperature), coeff_top_p=float(options.coeff_top_p),
                          fid=float(metric.compute()), elapsed_seconds=time.monotonic()-start,
                          backend="torchmetrics", world_size=world,
                          interpretation="Paired sampler comparison; training weights identical. Pilot FID is sample-count biased.")
            if inception is not None:
                mean, std = inception.compute()
                result.update(inception_score=float(mean), inception_score_std=float(std))
            results.append(result)
            if rank == 0:
                (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
                print(json.dumps(dict(phase="sampler_result", **result)), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
