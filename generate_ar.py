"""
Generation script for Autoregressive LASER Image Generation.

This script:
1. Loads a pretrained AR transformer
2. Generates pattern sequences autoregressively
3. Decodes patterns to sparse codes using LASER pattern quantizer
4. Reconstructs images using LASER decoder

Usage:
    python generate_ar.py \
        ar_ckpt=path/to/ar.ckpt \
        laser_ckpt=path/to/laser.ckpt \
        --num_samples=64 \
        --temperature=1.0

Output:
    Saves generated images to output directory
"""

import os
import argparse
import torch
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

from src.models.ar_transformer import ARTransformer
from src.models.laser import LASER


@torch.no_grad()
def generate_images(
    ar_model: ARTransformer,
    laser_model: LASER,
    num_samples: int = 64,
    batch_size: int = 16,
    temperature: float = 1.0,
    top_k: int = None,
    top_p: float = None,
    device: str = 'cuda',
) -> torch.Tensor:
    """
    Generate images using AR transformer and LASER decoder.

    Args:
        ar_model: Trained AR transformer
        laser_model: Trained LASER model with pattern quantization
        num_samples: Total number of images to generate
        batch_size: Batch size for generation
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter
        device: Device for generation

    Returns:
        Generated images tensor [N, C, H, W]
    """
    ar_model.eval()
    laser_model.eval()
    ar_model.to(device)
    laser_model.to(device)

    all_images = []
    num_batches = (num_samples + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc="Generating images"):
        current_batch_size = min(batch_size, num_samples - i * batch_size)

        # Step 1: Generate pattern indices from AR model
        pattern_indices = ar_model.generate(
            batch_size=current_batch_size,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            device=device,
        )  # [B, seq_len]

        # Step 2-4: Decode patterns through LASER bottleneck & decoder
        images = decode_sparse_codes(laser_model, pattern_indices, device)

        all_images.append(images.cpu())

    return torch.cat(all_images, dim=0)[:num_samples]


def decode_sparse_codes(laser_model: LASER, pattern_indices: torch.Tensor, device: str) -> torch.Tensor:
    """
    Decode pattern indices to images using LASER bottleneck helpers.

    Args:
        laser_model: LASER model
        pattern_indices: [B, num_patches] pattern indices
        device: Device

    Returns:
        Reconstructed images [B, C, H, W]
    """
    bottleneck = laser_model.bottleneck
    z_dl = bottleneck.decode_pattern_indices_to_latent(pattern_indices.to(device))
    z_post = laser_model.post_bottleneck(z_dl)
    images = laser_model.decoder(z_post)
    return images


def main():
    parser = argparse.ArgumentParser(description="Generate images from AR transformer + LASER")
    parser.add_argument("--ar_ckpt", type=str, required=True, help="Path to AR transformer checkpoint")
    parser.add_argument("--laser_ckpt", type=str, required=True, help="Path to LASER checkpoint")
    parser.add_argument("--num_samples", type=int, default=64, help="Number of images to generate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for generation")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=None, help="Nucleus sampling threshold")
    parser.add_argument("--output_dir", type=str, default="outputs/generated", help="Output directory")
    parser.add_argument("--grid_size", type=int, default=8, help="Grid size for visualization")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load models
    print(f"Loading AR transformer from: {args.ar_ckpt}")
    ar_model = ARTransformer.load_from_checkpoint(args.ar_ckpt, map_location='cpu')

    print(f"Loading LASER model from: {args.laser_ckpt}")
    laser_model = LASER.load_from_checkpoint(args.laser_ckpt, map_location='cpu')

    if not laser_model.use_pattern_quantizer:
        raise ValueError("LASER model doesn't have pattern quantization enabled")

    # Generate images
    print(f"\nGenerating {args.num_samples} images...")
    images = generate_images(
        ar_model=ar_model,
        laser_model=laser_model,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=args.device,
    )

    # Save individual images
    print(f"\nSaving images to {args.output_dir}")
    for i, img in enumerate(images):
        save_image(img, os.path.join(args.output_dir, f"sample_{i:04d}.png"), normalize=True)

    # Save grid visualization
    grid = make_grid(images[:args.grid_size**2], nrow=args.grid_size, normalize=True)
    save_image(grid, os.path.join(args.output_dir, "grid.png"))

    print(f"\nGeneration complete!")
    print(f"  Individual images: {args.output_dir}/sample_*.png")
    print(f"  Grid visualization: {args.output_dir}/grid.png")


if __name__ == "__main__":
    main()
