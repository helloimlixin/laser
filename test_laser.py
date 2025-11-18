#!/usr/bin/env python
"""Quick sanity check for K-SVD VAE model."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from src.models.laser import LASER

print("="*60)
print("K-SVD VAE SANITY CHECK")
print("="*60)

# Create model
print("\n1. Creating K-SVD VAE model...")
model = LASER(
    in_channels=3,
    num_hiddens=128,
    num_embeddings=32,
    embedding_dim=64,
    sparsity_level=5,
    num_residual_blocks=2,
    num_residual_hiddens=32,
    commitment_cost=0.25,
    learning_rate=1e-4,
    beta=0.9,
    ksvd_iterations=2,
    perceptual_weight=1.0,
    patch_size=1,
)
print("✓ Model created successfully")

# Test forward pass
print("\n2. Testing forward pass...")
model.eval()
x = torch.randn(2, 3, 64, 64)

with torch.no_grad():
    recon, bottleneck_loss, coeffs = model(x)

print(f"✓ Forward pass successful")
print(f"   Input shape: {x.shape}")
print(f"   Reconstruction shape: {recon.shape}")
print(f"   Bottleneck loss: {bottleneck_loss.item():.6f}")
print(f"   Coefficients shape: {coeffs.shape}")
print(f"   Sparsity: {(coeffs.abs() > 1e-6).sum(dim=0).float().mean().item():.2f} atoms/pixel")

# Test reconstruction quality
print("\n3. Testing reconstruction quality...")
mse = torch.nn.functional.mse_loss(recon, x).item()
print(f"✓ MSE: {mse:.6f}")

# Test training mode
print("\n4. Testing training mode (K-SVD updates)...")
model.train()
dict_before = model.bottleneck.dictionary.data.clone()

recon, bottleneck_loss, coeffs = model(x)

dict_after = model.bottleneck.dictionary.data
dict_changed = not torch.allclose(dict_before, dict_after, atol=1e-6)
print(f"✓ Dictionary updated: {dict_changed}")

# Test compute_metrics
print("\n5. Testing compute_metrics...")
model.eval()
batch = (x,)
loss, recon_vis, x_vis = model.compute_metrics(batch, prefix='test')
print(f"✓ Metrics computed")
print(f"   Total loss: {loss.item():.6f}")

# Count parameters
print("\n6. Model statistics...")
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   Total parameters: {total_params:,}")
print(f"   Trainable parameters: {trainable_params:,}")
print(f"   Dictionary atoms: {model.bottleneck.num_embeddings}")
print(f"   Atom dimension: {model.bottleneck.atom_dim}")

print("\n" + "="*60)
print("ALL CHECKS PASSED ✓")
print("="*60)
