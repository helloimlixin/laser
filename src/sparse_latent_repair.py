"""A separate denoiser for physical sparse latents; base models stay frozen."""

import torch
from torch import nn
from src.coefficient_history_training import categorical_draw


class SparseLatentRepair(nn.Module):
    def __init__(self, channels=256, width=256, layers=6, heads=8, height=8, grid_width=8):
        super().__init__()
        self.config = dict(channels=channels, width=width, layers=layers, heads=heads,
                           height=height, grid_width=grid_width)
        self.register_buffer("mean", torch.zeros(1, channels, 1, 1))
        self.register_buffer("scale", torch.ones(1, channels, 1, 1))
        self.input = nn.Linear(channels, width)
        self.position = nn.Parameter(torch.randn(1, height * grid_width, width) * .02)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(width, heads, dim_feedforward=width * 4,
                dropout=0., activation="gelu", batch_first=True, norm_first=True)
            for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, channels)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @torch.no_grad()
    def set_normalization(self, latents):
        self.mean.copy_(latents.mean((0, 2, 3), keepdim=True))
        self.scale.copy_(latents.std((0, 2, 3), keepdim=True).clamp_min(.05))

    def forward(self, z, strength=1.):
        if z.shape[1:] != (self.config["channels"], self.config["height"], self.config["grid_width"]):
            raise ValueError("Unexpected latent grid")
        normalized = (z - self.mean) / self.scale
        x = self.input(normalized.flatten(2).transpose(1, 2)) + self.position
        for block in self.blocks:
            x = block(x)
        residual = self.output(self.norm(x)).transpose(1, 2).reshape_as(z)
        return z + float(strength) * self.scale * residual


@torch.no_grad()
def sample_anchored_span(model, aux, truth, start_site, span_sites, uniforms, *, sample_atoms=True, top_k=250):
    """Replace one spatial span with causal AR predictions; anchor its exterior.

    Targets within the span are never consumed, except the explicitly forced
    atoms when sample_atoms=False. Ground-truth future sites are only for pairing
    a corrupted representation with a known clean image, not AR conditioning.
    """
    h, w, depth = model.block_size
    if not 0 <= start_site < h * w or not 1 <= span_sites <= h * w - start_site:
        raise ValueError("Invalid rollout span")
    if uniforms.shape != (len(truth), span_sites * depth, 2):
        raise ValueError("Need independent atom and coefficient uniforms")
    packed = truth.clone()
    was_training = model.training
    model.eval()
    model.init_cache()
    try:
        for relative in range(span_sites * depth):
            site, d = divmod(start_site * depth + relative, depth)
            row, col = divmod(site, w)
            with torch.autocast(truth.device.type, dtype=torch.bfloat16, enabled=truth.device.type == "cuda"):
                hidden = model.cached_head_output(packed, aux, None, (row, col, d), amp=False)
                if sample_atoms:
                    logits = model.classifier(hidden).float()
                    if d:
                        logits.scatter_(1, packed[:, row, col, :d] // model.coeff_vocab_size, -torch.inf)
                    values, indices = logits.topk(min(top_k, logits.shape[-1]), dim=-1)
                    chosen = categorical_draw(values, uniforms[:, relative, 0])
                    atom = indices.gather(-1, chosen[:, None]).squeeze(-1)
                else:
                    atom = truth[:, row, col, d] // model.coeff_vocab_size
                refined = model.refine_coefficient_hidden(hidden, aux.dictionary.t()[atom])
                coeff = categorical_draw(model.classify_coefficients(refined, d), uniforms[:, relative, 1])
            packed[:, row, col, d] = atom * model.coeff_vocab_size + coeff
    finally:
        model.init_cache()
        model.train(was_training)
    return packed


@torch.no_grad()
def synthetic_corruption(packed, neighbors, generator, bins=2048):
    """Mix local atom swaps, occasional unrelated atoms, and coefficient errors."""
    atoms, ids = packed // bins, packed % bins
    device, shape = packed.device, packed.shape
    rand = lambda size: torch.rand(size, device=device, generator=generator)
    severity = .02 + .13 * rand((len(packed), 1, 1, 1))
    atom_mask = rand(shape) < severity * .5
    choices = torch.randint(neighbors.shape[1], shape, device=device, generator=generator)
    proposed = neighbors[atoms, choices]
    random_atoms = torch.randint(len(neighbors), shape, device=device, generator=generator)
    proposed = torch.where(rand(shape) < .25, random_atoms, proposed)
    changed = atoms.clone()
    for d in range(shape[-1]):
        candidate = proposed[..., d]
        # Exclude other original supports and previously changed supports.
        invalid = (candidate[..., None] == atoms).any(-1)
        if d:
            invalid |= (candidate[..., None] == changed[..., :d]).any(-1)
        changed[..., d] = torch.where(atom_mask[..., d] & ~invalid, candidate, atoms[..., d])
    noise = torch.randn(shape, device=device, generator=generator) * (16 + 112 * rand((len(packed), 1, 1, 1)))
    ids = torch.where(rand(shape) < severity * 2, (ids + noise.round().long()).clamp(0, bins - 1), ids)
    ids = torch.where(rand(shape) < severity * .5, bins - 1 - ids, ids)
    return changed * bins + ids
