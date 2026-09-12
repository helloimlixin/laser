"""Short, causal coefficient rollouts for history-recovery training."""

import math
import torch
import torch.nn.functional as F


def categorical_draw(logits, uniform):
    cdf = logits.float().softmax(-1).cumsum(-1).contiguous()
    return torch.searchsorted(cdf, uniform.unsqueeze(-1).contiguous()).squeeze(-1).clamp_max(logits.shape[-1] - 1)


def sign_nll(logits, target):
    """Exact marginal sign NLL of a symmetric, ordered coefficient vocabulary."""
    half = logits.shape[-1] // 2
    if logits.shape[-1] % 2:
        raise ValueError("Expected an even symmetric coefficient vocabulary")
    grouped = torch.stack((logits[..., :half].float().logsumexp(-1),
                           logits[..., half:].float().logsumexp(-1)), -1)
    return F.cross_entropy(grouped.flatten(0, -2), (target >= half).long().flatten(), reduction="none").reshape_as(target)


def recovery_mask(shape, start_site, span_sites, recovery_sites=1, device=None):
    h, w, depth = shape
    first = start_site * depth
    end = min((start_site + span_sites + recovery_sites) * depth, h * w * depth)
    mask = torch.zeros(h * w * depth, dtype=torch.bool, device=device)
    # The first coefficient has no generated coefficient history yet.
    mask[first + 1:end] = True
    return mask.reshape(h, w, depth)


def recovery_strength(epoch_progress, start_epoch=5., ramp_epochs=10.):
    if epoch_progress <= start_epoch:
        return 0.
    if ramp_epochs <= 0:
        return 1.
    return min(1., (epoch_progress - start_epoch) / ramp_epochs)


def scheduled_lr(progress, total_epochs, peak=2e-4, minimum=1e-5, warmup=5.):
    if warmup > 0 and progress < warmup:
        return peak * max(.005, progress / warmup)
    fraction = min(1., max(0., (progress - warmup) / max(total_epochs - warmup, 1e-9)))
    return minimum + (peak - minimum) * .5 * (1 + math.cos(math.pi * fraction))


@torch.no_grad()
def sample_coefficient_span(model, aux, truth, start_site, span_sites, uniforms):
    """Generate consecutive sites after a true prefix, with true current atoms.

    Starting the cache at depth zero pre-fills the spatial prefix in parallel.
    No current/future ground-truth coefficient within the span is consumed.
    Samples are detached; a later training forward supplies recovery gradients.
    """
    h, w, depth = model.block_size
    if not 0 <= start_site < h * w or not 1 <= span_sites <= h * w - start_site:
        raise ValueError("Invalid coefficient rollout span")
    if uniforms.shape != (len(truth), span_sites * depth):
        raise ValueError("One independent uniform is required per generated coefficient")
    packed = truth.clone()
    atoms, _ = model.unpack(truth)
    was_training = model.training
    model.eval()
    model.init_cache()
    try:
        for relative in range(span_sites * depth):
            site, d = divmod(start_site * depth + relative, depth)
            row, col = divmod(site, w)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=truth.device.type == "cuda"):
                hidden = model.cached_head_output(packed, aux, None, (row, col, d), amp=False)
                refined = model.refine_coefficient_hidden(hidden, aux.dictionary.t()[atoms[:, row, col, d]])
                logits = model.classify_coefficients(refined, d).float()
                sampled = categorical_draw(logits, uniforms[:, relative])
            packed[:, row, col, d] = atoms[:, row, col, d] * model.coeff_vocab_size + sampled
    finally:
        model.init_cache()
        model.train(was_training)
    return packed


class EpochStream:
    """Exact finite-epoch shuffled batches, including the final partial batch."""
    def __init__(self, size, seed):
        self.size = size
        self.generator = torch.Generator().manual_seed(seed)
        self.epoch = 0
        self.position = 0
        self.permutation = torch.randperm(size, generator=self.generator)

    def next(self, batch_size):
        if self.position == self.size:
            self.epoch += 1
            self.position = 0
            self.permutation = torch.randperm(self.size, generator=self.generator)
        end = min(self.position + batch_size, self.size)
        indices = self.permutation[self.position:end]
        self.position = end
        return indices, self.epoch + end / self.size, end == self.size

    def state_dict(self):
        return {"size": self.size, "epoch": self.epoch, "position": self.position,
                "permutation": self.permutation, "generator": self.generator.get_state()}

    def load_state_dict(self, state):
        if state["size"] != self.size:
            raise ValueError("Training population changed on resume")
        self.epoch, self.position = state["epoch"], state["position"]
        self.permutation = state["permutation"].cpu()
        self.generator.set_state(state["generator"].cpu())
