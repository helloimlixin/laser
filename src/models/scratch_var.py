"""Matched random initialization for the VAR VQ and LASER experiments."""
from __future__ import annotations

import hashlib
import math

import torch
from torch.nn import functional as F

from .multiscale_laser_var import VQVAE, VAR, LaserVQVAE, LaserVAR


def state_digest(model, *, shared_only=False):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        if shared_only and name.startswith('quantize.') and not name.startswith('quantize.quant_resi.'):
            continue
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class ScratchVQVAE(VQVAE):
    """Released VQ architecture, with the common training-driver interface."""
    def __init__(self, *, channels=32, atoms=4096, ch=160, patch_nums):
        super().__init__(vocab_size=atoms, z_channels=channels, ch=ch,
                         v_patch_nums=patch_nums, test_mode=False)
        # Both treatments begin with the exact same random unit directions.
        with torch.no_grad():
            self.quantize.embedding.weight.copy_(F.normalize(self.quantize.embedding.weight, dim=1))
        self._atom_maps = []
        self._record_ids = False
        self.quantize.embedding.register_forward_pre_hook(self._record_embedding_ids)

    def _record_embedding_ids(self, module, args):
        if self._record_ids:
            self._atom_maps.append(args[0].detach())

    def forward(self, images, ret_usages=False):
        self._atom_maps = []
        self._record_ids = True
        try:
            result = super().forward(images, ret_usages=ret_usages)
        finally:
            self._record_ids = False
        self.last_atom_ids = torch.cat([x.flatten() for x in self._atom_maps])
        self._atom_maps = []
        return result

    @torch.no_grad()
    def tokenize(self, images, calibrate=False):
        self._atom_maps = []
        latent = self.quant_conv(self.encoder(images)).float()
        with torch.autocast(device_type=images.device.type, enabled=False):
            maps = self.quantize.f_to_idxBl_or_fhat(latent, to_fhat=False)
            inputs = self.quantize.idxBl_to_var_input(maps)
        atoms = torch.cat(maps, 1)[..., None]
        self._atom_maps = []
        return dict(atoms=atoms, coefficients=torch.zeros_like(atoms), inputs=inputs,
                    clip_fraction=latent.new_zeros(()))


def build_scratch_tokenizer(config, seed):
    """This path never loads a checkpoint, including a tokenizer codebook."""
    if config.get('pretrained_vae') is not None:
        raise ValueError('Scratch initialization forbids pretrained_vae; it must be null')
    kind = config.bottleneck
    if kind not in ('vq', 'laser'):
        raise ValueError('Choose vq or laser for the matched scratch experiment')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        options = dict(channels=int(config.channels), atoms=int(config.atoms),
                       ch=int(config.vae_width), patch_nums=tuple(config.patch_nums))
        if kind == 'vq':
            model = ScratchVQVAE(**options)
        else:
            model = LaserVQVAE(pretrained=None, sparsity=int(config.sparsity),
                               coefficient_bins=int(config.coefficient_bins), **options)
            model.quantize.coefficient_range_decay = float(config.coefficient_range_decay)
            policy = config.get('tokenized_sparse_policy')
            if policy is not None:
                from .sparse_token_codec import TOKEN_POLICY_VERSION
                expected = {'version', 'atom_temperature_ratio', 'coefficient_temperature_ratio'}
                if set(policy) != expected or policy['version'] != TOKEN_POLICY_VERSION:
                    raise ValueError('Unknown sparse-token training policy')
                if any(not math.isfinite(float(policy[k])) or float(policy[k]) <= 0
                       for k in expected - {'version'}):
                    raise ValueError('Sparse-token temperature ratios must be positive and finite')
                model.quantize.tokenized_sparse_policy = dict(policy)

            positions = config.get('residual_scale_positions')
            if positions is not None:
                positions = tuple(float(x) for x in positions)
                if (len(positions) != len(config.patch_nums) or positions[0] != 0. or positions[-1] != 1.
                        or any(a >= b for a, b in zip(positions, positions[1:]))):
                    raise ValueError('Residual scale positions must strictly increase from 0 to 1, one per scale')
                model.quantize.residual_scale_positions = positions
    return model


def load_finetune_tokenizer(model, checkpoint, source_patch_nums):
    """Project a LASER tokenizer checkpoint onto an explicitly retained scale subset."""
    source = tuple(int(x) for x in source_patch_nums)
    target = tuple(model.quantize.v_patch_nums)
    if (not source or source[0] != 1 or source[-1] != target[-1]
            or any(a >= b for a, b in zip(source, source[1:]))
            or any(p not in source for p in target)):
        raise ValueError('Fine-tuning requires an ordered scale subset with the same final resolution')
    indices = [source.index(p) for p in target]
    expected_positions = tuple(i / (len(source)-1) for i in indices)
    actual_positions = model.quantize.residual_scale_positions
    if actual_positions is None:
        actual_positions = tuple(i / (len(target)-1) for i in range(len(target)))
    if any(abs(a-b) > 1e-12 for a, b in zip(actual_positions, expected_positions)):
        raise ValueError('Retained scales must preserve the source residual convolution mapping')
    weights = dict(checkpoint['model'])
    ranges = weights['quantize.coefficient_max']
    if tuple(ranges.shape) != (len(source),):
        raise ValueError('Source scale count does not match checkpoint coefficient ranges')
    weights['quantize.coefficient_max'] = ranges[indices].clone()
    model.load_state_dict(weights, strict=True)
    return indices


class VQVAR(VAR):
    """Released VAR forward/sampling with a scalar NLL training interface."""
    def __init__(self, tokenizer, depth=16, width=None, heads=None, num_classes=1000):
        super().__init__(tokenizer, depth=depth, embed_dim=width or depth*64,
                         num_heads=heads or depth, num_classes=num_classes,
                         patch_nums=tuple(tokenizer.quantize.v_patch_nums),
                         attn_l2_norm=True, drop_path_rate=.1*depth/24,
                         norm_eps=1e-6, flash_if_available=False, fused_if_available=False)

    def forward(self, labels, inputs, atoms, coefficients=None):
        logits = super().forward(labels, inputs)
        nll = F.cross_entropy(logits.float().flatten(0, 1), atoms.flatten())
        return nll, torch.stack((nll.detach(), nll.detach().new_zeros(())))

    @torch.no_grad()
    def sample(self, labels, *, cfg=1.5, top_k=900, top_p=.96, seed=0):
        return super().autoregressive_infer_cfg(len(labels), labels, g_seed=seed,
                                               cfg=cfg, top_k=top_k, top_p=top_p,
                                               more_smooth=False)


def build_scratch_prior(tokenizer, kind, seed, depth=16, width=None, heads=None, num_classes=1000):
    # Explicitly isolate both constructor randomness (position parameters) and
    # weight initialization, so the shared VAR body and atom head match exactly.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        cls = VQVAR if kind == 'vq' else LaserVAR
        model = cls(tokenizer, depth=depth, width=width, heads=heads, num_classes=num_classes)
        torch.manual_seed(int(seed)+1)
        model.init_weights(init_adaln=.5, init_adaln_gamma=1e-3, init_head=.02, init_std=-1.)
        if kind == 'laser':
            model.coefficient_head.weight.data.mul_(.02)
    return model


def shared_prior_digest(model):
    digest = hashlib.sha256()
    extra = ('coefficient_head.', 'atom_context.', 'depth_context.', 'depth_embedding.')
    for name, value in model.state_dict().items():
        if name.startswith(extra):
            continue
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
