"""Native compound RQ with categorical supports and continuous coefficients.

The spatial/depth stacks and selected-atom micro-transformer match the original
compound model. Completed pairs use the original physical OMP coefficients.
"""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from src.continuous_sparse_codec import ContinuousSparseCodes
from src.models.continuous_coefficient_mixture import ContinuousCoefficientMixture
from src.models.laser_compound_rq_adapter import LaserCompoundRQTransformer
from src.rq_upstream import RQTransformer, RQTransformerConfig
from rqvae.models.rqtransformer.attentions import AttentionStack
from rqvae.utils.utils import top_k_logits, top_p_probs


class ContinuousCompoundRQTransformer(RQTransformer):
    def __init__(self, config, *, num_atoms, coefficient_scales, components=8, min_std=.03):
        super().__init__(config)
        self.num_atoms = int(num_atoms)
        self.pair_depth = int(config.block_size[-1])
        if (not config.input_emb_vqvae or not config.head_emb_vqvae
                or not config.shared_cls_emb or not config.cumsum_depth_ctx
                or config.block_size_cond != 1 or config.vocab_size != num_atoms):
            raise ValueError('Require native compound RQ physical embeddings and a shared atom classifier')
        scales = torch.as_tensor(coefficient_scales, dtype=torch.float32, device='cpu')
        if (scales.shape != (self.pair_depth,) or not scales.isfinite().all()
                or not (scales > 0).all()):
            raise ValueError('One positive finite physical coefficient scale per depth required')
        self.register_buffer('coefficient_scales', scales)
        dimension, width = int(config.input_embed_dim), int(config.embed_dim)
        self.coefficient_embedding = nn.Sequential(
            nn.Linear(1, dimension), nn.SiLU(), nn.Linear(dimension, dimension))
        self.pair_embedding_adapter = nn.Sequential(
            nn.LayerNorm(3 * dimension), nn.Linear(3 * dimension, dimension),
            nn.SiLU(), nn.Linear(dimension, dimension))
        self.coeff_atom_proj = nn.Linear(dimension, width, bias=False)
        micro = copy.deepcopy(config.head)
        micro.n_layer = 2
        self.coeff_micro_transformer = AttentionStack(micro)
        self.coeff_micro_pos = nn.Parameter(torch.empty(1, 2, width))
        nn.init.normal_(self.coeff_micro_pos, std=.02)
        self.coefficient_heads = nn.ModuleList([
            ContinuousCoefficientMixture(width, components, min_std) for _ in range(self.pair_depth)])
        self._native_hidden = self._codes = None
        self.classifier.register_forward_pre_hook(self._capture_hidden)

    coefficient_hidden = LaserCompoundRQTransformer.coefficient_hidden

    def _capture_hidden(self, module, inputs):
        self._native_hidden = inputs[0]

    def embed_with_model_aux(self, atoms, model_aux):
        c = self._codes.coefficients.reshape(len(atoms), -1, self.pair_depth)[:, :atoms.shape[1]]
        with torch.autocast(device_type=atoms.device.type, enabled=False):
            vectors = model_aux.dictionary.float().T[atoms]
            physical = vectors * c[..., None]
            scalar = torch.asinh(c / self.coefficient_scales)[..., None]
        identity = self.coefficient_embedding(scalar)
        return physical + self.pair_embedding_adapter(torch.cat((vectors, identity, physical), -1))

    def coefficient_parameters(self, hidden, vectors, depth=None):
        refined = self.coefficient_hidden(hidden, vectors)
        if depth is not None:
            return self.coefficient_heads[depth](refined)
        parameters = [head(refined[..., d, :]) for d, head in enumerate(self.coefficient_heads)]
        return tuple(torch.stack([p[i] for p in parameters], -2) for i in range(3))

    def forward(self, codes, model_aux=None, cond=None, amp=False):
        codes.validate(self.num_atoms, self.pair_depth)
        self._codes = codes
        try:
            with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                logits = RQTransformer.forward(self, codes.atoms, model_aux, cond, amp)
                hidden = self._native_hidden
                for d in range(1, self.pair_depth):
                    logits[..., d, :].scatter_(-1, codes.atoms[..., :d], -torch.inf)
                vectors = model_aux.dictionary.float().T[codes.atoms]
                return dict(atom_logits=logits,
                            coefficient_parameters=self.coefficient_parameters(hidden, vectors))
        finally:
            self._codes = self._native_hidden = None

    def init_cache(self):
        RQTransformer.init_cache(self)
        self._codes = self._native_hidden = None

    @torch.no_grad()
    def cached_forward(self, codes, model_aux=None, cond=None, amp=False, sample_loc=(0, 0, 0)):
        h, w, d = sample_loc
        self._codes = codes
        try:
            logits = RQTransformer.cached_forward(self, codes.atoms, model_aux, cond, amp, sample_loc)
            if d:
                logits.scatter_(-1, codes.atoms[:, h, w, :d], -torch.inf)
            return dict(atom_logits=logits, hidden=self._native_hidden)
        finally:
            self._codes = self._native_hidden = None

    @torch.no_grad()
    def sample_sparse(self, batch_size, model_aux, cond=None, *, atom_temperature=.9,
                      atom_top_k=0, atom_top_p=.9, amp=False, generator=None):
        if self.training:
            raise ValueError('Sampling requires evaluation mode')
        if (not math.isfinite(atom_temperature) or atom_temperature <= 0 or atom_top_k < 0
                or not 0 < atom_top_p <= 1):
            raise ValueError('Invalid categorical atom sampler settings')
        device = next(self.parameters()).device
        atoms = torch.zeros(batch_size, *self.block_size, dtype=torch.long, device=device)
        coefficients = torch.zeros_like(atoms, dtype=torch.float32)
        codes = ContinuousSparseCodes(atoms, coefficients)
        self.init_cache()
        try:
            for h in range(self.block_size[0]):
                for w in range(self.block_size[1]):
                    for d in range(self.pair_depth):
                        out = self.cached_forward(codes, model_aux, cond, amp, (h, w, d))
                        logits = out['atom_logits'].float() / atom_temperature
                        if 0 < atom_top_k < self.num_atoms:
                            logits = top_k_logits(logits, atom_top_k)
                        probabilities = logits.softmax(-1)
                        if atom_top_p < 1:
                            probabilities = top_p_probs(probabilities, atom_top_p)
                        a = torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)
                        atoms[:, h, w, d] = a
                        vectors = model_aux.dictionary.float().T[a][:, None]
                        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                            parameters = self.coefficient_parameters(out['hidden'], vectors, d)
                        c = ContinuousCoefficientMixture.sample(
                            parameters, self.coefficient_scales[d], generator)[:, 0]
                        if not c.isfinite().all():
                            raise FloatingPointError('Nonfinite continuous coefficient draw')
                        coefficients[:, h, w, d] = c
            codes.validate(self.num_atoms, self.pair_depth)
            return codes
        finally:
            self.init_cache()


def continuous_pair_loss(output, codes, coefficient_scales, *, atom_weight=1.5):
    """Same relative atom weight as compound128; coefficient density is physical."""
    if not math.isfinite(atom_weight) or atom_weight <= 0:
        raise ValueError('Positive finite atom likelihood weight required')
    logits = output['atom_logits'].float()
    atom_nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), codes.atoms.reshape(-1))
    scales = torch.as_tensor(coefficient_scales, device=logits.device)[..., None]
    logp = ContinuousCoefficientMixture.log_prob(
        output['coefficient_parameters'], codes.coefficients[..., None], scales).squeeze(-1)
    coefficient_nll = -logp.mean()
    loss = (atom_weight * atom_nll + coefficient_nll) / (atom_weight + 1)
    return loss, dict(atom_nll=atom_nll.detach(), coefficient_nll=coefficient_nll.detach(),
                      joint_nll=(atom_nll + coefficient_nll).detach())


def build_continuous_compound_model(arch, *, num_atoms, coefficient_scales, components=8, min_std=.03):
    from omegaconf import OmegaConf
    arch = OmegaConf.create(OmegaConf.to_container(arch, resolve=True))
    arch.vocab_size = int(num_atoms)
    arch.cumsum_depth_ctx = True
    model = ContinuousCompoundRQTransformer(RQTransformerConfig.create(arch), num_atoms=num_atoms,
        coefficient_scales=coefficient_scales, components=components, min_std=min_std)
    model.apply(model._init_weights)
    for head in model.coefficient_heads:
        head.reset_parameters()
    return model
