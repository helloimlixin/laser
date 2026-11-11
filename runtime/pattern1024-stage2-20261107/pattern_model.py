"""Native RQ with four causal support predictions and one conditional pattern."""
import torch
from torch import nn

from src.rq_upstream import RQTransformer, RQTransformerConfig
from rqvae.utils.utils import sample_from_logits


def pack(atoms, pattern_ids, vocabulary):
    if atoms.shape[:-1] != pattern_ids.shape or atoms.shape[-1] != 4:
        raise ValueError('Expected four atom IDs and one pattern per site')
    result = atoms.long().clone()
    result[..., -1] = atoms[..., -1]*vocabulary+pattern_ids
    return result


def unpack(codes, vocabulary):
    atoms = codes.long().clone()
    patterns = atoms[..., -1].remainder(vocabulary)
    atoms[..., -1] = atoms[..., -1].div(vocabulary, rounding_mode='floor')
    return atoms, patterns


def verify_codes(codes, *, atoms=16384, patterns=1024):
    a, p = unpack(codes, patterns)
    assert codes.dtype == torch.long and codes.shape[-1] == 4
    assert bool(((a>=0)&(a<atoms)).all())
    assert bool(((p>=0)&(p<patterns)).all())
    assert bool((a.sort(-1).values.diff(dim=-1)>0).all())


class PatternRQTransformer(RQTransformer):
    def __init__(self, config, num_atoms=16384, patterns=1024):
        super().__init__(config)
        assert self.block_size[-1] == 4 and config.cumsum_depth_ctx
        self.num_atoms, self.pattern_count = int(num_atoms), int(patterns)
        dim, width = int(config.input_embed_dim), int(config.embed_dim)
        # Preserve support order: each prototype coordinate belongs to one OMP slot.
        self.pattern_support_proj = nn.Linear(4*dim, width, bias=False)
        self.pattern_fusion = nn.Sequential(nn.LayerNorm(3*width),
            nn.Linear(3*width, width), nn.SiLU(), nn.Linear(width, width))
        self.pattern_classifier = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, patterns))
        self._teacher_atoms = self._teacher_aux = self._cached_hidden = None
        self.classifier.register_forward_pre_hook(self._capture_hidden)

    def _capture_hidden(self, module, inputs):
        self._cached_hidden = inputs[0]

    def embed_with_model_aux(self, codes, aux):
        atoms, patterns = unpack(codes, self.pattern_count)
        return aux.dictionary.T[atoms]*aux.coefficient_patterns[patterns][..., None]

    def embed_depth_with_model_aux(self, codes, aux):
        atoms, _ = unpack(codes, self.pattern_count)
        return aux.dictionary.T[atoms]

    def pattern_logits(self, hidden, atom_vectors):
        support = self.pattern_support_proj(atom_vectors.flatten(-2))
        fused = torch.cat((hidden, support, hidden*support), dim=-1)
        return self.pattern_classifier(hidden+self.pattern_fusion(fused))

    def classify_head_outputs(self, hidden):
        assert self._teacher_atoms is not None
        atoms = self.classifier(hidden)
        for depth in range(1, 4):
            atoms[..., depth, :].scatter_(-1, self._teacher_atoms[..., :depth], -torch.inf)
        vectors = self._teacher_aux.dictionary.T[self._teacher_atoms]
        pattern = self.pattern_logits(hidden[..., -1, :], vectors)
        return dict(atom_logits=atoms, pattern_logits=pattern)

    def forward(self, codes, model_aux=None, cond=None, amp=False):
        self._teacher_atoms, _ = unpack(codes, self.pattern_count)
        self._teacher_aux = model_aux
        try:
            return super().forward(codes, model_aux=model_aux, cond=cond, amp=amp)
        finally:
            self._teacher_atoms = self._teacher_aux = self._cached_hidden = None

    @torch.no_grad()
    def cached_forward(self, codes, model_aux=None, cond=None, amp=False, sample_loc=(0, 0, 0)):
        logits = super().cached_forward(codes, model_aux=model_aux, cond=cond, amp=amp, sample_loc=sample_loc)
        h, w, depth = sample_loc
        if depth:
            atoms, _ = unpack(codes[:, h, w], self.pattern_count)
            logits.scatter_(-1, atoms[:, :depth], -torch.inf)
        return logits

    @torch.no_grad()
    def cached_pattern_logits(self, atoms, aux, *, amp):
        assert self._cached_hidden is not None
        with torch.autocast(device_type=atoms.device.type, dtype=torch.float16,
                            enabled=amp and atoms.device.type=='cuda'):
            return self.pattern_logits(self._cached_hidden[:, 0], aux.dictionary.T[atoms])

    def init_cache(self):
        super().init_cache()
        self._cached_hidden = None

    @torch.no_grad()
    def sample_patterns(self, count, aux, cond, *, atom_temperature=.9, atom_top_k=0,
                        atom_top_p=.9, pattern_temperature=1., pattern_top_k=0, pattern_top_p=.85, amp=True):
        height, width, _ = self.block_size
        device = next(self.parameters()).device
        atoms = torch.zeros(count, height, width, 4, device=device, dtype=torch.long)
        patterns = torch.zeros(count, height, width, device=device, dtype=torch.long)
        codes = pack(atoms, patterns, self.pattern_count)
        self.init_cache()
        try:
            for h in range(height):
                for w in range(width):
                    for depth in range(4):
                        logits = self.cached_forward(codes[:, :h+1], model_aux=aux, cond=cond,
                            amp=amp, sample_loc=(h, w, depth))
                        chosen = sample_from_logits(logits, temperature=atom_temperature,
                            top_k=min(atom_top_k or self.num_atoms, self.num_atoms), top_p=atom_top_p)
                        atoms[:, h, w, depth] = chosen
                        codes[:, h, w, depth] = chosen if depth<3 else chosen*self.pattern_count
                    logits = self.cached_pattern_logits(atoms[:, h, w], aux, amp=amp)
                    patterns[:, h, w] = sample_from_logits(logits, temperature=pattern_temperature,
                        top_k=min(pattern_top_k or self.pattern_count, self.pattern_count), top_p=pattern_top_p)
                    codes[:, h, w, -1] += patterns[:, h, w]
        finally:
            self.init_cache()
        return codes


def build_model(arch, *, atoms=16384, patterns=1024):
    from omegaconf import OmegaConf
    options = OmegaConf.create(OmegaConf.to_container(arch, resolve=True))
    options.block_size = [8, 8, 4] if list(options.block_size[:2]) == [8, 8] else [*options.block_size[:2], 4]
    options.vocab_size = atoms
    model = PatternRQTransformer(RQTransformerConfig.create(options), atoms, patterns)
    model.apply(model._init_weights)
    return model
