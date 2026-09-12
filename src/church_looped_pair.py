"""Compound pair RQ prior with matched unrolled or recurrent depth computation."""
import torch
from torch import nn

from src.training.rqtransformer import CompoundLaserRQTransformer
from src.church_ffhq_recipe import recipe_config
def sample_field(logits, temperature=1., top_k=0, top_p=None):
    if temperature <= 0 or (top_p is not None and not 0 < top_p <= 1):
        raise ValueError('Invalid sampling temperature or probability cutoff')
    logits = logits.float() / temperature
    if top_k:
        values = logits.topk(min(top_k, logits.shape[-1])).values
        logits = logits.masked_fill(logits < values[..., -1:], -float('inf'))
    probabilities = logits.softmax(-1)
    if top_p is not None and top_p < 1:
        ordered, indices = probabilities.sort(descending=True)
        # CUDA cumsum is unsupported under strict deterministic execution.
        remove = ordered.cpu().cumsum(-1).to(probabilities.device) >= top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        probabilities = probabilities.masked_fill(remove.scatter(-1, indices, remove), 0.)
        probabilities = probabilities / probabilities.sum(-1, keepdim=True)
    return torch.multinomial(probabilities, 1).squeeze(-1)



class LoopedAttentionStack(nn.Module):
    """Reuse weights across passes, never KV state across effective layers."""

    def __init__(self, blocks, loops):
        super().__init__()
        if not blocks or loops < 1:
            raise ValueError('Expected nonempty blocks and positive loop count')
        self.blocks = nn.ModuleList(blocks)
        self.loops = int(loops)
        self.init_cache()

    def forward(self, x):
        for _ in range(self.loops):
            for block in self.blocks:
                x = block(x)
        return x

    def init_cache(self):
        self._pass_caches = [
            [{'past_kv': None} for _ in self.blocks] for _ in range(self.loops)
        ]
        for block in self.blocks:
            block.init_cache()

    def cached_forward(self, x):
        for loop in range(self.loops):
            for index, block in enumerate(self.blocks):
                block._cache = self._pass_caches[loop][index]
                x = block.cached_forward(x)
        return x


class LoopedPairRQTransformer(CompoundLaserRQTransformer):
    def __init__(self, config, num_atoms, coeff_vocab_size, variant='looped',
                 core_layers=2, loops=3, micro_transformer_layers=2):
        if variant not in {'looped', 'unrolled'}:
            raise ValueError('Expected looped or unrolled variant')
        if core_layers < 1 or loops < 1 or config.head.n_layer != core_layers * loops:
            raise ValueError('Head depth must equal core_layers * loops')
        config = config.copy()
        # Explicit prefix additions support deterministic CUDA execution.
        config.cumsum_depth_ctx = False
        super().__init__(config, num_atoms, coeff_vocab_size,
                         micro_transformer_layers=micro_transformer_layers,
                         depth_specific_coeff_heads=True, pair_autoregressive=True)
        self.variant, self.core_layers, self.loops = variant, core_layers, loops
        blocks = self.head_transformer.blocks
        # Both arms consume identical initialization RNG, then start from the
        # exact same function. The control's repeated copies train independently.
        for index in range(core_layers, len(blocks)):
            blocks[index].load_state_dict(blocks[index % core_layers].state_dict())
        if variant == 'looped':
            self.head_transformer = LoopedAttentionStack(list(blocks[:core_layers]), loops)

    def embed_depth_with_model_aux(self, packed, model_aux):
        pairs = super().embed_depth_with_model_aux(packed, model_aux)
        prefix, values = pairs[..., 0, :], []
        values.append(prefix)
        for depth in range(1, pairs.shape[-2]):
            prefix = prefix + pairs[..., depth, :]
            values.append(prefix)
        return torch.stack(values, -2)

    @torch.no_grad()
    def sample_compound(self, batch_size, model_aux, cond=None, temperature=1.,
                        atom_top_k=2048, atom_top_p=None, coeff_top_p=.5,
                        coeff_top_k=0, atom_temperature=None,
                        coeff_temperature=None, amp=True):
        h_size, w_size, depth = self.block_size
        device = next(self.parameters()).device
        atoms = torch.zeros(batch_size, h_size, w_size, depth, device=device, dtype=torch.long)
        ids = torch.full_like(atoms, self.coeff_vocab_size // 2)
        packed = atoms * self.coeff_vocab_size + ids
        atom_temperature = temperature if atom_temperature is None else atom_temperature
        coeff_temperature = temperature if coeff_temperature is None else coeff_temperature
        self.init_cache()
        try:
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                for h in range(h_size):
                    for w in range(w_size):
                        for d in range(depth):
                            hidden = self.cached_head_output(packed, model_aux, cond, (h, w, d), amp=amp)
                            logits = self.classifier(hidden)
                            if d:
                                logits = logits.clone()
                                logits.scatter_(1, atoms[:, h, w, :d], -float('inf'))
                            atom = sample_field(logits, atom_temperature, atom_top_k, atom_top_p)
                            coefficients = self.coefficient_logits(
                                hidden, model_aux.dictionary.t()[atom], depth_index=d)
                            coefficient = sample_field(coefficients, coeff_temperature, coeff_top_k, coeff_top_p)
                            atoms[:, h, w, d], ids[:, h, w, d] = atom, coefficient
                            packed[:, h, w, d] = atom * self.coeff_vocab_size + coefficient
        finally:
            self.init_cache()
        return atoms, ids


def looped_pair_prior(variant='looped', dropout=.15):
    config = recipe_config('balanced')
    config.body.block.resid_pdrop = float(dropout)
    config.head.block.resid_pdrop = float(dropout)
    return LoopedPairRQTransformer(config, 16384, 2048, variant=variant)
