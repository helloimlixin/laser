"""An initially neutral learned summary of the pairs at a completed site."""
import torch
from torch import nn
from torch.nn import functional as F


class SitePairPooling(nn.Module):
    """Keep the native sum and add a depth-aware MLP or attention summary.

    Mixing is bidirectional within a *completed* site. The caller must shift
    the pooled site tokens before a causal spatial transformer; no current-site
    targets may reach the state used to predict that site.
    """

    def __init__(self, embed_dim, depth, *, width=480, heads=8, mode='attention'):
        super().__init__()
        if mode not in ('attention', 'mlp'):
            raise ValueError('site pooling mode must be attention or mlp')
        if width % heads:
            raise ValueError('pooling width must be divisible by the head count')
        if (4 * width) % (depth + 1):
            raise ValueError('width cannot exactly match the MLP mixer capacity')
        self.mode, self.depth, self.width, self.heads = mode, depth, width, heads
        self.enabled = True
        self.input_norm = nn.LayerNorm(embed_dim)
        self.input_proj = nn.Linear(embed_dim, width)
        self.depth_embedding = nn.Parameter(torch.empty(depth, width))
        self.token_norm = nn.LayerNorm(width)
        self.token_ffn = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width),
        )
        self.query = nn.Parameter(torch.empty(width))
        self.mix_norm = nn.LayerNorm(width)
        if mode == 'attention':
            self.q_proj = nn.Linear(width, width, bias=False)
            self.k_proj = nn.Linear(width, width, bias=False)
            self.v_proj = nn.Linear(width, width, bias=False)
            self.mix_output = nn.Linear(width, width, bias=False)
        else:
            # Four attention projections cost 4*w*w parameters. This two-layer
            # MLP costs (depth+1)*w*hidden, giving an exact capacity match.
            hidden = 4 * width // (depth + 1)
            self.mixer = nn.Sequential(
                nn.Linear(depth * width, hidden, bias=False),
                nn.GELU(), nn.Linear(hidden, width, bias=False),
            )
        self.output_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, embed_dim, bias=False)
        nn.init.normal_(self.depth_embedding, std=.02)
        nn.init.normal_(self.query, std=.02)
        nn.init.zeros_(self.output.weight)

    def forward(self, pairs):
        if pairs.shape[-2] != self.depth:
            raise ValueError('pooling requires the complete depth axis')
        native_sum = pairs.sum(dim=-2)
        if not self.enabled:
            if self.training or torch.is_grad_enabled():
                raise RuntimeError('sum-only ablation is inference only')
            return native_sum
        tokens = self.input_proj(self.input_norm(pairs)) + self.depth_embedding
        tokens = tokens + self.token_ffn(self.token_norm(tokens))
        normalized = self.mix_norm(tokens)
        query = normalized.mean(dim=-2) + self.query
        if self.mode == 'attention':
            leading = normalized.shape[:-2]
            head_dim = self.width // self.heads
            # Merge batch and site axes so CUDA can use its four-dimensional
            # fused attention kernels for these four-key reads.
            q = self.q_proj(query).reshape(-1, 1, self.heads, head_dim).transpose(1, 2)
            k = self.k_proj(normalized).reshape(-1, self.depth, self.heads, head_dim).transpose(1, 2)
            v = self.v_proj(normalized).reshape(-1, self.depth, self.heads, head_dim).transpose(1, 2)
            mixed = F.scaled_dot_product_attention(q, k, v, dropout_p=0.)
            mixed = mixed.transpose(1, 2).reshape(*leading, self.width)
            mixed = self.mix_output(mixed)
        else:
            mixed = self.mixer(normalized.flatten(start_dim=-2))
        summary = self.output(self.output_norm(query + mixed))
        return native_sum + summary
