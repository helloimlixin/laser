"""A geometric residual-query correction to an existing atom classifier."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class DictionaryAtomReadout(nn.Module):
    """Share learning across similar atoms using fixed dictionary directions.

    OMP selects by absolute residual correlation. Marginalizing the two signs
    gives logcosh(query dot atom), rather than an unrelated logit per atom.
    The existing classifier is retained, including its exact parameter objects.
    """
    def __init__(self, original, dictionary, initial_gain=.1):
        super().__init__()
        self.layer_norm = original.layer_norm
        self.linear = original.linear
        self.register_buffer('dictionary', dictionary.detach().float(), persistent=False)
        self.query = nn.Linear(self.linear.in_features, dictionary.shape[0])
        nn.init.normal_(self.query.weight, std=.01/math.sqrt(self.linear.in_features))
        nn.init.zeros_(self.query.bias)
        self.log_scale = nn.Parameter(torch.tensor(math.log(math.sqrt(dictionary.shape[0]))))
        self.raw_gain = nn.Parameter(torch.tensor(math.log(math.expm1(initial_gain))))

    def forward(self, hidden):
        normalized = self.layer_norm(hidden)
        query = F.normalize(self.query(normalized).float(), dim=-1)
        projection = F.linear(query, self.dictionary.T).float()*self.log_scale.exp()
        geometry = torch.logaddexp(projection, -projection)-math.log(2.)
        geometry = geometry-geometry.mean(-1, keepdim=True)
        return self.linear(normalized).float()+F.softplus(self.raw_gain)*geometry


def install_dictionary_atom_readout(model, dictionary):
    if isinstance(model.classifier, DictionaryAtomReadout):
        raise ValueError('dictionary atom readout is already installed')
    head = DictionaryAtomReadout(model.classifier, dictionary).to(next(model.parameters()).device)
    model.classifier = head
    return [*head.query.parameters(), head.log_scale, head.raw_gain]


@torch.no_grad()
def cluster_dictionary_directions(dictionary, groups=256, iterations=20):
    """Projective spherical clusters: an atom and its negative share geometry."""
    vectors = F.normalize(dictionary.T.float(), dim=-1)
    rng = torch.Generator(device=vectors.device).manual_seed(261029)
    centers = vectors[torch.randperm(len(vectors), generator=rng, device=vectors.device)[:groups]].clone()
    for _ in range(iterations):
        similarities = vectors@centers.T
        labels = similarities.abs().argmax(-1)
        signs = (vectors*centers[labels]).sum(-1).sign().unsqueeze(-1)
        sums = torch.zeros_like(centers).index_add_(0, labels, vectors*signs)
        occupied = sums.norm(dim=-1)>1e-8
        centers[occupied] = F.normalize(sums[occupied], dim=-1)
        if not occupied.all():
            count = int((~occupied).sum())
            worst = similarities.abs().amax(-1).topk(count, largest=False).indices
            centers[~occupied] = vectors[worst]
    return (vectors@centers.T).abs().argmax(-1)


class ClusteredAtomReadout(nn.Module):
    """Depth-specific coarse corrections shared by geometrically related IDs."""
    def __init__(self, original, labels, depth=4, groups=256):
        super().__init__()
        self.layer_norm, self.linear = original.layer_norm, original.linear
        self.register_buffer('groups', labels.long())
        self.coarse = nn.ModuleList([nn.Linear(self.linear.in_features, groups) for _ in range(depth)])
        for head in self.coarse:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.active_depth = 0

    def forward(self, hidden):
        normalized = self.layer_norm(hidden)
        if hidden.ndim >= 3:
            correction = torch.stack([head(normalized[..., d, :])
                                      for d, head in enumerate(self.coarse)], -2)
        else:
            correction = self.coarse[self.active_depth](normalized)
        return self.linear(normalized)+correction[..., self.groups]


def install_clustered_atom_readout(model, dictionary):
    labels = cluster_dictionary_directions(dictionary)
    head = ClusteredAtomReadout(model.classifier, labels).to(next(model.parameters()).device)
    model.classifier = head
    cached = model.cached_head_output

    def cached_with_depth(packed, model_aux, cond, sample_loc, amp=True):
        head.active_depth = int(sample_loc[-1])
        return cached(packed, model_aux, cond, sample_loc, amp)

    model.cached_head_output = cached_with_depth
    return list(head.coarse.parameters())


def install_projected_dictionary_readout(model, dictionary):
    """Constrain each support query to the complement of its selected prefix."""
    from src.orthogonal_sparse_codec import ordered_support_basis
    head = DictionaryAtomReadout(model.classifier, dictionary, initial_gain=1.).to(next(model.parameters()).device)
    model.classifier = head
    state = {}

    def full_input(module, args, kwargs):
        packed = args[0] if args else kwargs['packed']
        state.update(mode='full', atoms=packed[..., 0].long())

    def project(module, args, query):
        atoms = state['atoms']
        if state['mode'] == 'full':
            basis, _ = ordered_support_basis(dictionary.T[atoms])
            residuals = []
            for d in range(atoms.shape[-1]):
                q = query[..., d, :].float()
                prior = basis[..., :d, :]
                residuals.append(q-((q.unsqueeze(-2)*prior).sum(-1).unsqueeze(-1)*prior).sum(-2))
            return torch.stack(residuals, -2)
        if atoms.shape[-1] == 0:
            return query
        basis, _ = ordered_support_basis(dictionary.T[atoms])
        q = query.float()
        return q-((q.unsqueeze(-2)*basis).sum(-1).unsqueeze(-1)*basis).sum(-2)

    model.register_forward_pre_hook(full_input, with_kwargs=True)
    head.query.register_forward_hook(project)
    cached = model.cached_head_output

    def cached_with_projection(packed, model_aux, cond, sample_loc, amp=True):
        h, w, d = sample_loc
        state.update(mode='cached', atoms=packed[:, h, w, :d, 0].long())
        return cached(packed, model_aux, cond, sample_loc, amp)

    model.cached_head_output = cached_with_projection
    return [*head.query.parameters(), head.log_scale, head.raw_gain]
