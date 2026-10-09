"""Replay the existing stochastic OMP draws without per-kernel Python launches.

Chunk order, multinomial calls, FP32 arithmetic and joint Cholesky refits match
the eager teacher. CUDA graph construction preserves the caller's RNG streams.
Only the numerical-dependence check is deferred to the end of the bank draw.
"""
import torch


def _trajectory(signals, dictionary, gram, depth, temperature):
    correlation = signals @ dictionary
    residual = correlation
    available = torch.ones_like(correlation, dtype=torch.bool)
    rows = torch.arange(len(signals), device=signals.device)
    support = torch.empty(len(signals), 0, dtype=torch.long, device=signals.device)
    invalid = torch.zeros((), dtype=torch.bool, device=signals.device)
    chol = None
    for d in range(depth):
        scores = residual.square().masked_fill(~available, -torch.inf)
        probabilities = ((scores-scores.amax(-1,keepdim=True))/temperature).softmax(-1)
        atom = torch.multinomial(probabilities,1).squeeze(-1)
        available.scatter_(1,atom[:,None],False)
        if d == 0:
            chol = gram[atom,atom].sqrt()[:,None,None]
        else:
            cross = gram[support,atom[:,None]].unsqueeze(-1)
            solved = torch.linalg.solve_triangular(chol,cross,upper=False).transpose(1,2)
            diagonal = gram[atom,atom][:,None,None]-solved.square().sum(-1,keepdim=True)
            invalid = invalid | (diagonal <= 1e-7).any()
            # Keep a rejected trajectory safe until the deferred host check.
            chol = torch.cat((torch.cat((chol,signals.new_zeros(len(signals),d,1)),dim=-1),
                torch.cat((solved,diagonal.clamp_min(1e-7).sqrt()),dim=-1)),dim=-2)
        support = torch.cat((support,atom[:,None]),dim=-1)
        rhs = correlation.gather(1,support)
        coefficients = torch.cholesky_solve(rhs.unsqueeze(-1),chol).squeeze(-1)
        residual = correlation-coefficients.unsqueeze(1).bmm(gram[support]).squeeze(1)
    return support,coefficients,invalid


class StochasticOMPGraph:
    """One static chunk buffer, reused in the eager teacher's exact draw order."""
    def __init__(self, dictionary, gram, *, depth, temperature, sites):
        if dictionary.device.type != 'cuda' or dictionary.dtype != torch.float32:
            raise ValueError('CUDA graph OMP requires an FP32 CUDA dictionary')
        self.signals = dictionary.new_zeros(sites,dictionary.shape[0])
        self.graph = torch.cuda.CUDAGraph()
        device = dictionary.device
        # Warmup/capture must not spend any of the training RNG stream.
        with torch.random.fork_rng(devices=[device.index]),torch.cuda.device(device):
            current = torch.cuda.current_stream(device)
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                for _ in range(2):
                    _trajectory(self.signals,dictionary,gram,depth,temperature)
            current.wait_stream(stream)
            with torch.cuda.graph(self.graph,stream=stream):
                self.atoms,self.coefficients,self.invalid = _trajectory(
                    self.signals,dictionary,gram,depth,temperature)
            current.wait_stream(stream)

    def draw_into(self, signals, atoms, coefficients, invalid):
        self.signals.copy_(signals)
        self.graph.replay()
        atoms.copy_(self.atoms)
        coefficients.copy_(self.coefficients)
        invalid.copy_(self.invalid)


@torch.no_grad()
def stochastic_bank_graph(aux, signals, *, temperature, variants, site_chunk_size):
    """Keep all16 fresh alternatives; reduce launches without changing draws."""
    if signals.device.type != 'cuda':
        raise ValueError('CUDA graph stochastic teacher requires CUDA signals')
    dictionary,gram = aux.dictionary.float(),aux._stochastic_pair_gram
    x = signals.reshape(-1,signals.shape[-1]).float()
    depth = aux.coeff_scales.numel()
    if not hasattr(aux,'_stochastic_pair_graphs'):
        aux._stochastic_pair_graphs = {}
    cache = aux._stochastic_pair_graphs
    atoms = torch.empty(variants,len(x),depth,dtype=torch.long,device=x.device)
    physical = x.new_empty(variants,len(x),depth)
    chunks = (len(x)+site_chunk_size-1)//site_chunk_size
    invalid = torch.empty(variants,chunks,dtype=torch.bool,device=x.device)
    for variant in range(variants):
        for chunk,start in enumerate(range(0,len(x),site_chunk_size)):
            stop = min(start+site_chunk_size,len(x))
            key = (dictionary.data_ptr(),gram.data_ptr(),depth,temperature,stop-start)
            if key not in cache:
                cache[key] = StochasticOMPGraph(dictionary,gram,depth=depth,
                    temperature=temperature,sites=stop-start)
            cache[key].draw_into(x[start:stop],atoms[variant,start:stop],
                physical[variant,start:stop],invalid[variant,chunk])
    if invalid.any().item():
        raise ValueError('sampled support is numerically dependent; reject this trajectory')
    shape = (*signals.shape[:-1],variants,depth)
    return (atoms.permute(1,0,2).reshape(shape),
        (physical/aux.coeff_scales.float()).permute(1,0,2).reshape(shape))
