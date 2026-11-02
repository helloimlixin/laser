"""Look ahead over fixed integer products without changing any coefficients."""
import torch


@torch.no_grad()
def beam_quantize(quantizer, signals, *, beam_width=4, chunk_size=256, greedy=None):
    if beam_width<1 or chunk_size<1:raise ValueError('Positive beam width and chunk size required')
    greedy=quantizer._quantize_greedy(signals,chunk_size=chunk_size) if greedy is None else greedy
    if beam_width==1:return greedy
    flat=signals.reshape(-1,signals.shape[-1]).float()
    anchor_codes=greedy['codes'].reshape(-1,quantizer.depth)
    anchor_values=greedy['quantized'].reshape_as(flat)
    result_codes=torch.empty_like(anchor_codes);result_values=torch.empty_like(flat)
    for start in range(0,len(flat),chunk_size):
        x=flat[start:start+chunk_size];n=len(x)
        residual=x[:,None].clone()
        history=torch.empty(n,1,0,device=x.device,dtype=torch.long)
        for depth in range(quantizer.depth):
            branches=residual.shape[1]
            score=quantizer.scores(residual.reshape(-1,x.shape[-1]),depth).reshape(n,branches,-1)
            # Row constants differ between branches and must be retained.
            cost=residual.square().sum(-1,keepdim=True)-score
            width=min(beam_width,cost.shape[1]*cost.shape[2])
            chosen=cost.flatten(1).topk(width,dim=-1,largest=False).indices
            parent=chosen//quantizer.vocab_size;token=chosen%quantizer.vocab_size
            residual=residual.gather(1,parent[...,None].expand(-1,-1,x.shape[-1]))
            residual=residual-quantizer.embed_depth(token,depth)
            history=torch.cat([history.gather(1,parent[...,None].expand(-1,-1,depth)),token[...,None]],-1)
        best=residual.square().sum(-1).argmin(-1)
        row=torch.arange(n,device=x.device)
        codes=history[row,best];reconstruction=quantizer.embed(codes).sum(-2)
        reference=anchor_values[start:start+n]
        use_beam=(x-reconstruction).square().sum(-1)<(x-reference).square().sum(-1)
        result_codes[start:start+n]=torch.where(use_beam[:,None],codes,anchor_codes[start:start+n])
        result_values[start:start+n]=torch.where(use_beam[:,None],reconstruction,reference)
    return dict(codes=result_codes.reshape(*signals.shape[:-1],quantizer.depth),quantized=result_values.reshape_as(signals))
