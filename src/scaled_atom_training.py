"""Training adapters for the frozen scaled-atom RQ codebook.

Keep the reconstruction experiment and released architecture source unchanged.
Soft targets include every codeword, with bounded intermediate allocations.
"""
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn
import torch.distributed as dist

from src.scaled_atom_rq import ScaledAtomRQ, FrozenSparseBackbone


class TrainingScaledAtomRQ(ScaledAtomRQ):
    @torch.no_grad()
    def get_soft_codes(self, x, temp=.5, stochastic=True, chunk_size=128):
        if temp <= 0 or chunk_size < 1:
            raise ValueError('Positive temperature and chunk size required')
        shape = x.shape[:-1]
        residual = x.reshape(-1, x.shape[-1]).float().clone()
        targets = residual.new_empty(len(residual), self.depth, self.vocab_size)
        codes = torch.empty(len(residual), self.depth, device=x.device, dtype=torch.long)
        for d in range(self.depth):
            for start in range(0, len(residual), chunk_size):
                r = residual[start:start+chunk_size]
                correlations = r @ self.dictionary
                # ||r||² cancels in the softmax. This is the full expanded book,
                # evaluated via one dictionary projection instead of B projections.
                score = (2*correlations[...,None]*self.levels
                    - self.atom_norms_squared[:,None]*self.levels.square()).flatten(1)
                score = torch.cat([score.new_zeros(len(r),1), score], 1)
                probability = (score/temp).softmax(-1)
                chosen = (torch.multinomial(probability, 1).squeeze(-1) if stochastic
                          else score.argmax(-1))
                targets[start:start+len(r),d] = probability
                codes[start:start+len(r),d] = chosen
                r.sub_(self.embed(chosen))
        return targets.reshape(*shape,self.depth,self.vocab_size), codes.reshape(*shape,self.depth)

    def forward(self, z):
        value = self.quantize(z, return_trajectory=True)
        loss = (z[...,None,:]-value['prefixes']).square().mean()
        return value['quantized'], loss, value['codes']


class FrozenScaledTokenizer(nn.Module):
    def __init__(self, checkpoint, codebook, levels=8):
        super().__init__()
        self.backbone = FrozenSparseBackbone(Path(checkpoint))
        spec = torch.load(codebook, map_location='cpu', weights_only=True)
        torch.testing.assert_close(spec['dictionary'], self.backbone.dictionary, rtol=0, atol=0)
        self.quantizer = TrainingScaledAtomRQ(spec['dictionary'], spec['levels'][str(levels)], depth=4)
        self.code_shape = torch.Size([8,8,4])
        self.requires_grad_(False).eval()

    def encode(self, images):
        return self.backbone.encode(images)

    def decode(self, z):
        return self.backbone.decode(z)

    def decode_code(self, codes):
        return self.decode(self.quantizer.embed(codes).sum(-2))

    def get_code_emb_with_depth(self, codes):
        return self.quantizer.get_code_emb_with_depth(codes)


class ChunkedSoftCrossEntropy(torch.autograd.Function):
    """Exact FP32 soft CE, recomputing row chunks in backward to bound memory."""
    @staticmethod
    def forward(ctx, logits, targets, chunk_size):
        assert logits.shape == targets.shape and not targets.requires_grad
        ctx.save_for_backward(logits, targets)
        ctx.chunk_size = chunk_size
        xs, ys = logits.reshape(-1, logits.shape[-1]), targets.reshape(-1, targets.shape[-1])
        loss = torch.zeros((), device=logits.device, dtype=torch.float32)
        for start in range(0, len(xs), chunk_size):
            loss -= (xs[start:start+chunk_size].float().log_softmax(-1)*ys[start:start+chunk_size]).sum()
        return loss/len(xs)

    @staticmethod
    def backward(ctx, grad_output):
        logits, targets = ctx.saved_tensors
        xs, ys = logits.reshape(-1, logits.shape[-1]), targets.reshape(-1, targets.shape[-1])
        gradient = torch.empty_like(xs)
        for start in range(0, len(xs), ctx.chunk_size):
            y = ys[start:start+ctx.chunk_size]
            g = xs[start:start+ctx.chunk_size].float().softmax(-1)*y.sum(-1,keepdim=True)-y
            gradient[start:start+ctx.chunk_size] = g*(grad_output/len(xs))
        return gradient.reshape_as(logits), None, None


def soft_cross_entropy(logits, targets, chunk_size=128):
    return ChunkedSoftCrossEntropy.apply(logits, targets, chunk_size)


def advance_cosine_scheduler(scheduler,optimizer,multiplier):
    # PyTorch CosineAnnealingLR updates recursively from the optimizer LR.
    # Restore the unscaled value before stepping, then apply the multiplier once.
    if multiplier<=0:
        raise ValueError('Learning-rate multiplier must stay positive')
    for group in optimizer.param_groups:
        group['lr']/=multiplier
    scheduler.step()
    for group in optimizer.param_groups:
        group['lr']*=multiplier


def accumulated_update(ddp, tokenizer, optimizer, scaler, batches, temperature, max_gn=1.):
    total = sum(len(z) for z in batches)
    device = next(ddp.parameters()).device
    optimizer.zero_grad(set_to_none=True)
    weighted_loss = torch.zeros((),device=device)
    for i,z in enumerate(batches):
        with (ddp.no_sync() if i+1 < len(batches) else nullcontext()):
            with torch.no_grad():
                targets,codes = tokenizer.quantizer.get_soft_codes(z.to(device,non_blocking=True),
                    temp=temperature,stochastic=True)
            logits = ddp(codes,model_aux=tokenizer,amp=True)
            loss = soft_cross_entropy(logits,targets)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite soft cross entropy')
            weight = len(z)/total
            scaler.scale(loss*weight).backward()
            weighted_loss += loss.detach()*weight
            del targets, codes, logits, loss
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(ddp.parameters(),max_gn)
    old_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    success = scaler.get_scale() >= old_scale
    success_values = [None]*dist.get_world_size()
    dist.all_gather_object(success_values,success)
    assert len(set(success_values)) == 1, 'Ranks disagree on optimizer update'
    values = torch.stack([weighted_loss,norm.float()])
    dist.all_reduce(values)
    values /= dist.get_world_size()
    return dict(loss=values[0].item(),gradient_norm=values[1].item(),amp_scale=scaler.get_scale(),
        optimizer_updated=success,global_images=total*dist.get_world_size())


class FidLearningRateControl:
    """Downward-only multiplier on the released cosine schedule, using 4k FID."""
    def __init__(self):
        self.best = float('inf')
        self.bad = 0
        self.stale = 0
        self.last_reduction_epoch = -100
        self.multiplier = 1.

    def observe(self, epoch, fid):
        improved = fid < self.best-.1
        self.stale = 0 if improved else self.stale+1
        self.bad = self.bad+1 if fid > self.best+.25 else 0
        self.best = min(self.best,fid)
        reason = None
        if epoch-self.last_reduction_epoch >= 10 and (self.bad>=2 or self.stale>=3):
            reason = 'two_regressions' if self.bad>=2 else 'three_checks_without_improvement'
            self.multiplier *= .5
            self.bad = self.stale = 0
            self.last_reduction_epoch = epoch
        return reason


@torch.no_grad()
def evaluate_validation(model,tokenizer,latents,device,rank,temperature):
    model.eval()
    total=torch.zeros(7,device=device,dtype=torch.float64)
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(61000+rank)
        for z in latents.split(4):
            z=z.to(device)
            targets,codes=tokenizer.quantizer.get_soft_codes(z,temp=temperature,stochastic=True)
            logits=model(codes,model_aux=tokenizer,amp=True)
            total[0]+=soft_cross_entropy(logits,targets)*len(z)
            del targets,logits
            hard=tokenizer.quantizer.quantize(z)['codes']
            logits=model(hard,model_aux=tokenizer,amp=True).float()
            depth=model.compute_codebook_loss(logits,hard)
            total[1]+=depth.mean()*len(z)
            total[2:6]+=depth.double()*len(z)
            total[6]+=len(z)
    dist.all_reduce(total)
    assert total[6]==300 and torch.isfinite(total).all()
    total[:6]/=total[6]
    model.train()
    return dict(soft_ce=total[0].item(),hard_code_nll=total[1].item(),
        **{f'depth_{d}_nll':total[2+d].item() for d in range(4)},images=300)
