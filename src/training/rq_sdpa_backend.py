"""Opt-in SDPA execution of the pinned RQ attention, with unchanged weights.

Cache tensors keep the upstream layout. In particular, a cached query may
attend to every existing key; SDPA's upper-left non-square causal mask must
not be used for that case.
"""
from contextlib import contextmanager
from types import MethodType

import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

from src import rq_upstream
from rqvae.models.rqtransformer.attentions import MultiSelfAttention


def _sdpa_forward(self,x,caching=False,past_kv=None):
    batch,length,width=x.shape
    if not caching and past_kv is not None:
        raise ValueError('Non-cached attention cannot receive cached keys')
    transposed=x.transpose(0,1).contiguous()
    def project(layer):
        return layer(transposed).view(length,batch*self.n_head,width//self.n_head).transpose(0,1)
    k,q,v=project(self.key),project(self.query),project(self.value)
    past_length=0
    if past_kv is not None:
        past_key,past_value=past_kv
        past_length=past_key.shape[1]
        k=torch.cat((past_key,k),-2);v=torch.cat((past_value,v),-2)
    present=torch.stack((k,v)) if caching else None
    mask=None;causal=bool(self.mask and not past_length)
    if self.mask and past_length and length>1:
        query_position=torch.arange(length,device=x.device)[:,None]+past_length
        key_position=torch.arange(past_length+length,device=x.device)[None,:]
        mask=query_position>=key_position
    q=q.reshape(batch,self.n_head,length,-1)
    k=k.reshape(batch,self.n_head,past_length+length,-1)
    v=v.reshape(batch,self.n_head,past_length+length,-1)
    options=dict(attn_mask=mask,is_causal=causal,
        dropout_p=self.attn_drop.p if self.training else 0.)
    limit=getattr(self,'_sdpa_batch_limit',None)
    short_math=(getattr(self,'_sdpa_short_sequence_math',False)
        and self.training and not caching and length<=8)
    if short_math:
        # For short depth sequences, math avoids the fused kernel's padded
        # workspace and uses the full independent batch safely.
        with sdpa_kernel(SDPBackend.MATH):
            attended=F.scaled_dot_product_attention(q,k,v,**options)
    elif limit is not None and batch>limit:
        # Attention is independent along this axis. Bound the fused kernel's
        # batch/head launch geometry without changing the attention domain.
        attended=torch.cat([F.scaled_dot_product_attention(
            q[first:first+limit],k[first:first+limit],v[first:first+limit],**options)
            for first in range(0,batch,limit)],0)
    else:
        attended=F.scaled_dot_product_attention(q,k,v,**options)
    attended=attended.reshape(batch*self.n_head,length,-1).transpose(0,1).contiguous().view(length,batch,width)
    output=self.resid_drop(self.proj(attended)).transpose(0,1).contiguous()
    return (output,present) if caching else output


@contextmanager
def sdpa_attention(model,batch_limit=None,short_sequence_math=False):
    """Temporarily select SDPA execution without changing weights or cache layout."""
    if batch_limit is not None and (not isinstance(batch_limit,int) or batch_limit<1):
        raise ValueError('SDPA batch limit must be a positive integer')
    if not isinstance(short_sequence_math,bool):
        raise ValueError('Short-sequence math selection must be boolean')
    originals=[]
    try:
        for module in model.modules():
            if isinstance(module,MultiSelfAttention):
                originals.append((module,module.__dict__.get('forward'),
                    module.__dict__.get('_sdpa_batch_limit'),
                    module.__dict__.get('_sdpa_short_sequence_math')))
                if batch_limit is not None:module._sdpa_batch_limit=batch_limit
                module._sdpa_short_sequence_math=short_sequence_math
                module.forward=MethodType(_sdpa_forward,module)
        if not originals:raise ValueError('No native RQ attention modules found')
        yield len(originals)
    finally:
        for module,original,previous_limit,previous_math in reversed(originals):
            if original is None:module.__dict__.pop('forward',None)
            else:module.forward=original
            if previous_limit is None:module.__dict__.pop('_sdpa_batch_limit',None)
            else:module._sdpa_batch_limit=previous_limit
            if previous_math is None:module.__dict__.pop('_sdpa_short_sequence_math',None)
            else:module._sdpa_short_sequence_math=previous_math
