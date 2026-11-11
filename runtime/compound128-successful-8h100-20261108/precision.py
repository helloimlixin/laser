"""Use BF16 training and native FP16 sampling, matching the successful run."""
import torch

def autocast(*args,enabled=True,**kwargs):
 return torch.autocast('cuda',enabled=enabled,dtype=torch.bfloat16 if torch.is_grad_enabled() else torch.float16)

def install_bf16():
 from rqvae.models.rqtransformer import transformers
 transformers.autocast=autocast
