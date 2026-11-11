"""Load the exact selected trained 680-token LASER autoencoder, with no refitting."""
import hashlib
from pathlib import Path
import torch
from src.models.trainable_joint_var_tokenizer import FourScaleJointVAE
LASER_HASH='cdea5ae33dff9943653f0fa8919db277547e6a4be39588472f26c45f0ce80599'

def digest_state(model):
 h=hashlib.sha256()
 for n,t in model.state_dict().items():
  h.update(n.encode());h.update(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
 return h.hexdigest()

def load_source(path,kind,device):
 assert kind=='laser'
 with Path(path).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==LASER_HASH
 p=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
 assert p['progress']==dict(epoch=10,batch=0,step=100090)
 model=FourScaleJointVAE()
 model.load_state_dict(p['model'],strict=True)
 model.to(device).eval().requires_grad_(False)
 model.quantize.sync_assignment_tables()
 assert list(model.quantize.v_patch_nums)==[2,4,8,16]
 assert model.quantize.vocab_size==4096 and model.quantize.sparsity==2
 assert sum(x*x for x in model.quantize.v_patch_nums)*model.quantize.sparsity==680
 return model
