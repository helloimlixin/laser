#!/usr/bin/env python3
"""Exercise delayed geometry gradients at the actual looped-model microbatch."""
import codecs
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.church_looped_pair import looped_pair_prior
from src.training.rqtransformer import LaserAux
from archive.scripts.train_church_calibrated import objective
p=ROOT/'outputs/church-looped-pair-20260911'
torch.serialization.add_safe_globals([codecs.encode])
torch.use_deterministic_algorithms(True)
torch.set_num_threads(8)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.allow_tf32=False
torch.set_float32_matmul_precision('highest')
saved=torch.load(p/'verify-batch-looped/last.pt',map_location='cpu',weights_only=False)
c=saved['config'];model=looped_pair_prior('looped',c['dropout']).cuda().train()
model.load_state_dict(saved['state_dict']);del saved
raw=torch.load(c['cache'],map_location='cpu',weights_only=False)
aux=LaserAux(Path(c['stage1']),16384,2048,3.,coeff_scales=c['tokenizer']['coeff_scales'],sparsity_level=4,soft_target_physical=True,clamp_coeffs=False).cuda().eval().requires_grad_(False)
loss,metrics=objective(model,aux,raw['holdout']['atoms'][:64].cuda().long(),raw['holdout']['coefficients'][:64].cuda(),'hard',.05,.05,stochastic=False)
loss.backward()
norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
assert torch.isfinite(norm) and metrics['geometry']>0
assert all(v.grad is None for v in aux.parameters())
r={'passed':True,'microbatch':64,'geometry_weight':.05,'geometry':metrics['geometry'],'gradient_norm':float(norm),'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'tokenizer_has_no_gradients':True}
(p/'geometry-backward.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r))
