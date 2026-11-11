"""Require bitwise teacher parity on real CC3M image latents before benchmarking."""
import os,sys,json
from pathlib import Path
BASE=Path('/tmp/laser-cc3m-aligned-20261106')
os.environ['RQ_UPSTREAM_ROOT']=str(BASE/'upstream');os.environ['LASER_FAST_BEAM_TOPK']='0'
sys.path[:0]=[str(BASE/'source')]
import torch,yaml
from src.training.cc3m_joint import build_aux,dataset
from src.training import beam_aligned_teacher as native
from src.training import beam_aligned_kernels as fast
cfg=yaml.safe_load((BASE/'config.yaml').read_text())
torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cudnn.allow_tf32=False
aux=build_aux(cfg,torch.device('cuda',0));data=dataset(cfg,'validation')
images=torch.stack([data[i][0] for i in range(2)]).cuda()
with torch.no_grad(),torch.autocast('cuda',enabled=False):
 latent=aux.encode(images)
 torch.manual_seed(261106)
 a=native.aligned_targets(aux.physical_book,latent,2.,return_details=True)
 torch.manual_seed(261106)
 b=native.aligned_targets(aux.physical_book,latent,2.,return_details=True,pool_fn=fast.beam_pool)
 for x,y in zip(a[:2],b[:2]):assert torch.equal(x,y),'Teacher codes or conditional probabilities changed'
 for key in ['candidates','costs','canonical_index','valid']:assert torch.equal(a[2][key],b[2][key]),key
 assert torch.equal(a[3],b[3]) and torch.equal(a[4],b[4])
 canonical=a[2]['candidates'][torch.arange(len(a[4]),device=latent.device),a[2]['canonical_index']].reshape(*latent.shape[:-1],4)
 reconstruction=aux.quantize_reconstruction(latent)['codes']
 assert torch.equal(canonical,reconstruction),'Aligned canonical teacher differs from saved beam reconstruction'
 assert all(not p.requires_grad for p in aux.parameters())
 report=dict(passed=True,real_CC3M_images=2,latent_sites=len(a[4]),beam_width=16,sequence_temperature=2.,
  sampled_codes_bitwise_equal=True,conditional_targets_bitwise_equal=True,
  beam_costs_and_candidates_bitwise_equal=True,canonical_codes_match_frozen_reconstruction=True,
  tokenizer_frozen=True)
 (BASE/'gpu-teacher-parity.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
