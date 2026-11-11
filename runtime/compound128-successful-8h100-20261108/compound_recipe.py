"""Frozen successful LASER dictionary with 128-bin compound coefficients."""
import json,hashlib
from pathlib import Path
import torch
from torch.nn import functional as F
from omegaconf import OmegaConf
from src.training.rq_imagenet_recipes import LaserRecipe
from src.models.laser_tokenizer_adapter import FrozenLaserTokenizer
from src.training.laser_hard_ar import sparse_encoding,verify_tokens,nearest_coefficient_ids
from src.training.laser_rq_minimal import frozen_digest


def pair_loss(atom_logits,coefficient_logits,atoms,probabilities):
 atom=F.cross_entropy(atom_logits.float().reshape(-1,atom_logits.shape[-1]),atoms.reshape(-1))
 coefficient=-(probabilities*coefficient_logits.float().log_softmax(-1)).sum(-1).mean()
 return atom+coefficient,atom.detach(),coefficient.detach()


class CompoundRecipe(LaserRecipe):
 kind='compound128_successful'
 def __init__(self,base):
  super().__init__(base)
  self.codebook_sha256=hashlib.sha256((self.base/'inputs/compound-codec.pt').read_bytes()).hexdigest() if (self.base/'inputs/compound-codec.pt').exists() else 'calibration-pending'
  self._loss=torch.compile(pair_loss,fullgraph=True,dynamic=True)
 def tokenizer(self,device):
  spec=torch.load(self.base/'inputs/compound-codec.pt',map_location='cpu',weights_only=True)
  aux=FrozenLaserTokenizer(self.base/'inputs/stage1-model.pt',num_atoms=int(spec['dictionary'].shape[1]),coeff_vocab_size=128,coeff_max=3.,coeff_scales=spec['coefficient_scales'].tolist(),sparsity_level=4)
  aux.dictionary=spec['dictionary'].float().clone()
  aux.vocab_size=aux.num_atoms+128
  return aux.to(device).requires_grad_(False).eval(),None
 def model(self,arch):
  from src.models.laser_native_compound_adapter import build_native_compound_model
  model=build_native_compound_model(arch,num_atoms=int(self.recipe.tokenizer.num_atoms),coefficient_bins=128,depth=4)
  model.apply(model._init_weights)
  return model
 def adapt(self,model,aux):return None
 @torch.no_grad()
 def clean_components(self,aux,images):
  parts=[]
  with torch.autocast('cuda',enabled=False):
   if not hasattr(aux,'_hard_ar_gram'):aux._hard_ar_gram=(aux.dictionary.T@aux.dictionary).contiguous()
   for chunk in images.split(int(self.recipe.targets.encoder_batch_size)):
    z=aux.quant_conv(aux.encoder(chunk.float())).permute(0,2,3,1).contiguous()
    a,c=sparse_encoding(z,aux.dictionary,4,gram=aux._hard_ar_gram)
    parts.append((a,c/aux.coeff_scales))
  return tuple(torch.cat([p[i] for p in parts]) for i in (0,1))
 @torch.no_grad()
 def targets(self,aux,images):
  atoms,values=self.clean_components(aux,images)
  sigma=float(self.recipe.targets.coefficient_sigma_bins)*(float(aux.coeff_bins[1]-aux.coeff_bins[0]))
  probability=(-(values[...,None]-aux.coeff_bins).square()/(2*sigma*sigma)).softmax(-1)
  ids=torch.multinomial(probability.reshape(-1,128),1).reshape_as(atoms)
  tokens=torch.empty(*atoms.shape[:-1],8,device=images.device,dtype=torch.long)
  tokens[...,0::2]=atoms;tokens[...,1::2]=ids+aux.num_atoms
  return dict(tokens=tokens,coefficient_probabilities=probability),tokens
 @torch.no_grad()
 def hard_codes(self,aux,images):
  atoms,values=self.clean_components(aux,images);ids=nearest_coefficient_ids(values,aux.coeff_bins)
  tokens=torch.empty(*atoms.shape[:-1],8,device=images.device,dtype=torch.long);tokens[...,0::2]=atoms;tokens[...,1::2]=ids+aux.num_atoms
  return tokens
 def verify_targets(self,aux,targets,codes):
  verify_tokens(aux,codes);p=targets['coefficient_probabilities'];assert p.shape==(*codes[...,0::2].shape,128)
  assert p.dtype==torch.float32 and torch.isfinite(p).all() and (p>=0).all()
  torch.testing.assert_close(p.sum(-1),torch.ones_like(p[...,0]),rtol=2e-5,atol=2e-5)
  return dict(soft_targets=True,stochastic_prefixes=True,soft_coefficient_labels=True,atom_targets='hard ordered OMP support',atoms=aux.num_atoms,coefficient_bins=128,coefficient_sigma_bins=float(self.recipe.targets.coefficient_sigma_bins),mean_coefficient_target_entropy=float(-(p*p.clamp_min(1e-30).log()).sum(-1).mean()),integer_codes=True,shape=list(codes.shape),auxiliary_loss_terms=0)
 def loss(self,model,logits,targets):
  value,a,c=self._loss(logits['atom_logits'],logits['coeff_logits'],targets['tokens'][...,0::2],targets['coefficient_probabilities'])
  self.last_loss_metrics=dict(atom_ce=a,coefficient_ce=c)
  return value
 def target_temperature(self):return float(self.recipe.targets.coefficient_sigma_bins)
 def sampling_settings(self):
  s=self.recipe.sampling
  return dict(temperature=float(s.atom_temperature),top_k=[int(s.atom_top_k)],top_p=[float(s.atom_top_p)],coefficient_temperature=float(s.coefficient_temperature),coefficient_top_k=int(s.coefficient_top_k),coefficient_top_p=float(s.coefficient_top_p))
 def logging_sampling_settings(self):return dict(sampler_temperature=self.recipe.sampling.atom_temperature,sampler_top_k=self.recipe.sampling.atom_top_k,sampler_top_p=self.recipe.sampling.atom_top_p)
 def digest(self,aux):return frozen_digest(aux)
 def checkpoint_extra(self):return dict(recipe=OmegaConf.to_container(self.recipe,resolve=True),compound_codec_sha256=self.codebook_sha256)
 def verify_resume(self,payload):
  saved=dict(payload['recipe']);current=OmegaConf.to_container(self.recipe,resolve=True)
  saved.pop('microbatch',None);current.pop('microbatch',None)
  assert saved==current
  assert payload['compound_codec_sha256']==self.codebook_sha256
 def verify_codec(self,aux):
  spec=torch.load(self.base/'inputs/compound-codec.pt',map_location='cpu',weights_only=True)
  assert not aux.training and not any(p.requires_grad for p in aux.parameters())
  torch.testing.assert_close(aux.dictionary.cpu(),spec['dictionary'],rtol=0,atol=0)
  assert aux.num_atoms==int(self.recipe.tokenizer.num_atoms) and aux.coeff_vocab_size==128
  return dict(stage1_frozen=True,stage1_preservation_guard_passed=True,compound_codec_sha256=self.codebook_sha256)
 def metadata(self):
  proof=self.base/'tokenizer-verification.json';q=json.loads(proof.read_text()) if proof.exists() else {}
  return dict(architecture='upstream RQ spatial42/depth6 with atom-conditioned compound coefficient head',stage1='frozen successful learned dictionary and original CNN',num_atoms=int(self.recipe.tokenizer.num_atoms),coefficient_bins=128,coefficient_bits=7,vocabulary=int(self.recipe.tokenizer.num_atoms)+128,scalar_events_per_site=8,codebook_sha256=self.codebook_sha256,objective='atom CE plus conditional soft coefficient CE; joint pair NLL',atom_ce_weight=1.,coefficient_ce_weight=1.,soft_labels=True,stochastic_codes=True,coefficient_sigma_bins=float(self.recipe.targets.coefficient_sigma_bins),auxiliary_loss_terms=0,measured_tokenizer_rfid=q.get('rfid_hard'),source_best_fid=14.1633995743926,native_sampling=self.sampling_settings(),recipe_lineage=OmegaConf.to_container(self.recipe.lineage),precision='bfloat16',execution=json.loads((self.base/'inputs/execution.json').read_text()))
