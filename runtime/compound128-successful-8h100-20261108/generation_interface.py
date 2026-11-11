"""Check valid integer generation and equivalence of cached and teacher-forced logits."""
import torch

@torch.no_grad()
def verify_generation(profile,model,aux,device):
 training=model.training;model.eval()
 with torch.random.fork_rng(devices=[device.index]):
  torch.manual_seed(261108128);labels=torch.tensor([933,849],device=device)
  codes=profile.sample(model,aux,2,labels)
  assert codes.shape==(2,8,8,8)
  from src.training.laser_hard_ar import verify_tokens
  verify_tokens(aux,codes)
  images=profile.decode(aux,codes);assert images.shape==(2,3,256,256) and torch.isfinite(images).all()
  full=model(codes,model_aux=aux,cond=labels,amp=True)
  future=codes.flip(0).clone()
  altered=model(future,model_aux=aux,cond=labels,amp=True)
  torch.testing.assert_close(full['atom_logits'][:,0,0,0].float(),altered['atom_logits'][:,0,0,0].float(),rtol=.02,atol=.04)
  future[:,0,0,0]=codes[:,0,0,0]
  altered=model(future,model_aux=aux,cond=labels,amp=True)
  torch.testing.assert_close(full['coeff_logits'][:,0,0,0].float(),altered['coeff_logits'][:,0,0,0].float(),rtol=.02,atol=.04)
  model.init_cache();errors=[]
  try:
   for column in range(2):
    for event in range(8):
     value=model.cached_forward(codes[:,:1],model_aux=aux,cond=labels,amp=True,sample_loc=(0,column,event))
     expected=full['atom_logits' if event%2==0 else 'coeff_logits'][:,0,column,event//2]
     actual=value[:,:aux.num_atoms] if event%2==0 else value[:,aux.num_atoms:]
     assert torch.equal(torch.isfinite(actual),torch.isfinite(expected))
     mask=torch.isfinite(expected);err=float((actual[mask].float()-expected[mask].float()).abs().max());errors.append(err)
     torch.testing.assert_close(actual[mask].float(),expected[mask].float(),rtol=.02,atol=.04)
  finally:model.init_cache()
 model.train(training)
 return dict(passed=True,compound_autoregressive_sampling=True,atoms=aux.num_atoms,coefficient_bins=128,finite_images=True,integer_codes=True,cached_vs_teacher_forced_sites=2,all_eight_events_checked=True,future_token_independence_checked=True,max_cached_logit_error=max(errors))
