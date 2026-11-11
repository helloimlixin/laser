import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from types import SimpleNamespace
import torch
from src.models.native_compound128_var import NativeCompound128VAR
from native_performance import install_typed_head


def test_type_specific_classifier_has_same_float32_loss_gradients_and_state_keys(monkeypatch):
 import dist
 monkeypatch.setattr(dist,'get_device',lambda:'cpu')
 torch.set_num_threads(4);torch.manual_seed(51)
 q=SimpleNamespace(Cvae=4,vocab_size=8,coefficient_bins=128,sparsity=2,v_patch_nums=(1,2,3))
 model=NativeCompound128VAR(SimpleNamespace(quantize=q),depth=2,width=32,heads=2,num_classes=3,token_width=16,token_layers=2,token_heads=2).train()
 model.cond_drop_rate=0
 for block in model.blocks:
  if hasattr(block.drop_path,'drop_prob'):block.drop_path.drop_prob=0
 labels=torch.tensor([0,1]);inputs=torch.randn(2,13,4);first=torch.randint(8,(2,14));atoms=torch.stack((first,(first+1)%8),-1);coefficients=torch.randint(128,(2,14,2))
 args=(labels,inputs,atoms,coefficients);keys=list(model.state_dict());reference=model(*args);reference.backward()
 gradients={n:p.grad.detach().clone() for n,p in model.named_parameters()};model.zero_grad(set_to_none=True)
 original=install_typed_head(model,compile_forward=False);actual=model(*args);actual.backward()
 torch.testing.assert_close(actual,reference,atol=1e-6,rtol=1e-6)
 for n,p in model.named_parameters():torch.testing.assert_close(p.grad,gradients[n],atol=1e-6,rtol=1e-4)
 assert list(model.state_dict())==keys
 model.eval()
 with torch.no_grad():torch.testing.assert_close(model(*args),original(*args),atol=0,rtol=0)
