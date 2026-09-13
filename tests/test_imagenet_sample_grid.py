import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from src.imagenet_sample_grid import REQUESTED_CLASSES,sample_requested_grid


@pytest.mark.parametrize('training',[True,False])
@pytest.mark.parametrize('sampler_name,temperature',[('original',1.),('original_t09',.9)])
def test_fixed_class_rows_sampler_settings_and_training_rng_are_preserved(tmp_path,training,sampler_name,temperature):
    calls=[]
    class Model(torch.nn.Module):
        def sample(self,partial,**kwargs):
            torch.rand(3)  # Exercise RNG preservation despite stochastic sampling.
            assert kwargs['temperature']==temperature and kwargs['top_k']==16384 and kwargs['top_p']==.92
            assert kwargs['amp'] and kwargs['cached']
            calls.extend(kwargs['cond'].tolist())
            return kwargs['cond'][:,None,None,None].expand_as(partial).clone()
    class Tokenizer:
        quantizer=SimpleNamespace(vocab_size=131073)
        def decode_code(self,codes):
            return (codes[:,0,0,0].float()/1000*2-1)[:,None,None,None].expand(-1,3,32,32)
    model=Model().train(training)
    torch.manual_seed(57)
    before=torch.get_rng_state().clone()
    path=sample_requested_grid(model,Tokenizer(),tmp_path,1234,2.5,device=torch.device('cpu'),sampler_name=sampler_name)
    assert torch.equal(before,torch.get_rng_state()) and model.training==training
    expected=[c for c,_ in REQUESTED_CLASSES for _ in range(8)]
    assert calls==expected
    manifest=json.loads(path.with_suffix('.json').read_text())
    assert manifest['labels']==expected and manifest['rows']==10 and manifest['samples_per_class']==8
    assert manifest['sampler_name']==sampler_name and manifest['temperature']==temperature
    assert [c for c,_ in REQUESTED_CLASSES]==[9,22,90,200,289,849,106,277,258,926]
    pixels=np.asarray(Image.open(path))
    assert pixels.shape==(320,512,3)
    for row,(class_id,_) in enumerate(REQUESTED_CLASSES):
        assert np.all(pixels[row*32:(row+1)*32,256:]==round(class_id/1000*255))
    codes=torch.load(path.with_suffix('.codes.pt'),weights_only=True)
    assert codes[:,0,0,0].tolist()==expected
