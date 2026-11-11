"""Derive a frozen beam/continuous codec from the selected learned stage1 weights."""
from pathlib import Path
import torch
from frozen_tokenizer import load_source as load_joint_source,LASER_HASH,digest_state
from src.models.continuous_beam_var import ContinuousBeamQuantizer
BOOK=Path('/tmp/laser-var-stage2-beam-continuous-20261106/inputs/continuous-codebook.pt')

def load_source(path,kind,device):
    model=load_joint_source(path,kind,device)
    model.quantize=ContinuousBeamQuantizer(model.quantize,beam_width=16,chunk_size=256,
        sequence_temperature=.01,max_relative_error=.01)
    if BOOK.exists():
        spec=torch.load(BOOK,map_location='cpu',weights_only=True)
        assert spec['source_checkpoint_sha256']==LASER_HASH
        assert spec['coefficient_representation']=='continuous_fp32' and spec['beam_width']==16
        torch.testing.assert_close(model.quantize.dictionary.cpu(),spec['dictionary'],atol=0,rtol=0)
        model.quantize.coefficient_scales.copy_(spec['coefficient_scales'])
    return model.eval().requires_grad_(False)
