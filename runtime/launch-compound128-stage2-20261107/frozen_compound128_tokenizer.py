"""Frozen selected CNN/dictionary with beam16 supports and 128 coefficient bins."""
from pathlib import Path
import torch
from frozen_tokenizer import load_source as load_original,LASER_HASH,digest_state
from src.models.compound_beam128_var import CompoundBeam128Quantizer
BOOK=Path('/tmp/laser-var-stage2-compound128-20261107/inputs/compound128-codebook.pt')

def load_source(path,kind,device):
    model=load_original(path,kind,device)
    model.quantize=CompoundBeam128Quantizer(model.quantize,beam_width=16,chunk_size=256,
        sequence_temperature=.01,max_relative_error=.01)
    if BOOK.exists():
        spec=torch.load(BOOK,map_location='cpu',weights_only=True)
        assert spec['source_checkpoint_sha256']==LASER_HASH and spec['coefficient_bins']==128
        torch.testing.assert_close(model.quantize.dictionary.cpu(),spec['dictionary'],atol=0,rtol=0)
        model.quantize.coefficient_max.copy_(spec['coefficient_max']);model.quantize.companding.copy_(spec['companding'])
    return model.eval().requires_grad_(False)
