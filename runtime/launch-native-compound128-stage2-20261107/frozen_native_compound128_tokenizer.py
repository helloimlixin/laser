"""Frozen native OMP compound128 codec with a fixed coefficient grid."""
from pathlib import Path
import torch
from frozen_tokenizer import load_source as load_original,LASER_HASH,digest_state
from src.models.native_compound128_var import NativeCompound128Quantizer
BOOK=Path('/tmp/laser-var-stage2-compound128-omp-20261107/inputs/compound128-codebook.pt')

def load_source(path,kind,device):
    model=load_original(path,kind,device)
    model.quantize=NativeCompound128Quantizer(model.quantize,chunk_size=2048,ridge=1e-5,max_coherence=.999)
    if BOOK.exists():
        spec=torch.load(BOOK,map_location='cpu',weights_only=True)
        assert spec['source_checkpoint_sha256']==LASER_HASH and spec['coefficient_bins']==128 and spec['encoding']=='native_greedy_omp'
        torch.testing.assert_close(model.quantize.dictionary.cpu(),spec['dictionary'],atol=0,rtol=0)
        model.quantize.coefficient_max.copy_(spec['coefficient_max']);model.quantize.companding.copy_(spec['companding'])
    return model.eval().requires_grad_(False)
