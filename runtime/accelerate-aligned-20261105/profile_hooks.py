"""Optional per-component CUDA event timing for private benchmark runs."""
import atexit
from functools import wraps
import json
from pathlib import Path
import torch


def install(path,accumulation):
    from rqvae.models.rqtransformer.transformers import RQTransformer
    from runtime.aligned_recipe import AlignedBeamRecipe
    records={}
    def wrap(cls,method,label):
        original=getattr(cls,method)
        @wraps(original)
        def call(*args,**kwargs):
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
            a.record()
            result=original(*args,**kwargs)
            b.record();records.setdefault(label,[]).append((a,b))
            return result
        setattr(cls,method,call)
    wrap(AlignedBeamRecipe,'targets','teacher')
    wrap(AlignedBeamRecipe,'loss','native_soft_ce')
    wrap(RQTransformer,'forward','ar_forward')
    wrap(torch.Tensor,'backward','backward')
    def save():
        try:
            torch.cuda.synchronize()
            result={}
            for name,pairs in records.items():
                values=[a.elapsed_time(b) for a,b in pairs]
                result[name]=dict(calls=len(values),last_update_ms=sum(values[-accumulation:]),
                                  last_microbatch_ms=values[-1])
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(result,indent=2)+'\n')
        except Exception as error:
            path.with_suffix('.error.txt').write_text(str(error))
    atexit.register(save)
