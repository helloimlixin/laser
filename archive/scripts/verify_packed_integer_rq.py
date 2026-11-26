#!/usr/bin/env python3
"""Check all 8M pair classes and benchmark exact global training/sampling."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    import torch
    from torch.nn import functional as F
    from src import rq_upstream
    from src.models.laser_pair_id_head import ProjectedPairIDLinear
    from src.training.pair_id_linear_loss import chunked_pair_linear_cross_entropy
    from rqvae.utils.utils import sample_from_logits

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',type=int,default=0)
    parser.add_argument('--gpu-memory-gib',type=float,default=6.)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(261017512)
    vocabulary = 16384*512
    head = ProjectedPairIDLinear(1536,vocabulary,128,chunked_training_loss=True)
    for p in head.parameters():
        torch.nn.init.normal_(p,mean=0.,std=.02)
    features = torch.randn(1,128,requires_grad=True)
    ids = (933*512+torch.arange(512))[None]
    p = (-(torch.arange(512)-255.3).square()/(2*6.3875**2)).softmax(-1)[None]
    logits = head.classes(features)
    expected = logits.logsumexp(-1).mean()-(logits.gather(-1,ids)*p).sum(-1).mean()
    expected.backward()
    reference_loss = float(expected.detach())
    reference_dx = features.grad.clone()
    reference_dw,reference_db = head.classes.weight.grad,head.classes.bias.grad
    head.zero_grad(set_to_none=True); features.grad = None
    del logits,expected
    actual = chunked_pair_linear_cross_entropy(features,head.classes.weight,head.classes.bias,ids,p)
    actual.backward()
    torch.testing.assert_close(actual,torch.tensor(reference_loss),rtol=3.e-6,atol=3.e-6)
    torch.testing.assert_close(features.grad,reference_dx,rtol=3.e-5,atol=2.e-6)
    dw_error = 0.
    for a,b in zip(head.classes.weight.grad.split(65536),reference_dw.split(65536)):
        assert torch.isfinite(a).all()
        dw_error = max(dw_error,float((a-b).abs().max()))
        torch.testing.assert_close(a,b,rtol=3.e-5,atol=1.e-6)
    torch.testing.assert_close(head.classes.bias.grad,reference_db,rtol=3.e-5,atol=1.e-6)
    result = dict(passed=True,vocabulary=vocabulary,independent_pair_rows=vocabulary,
        classifier_parameters=sum(x.numel() for x in head.parameters()),
        cpu_full_vocabulary_loss=reference_loss,cpu_weight_gradient_max_error=dw_error,
        exact_normalizer_over_all_pair_ids=True,sampled_negatives=False,
        training_run_launched=False,training_weights_modified=False)
    del reference_dw,reference_db,features,ids,p,actual
    head.zero_grad(set_to_none=True); gc.collect()
    device = torch.device('cuda',args.device)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_gib*2**30/torch.cuda.get_device_properties(device).total_memory,device)
    torch.backends.cuda.matmul.allow_tf32 = False
    head = head.to(device=device,dtype=torch.float16)
    # This stress-tests real all-class logits/loss/gradients. Half weights fit
    # alongside the protected running job; the resource audit separately
    # budgets real FP32 master weights, gradients and Adam states.
    rows = 256
    features = torch.randn(rows,128,device=device,dtype=torch.float16,requires_grad=True)
    atoms = torch.arange(rows,device=device)%16384
    ids = atoms[:,None]*512+torch.arange(512,device=device)
    p = (-(torch.arange(512,device=device)-255.3).square()/(2*6.3875**2)).softmax(-1).expand(rows,-1)
    torch.cuda.synchronize(); started = time.monotonic()
    loss = chunked_pair_linear_cross_entropy(features,head.classes.weight,head.classes.bias,ids,p)
    (loss*128).backward()
    torch.cuda.synchronize(); elapsed = time.monotonic()-started
    assert torch.isfinite(loss) and torch.isfinite(features.grad).all()
    for chunk in head.classes.weight.grad.split(65536):
        assert torch.isfinite(chunk).all()
    result.update(gpu_full_vocabulary_loss=float(loss.detach()),
        gpu_head_forward_backward_rows=rows,gpu_head_forward_backward_seconds=elapsed,
        gpu_weight_dtype='float16 for isolated memory-bounded preflight',
        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        full_optimizer_step_tested=False)
    head.zero_grad(set_to_none=True)
    del features,ids,p,loss
    torch.cuda.empty_cache()
    head.eval()
    times=[]; chosen=[]
    with torch.no_grad():
        x = torch.randn(1,1536,device=device,dtype=torch.float16)
        logits = head(x)
        assert logits.shape == (1,vocabulary)
        for _ in range(3):
            torch.cuda.synchronize(); started=time.monotonic()
            code=sample_from_logits(logits,temperature=1.,top_k=None,top_p=.95)
            torch.cuda.synchronize(); times.append(time.monotonic()-started)
            chosen.append(int(code))
    result.update(native_joint_nucleus_seconds_per_single_event=times,
        valid_sample_ids=all(0<=x<vocabulary for x in chosen),
        sparse_encoder_and_decoder_changed=False,
        shared_upstream_commit=rq_upstream.UPSTREAM_COMMIT,
        implementation_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in ['src/models/laser_pair_id_head.py','src/training/pair_id_linear_loss.py',
                         'src/training/laser_packed_integer_recipe.py']})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
