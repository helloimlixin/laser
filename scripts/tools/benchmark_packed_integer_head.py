#!/usr/bin/env python3
"""Tune exact softmax tile sizes with all 8,388,608 pair classes present."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    import torch
    from src.training.pair_id_linear_loss import chunked_pair_linear_cross_entropy
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rows',type=int,default=256)
    parser.add_argument('--memory-gib',type=float,default=6.)
    parser.add_argument('--wait-for-comparison',type=Path)
    args=parser.parse_args()
    while args.wait_for_comparison:
        status=args.wait_for_comparison/'status.json'
        if status.exists() and json.loads(status.read_text())['phase']=='completed':break
        time.sleep(2)
    torch.set_num_threads(4);torch.cuda.set_device(0)
    if args.wait_for_comparison:
        while torch.cuda.mem_get_info()[0] < (args.memory_gib+2)*2**30:time.sleep(2)
    torch.cuda.set_per_process_memory_fraction(args.memory_gib*2**30/torch.cuda.get_device_properties(0).total_memory,0)
    torch.manual_seed(261017513)
    weight=torch.empty(8388608,128,device='cuda',dtype=torch.float16).normal_(std=.02).requires_grad_()
    bias=torch.zeros(8388608,device='cuda',dtype=torch.float16,requires_grad=True)
    x=torch.randn(args.rows,128,device='cuda',dtype=torch.float16,requires_grad=True)
    ids=((torch.arange(args.rows,device='cuda')%16384)[:,None]*512+torch.arange(512,device='cuda'))
    p=(-(torch.arange(512,device='cuda')-255.3).square()/(2*6.3875**2)).softmax(-1).expand(args.rows,-1)
    results=[]
    configurations=([(128,65536),(128,262144),(128,1048576),(256,262144)] if args.rows<=256
        else [(128,262144),(512,262144),(1024,262144),(512,524288)])
    for rows,classes in configurations:
        times=[];losses=[]
        torch.cuda.reset_peak_memory_stats()
        try:
            for _ in range(3):
                weight.grad=bias.grad=x.grad=None
                torch.cuda.synchronize();started=time.monotonic()
                loss=chunked_pair_linear_cross_entropy(x,weight,bias,ids,p,rows,classes)
                (loss*128).backward()
                torch.cuda.synchronize();times.append(time.monotonic()-started)
                losses.append(float(loss.detach()))
                assert torch.isfinite(x.grad).all()
        except torch.OutOfMemoryError:
            weight.grad=bias.grad=x.grad=None
            torch.cuda.empty_cache()
            results.append(dict(row_chunk=rows,class_chunk=classes,
                exceeded_isolated_memory_budget=True))
            break
        row=dict(row_chunk=rows,class_chunk=classes,seconds=times,losses=losses,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            loss_difference_from_first=0. if not results else losses[-1]-results[0]['losses'][-1])
        assert abs(row['loss_difference_from_first'])<1.e-4
        results.append(row);print(json.dumps(row),flush=True)
        weight.grad=bias.grad=x.grad=None
        torch.cuda.empty_cache()
    winner=min([r for r in results if 'seconds' in r],key=lambda r:sorted(r['seconds'])[1])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(vocabulary=8388608,rows=args.rows,
        exact_global_normalizer=True,all_classes_retained=True,
        half_weight_head_only_benchmark=True,training_unchanged=True,
        results=results,selected=winner),indent=2)+'\n')


if __name__=='__main__':main()
