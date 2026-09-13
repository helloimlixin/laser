#!/usr/bin/env python3
"""Bind a matched full-validation result to the exact selected RQ codebook."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.original_rq_training import atomic_json,file_sha256
from src.tokenizer_fidelity import require_tokenizer_fidelity


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--codebook',type=Path,required=True)
    p.add_argument('--variant',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-rfid-drift',type=float,default=.1)
    args=p.parse_args()
    evaluation=json.loads((args.evaluation/'results.json').read_text())
    assert evaluation['images']==50000 and evaluation['full_validation']
    assert evaluation['validation_indices']==list(range(50000))
    codebook=torch.load(args.codebook,weights_only=True,map_location='cpu')
    candidate_path=args.evaluation/'candidate-codebooks.pt'
    source_levels_path=candidate_path if candidate_path.exists() else args.evaluation/'coefficient-levels.pt'
    tested=torch.load(source_levels_path,weights_only=True,map_location='cpu')
    assert codebook['checkpoint_sha256']==evaluation['checkpoint_sha256']==tested['checkpoint_sha256']
    torch.testing.assert_close(codebook['dictionary'],tested['dictionary'],rtol=0,atol=0)
    if 'candidates' in tested and args.variant in tested['candidates']:
        expected=tested['candidates'][args.variant]
        n=len(expected)
    elif args.variant=='rq8':
        n=8
        original=torch.load(ROOT/'outputs/imagenet-scaled-rq-stage2-20260913/scaled-atom-codebook.pt',
                            weights_only=True,map_location='cpu')
        expected=original['levels']['8']
    else:
        assert args.variant.startswith('rq')
        n=int(args.variant[2:])
        expected=tested['levels']['mp'][str(n)]
    torch.testing.assert_close(codebook['levels'][str(n)],expected,rtol=0,atol=0)
    reference=Path(evaluation['reference']).resolve()
    stats=np.load(reference)
    assert int(stats['images'])==50000 and np.array_equal(stats['validation_indices'],np.arange(50000))
    report=dict(checkpoint_sha256=evaluation['checkpoint_sha256'],codebook_sha256=file_sha256(args.codebook),
        images=50000,same_validation_images=True,same_reference_statistics=True,
        original_rfid=evaluation['results']['omp4']['rfid'],converted_rfid=evaluation['results'][args.variant]['rfid'],
        source_reported_rfid=4.210914134979248,reference_statistics=str(reference),
        reference_statistics_sha256=file_sha256(reference),variant=args.variant,
        validation_indices_sha256=file_sha256(args.evaluation/'results.json'),
        evaluation=str(args.evaluation.resolve()),source_levels_sha256=file_sha256(source_levels_path))
    atomic_json(args.output,report)
    approved=require_tokenizer_fidelity(args.output,report['checkpoint_sha256'],report['codebook_sha256'],args.max_rfid_drift)
    atomic_json(args.output,approved)
    print(json.dumps(approved,indent=2))


if __name__=='__main__':main()
