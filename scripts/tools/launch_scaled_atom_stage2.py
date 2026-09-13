#!/usr/bin/env python3
"""Launch the verified fresh eight-level prior on both H200s."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from relaunch_original_church_rq import current_conversation_credential

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/church-scaled-atom-stage2-20260913'


def main():
    proof=json.loads((BASE/'verification.json').read_text())
    assert proof['preflight_passed'] and proof['strict_checkpoint_reload']
    assert proof['released_sampler_decode_smoke_passed']
    assert proof['microbatch_per_gpu']==128 and proof['peak_gpu_allocated_gib']<120
    assert not (BASE/'train').exists() and not (BASE/'launch.json').exists()
    hashes=dict(proof['source_hashes'])
    upstream=ROOT/'outputs/church-rq-baseline-scratch-20260912/upstream-source'
    previous=json.loads((ROOT/'outputs/church-rq-baseline-scratch-20260912/verification.json').read_text())
    for name,expected in previous['source_hashes'].items():
        if str(upstream.relative_to(ROOT)) in name:
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
            hashes[name]=expected
    hashes[str(Path(__file__).relative_to(ROOT))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for name,digest in hashes.items():
        source=ROOT/name
        assert hashlib.sha256(source.read_bytes()).hexdigest()==digest,name
        dest=BASE/'source-snapshot'/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,dest)
    (BASE/'source-manifest.json').write_text(json.dumps(hashes,indent=2)+'\n')
    credential=current_conversation_credential()
    run_id='church-scaled-rq8-scratch-20260913'
    command=['/tmp/laser-sign-venv/bin/python','-m','torch.distributed.run','--standalone',
        '--nproc_per_node=2',str(ROOT/'scripts/tools/train_scaled_atom_stage2.py'),
        '--cache',str(BASE/'cache'),'--calibration',str(BASE/'temperature-calibration.json'),
        '--output',str(BASE/'train'),'--run-id',run_id,'--batch-size','128']
    env={**os.environ,'WANDB_API_KEY':credential,'CUDA_VISIBLE_DEVICES':'0,1',
         'OMP_NUM_THREADS':'8','OPENBLAS_NUM_THREADS':'8','MKL_NUM_THREADS':'8',
         'TORCH_HOME':'/workspace/tmp/official-rqvae-eval-cache'}
    for name in ('WANDB_SERVICE','_WANDB_SERVICE'):
        env.pop(name,None)
    with (BASE/'production.log').open('ab') as log:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt=dict(torchrun_pid=child.pid,command=command,run_id=run_id,started_unix=time.time(),
        expected_initial_weights_sha256=proof['model_initialization_sha256'],
        source_files_verified=len(hashes),production_from_scratch=True,
        preflight_checkpoint_used_for_training=False)
    (BASE/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    main()
