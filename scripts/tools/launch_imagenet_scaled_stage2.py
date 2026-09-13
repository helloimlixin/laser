#!/usr/bin/env python3
"""Launch the verified ImageNet cache-to-training workflow on four H200 GPUs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/imagenet-scaled-rq-stage2-20260913'
RUN_ID='imagenet-rfid421-scaled-rq8-480m-20260913'


def sha256(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    global BASE,RUN_ID
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--base',type=Path,default=BASE)
    parser.add_argument('--run-id',default=RUN_ID)
    parser.add_argument('--verification',type=Path)
    parser.add_argument('--reuse-cache',type=Path)
    parser.add_argument('--max-rfid-drift',type=float,default=.1)
    parser.add_argument('--sample-grid-on-start',action='store_true')
    parser.add_argument('--sampler')
    args=parser.parse_args()
    BASE=args.base.resolve();RUN_ID=args.run_id
    proof=json.loads((args.verification or BASE/'preflight-verified/verification.json').read_text())
    assert proof['preflight_passed'] and proof['strict_checkpoint_reload']
    assert proof['saved_tensors_finite'] and proof['class_conditioning_verified']
    assert proof['released_sampler_decode_passed'] and proof['tokenizer_frozen']
    assert proof['world_size']==4
    assert proof['effective_batch_size']==2048 and proof['optimizer_updates']>=3
    assert json.loads((BASE/'cache-smoke-verification.json').read_text())['passed']
    assert (BASE/'assets/best_rfid_slot1_model.pt').is_file()
    assert sha256(BASE/'assets/best_rfid_slot1_model.pt')==proof['tokenizer_sha256']
    assert sha256(BASE/'scaled-atom-codebook.pt')==proof['codebook_sha256']
    import sys
    sys.path.insert(0,str(ROOT))
    from src.tokenizer_fidelity import require_tokenizer_fidelity
    from src.scaled_atom_sampling import SAMPLER_SETTINGS
    quality=require_tokenizer_fidelity(BASE/'fidelity-gate.json',proof['tokenizer_sha256'],proof['codebook_sha256'],args.max_rfid_drift)
    receipt_path=BASE/'launch.json'
    previous={}
    if receipt_path.exists():
        previous=json.loads(receipt_path.read_text())
        proc=Path(f'/proc/{previous["torchrun_pid"]}/cmdline')
        if proc.exists() and str(BASE).encode() in proc.read_bytes():
            raise RuntimeError('The ImageNet workflow is already active')
        assert args.resume,'A previous launch exists; use --resume after inspecting its status'
    else:
        assert not (BASE/'train').exists(),'Unexpected existing production directory'
    sampler=args.sampler or previous.get('generation_sampler','original')
    assert sampler in SAMPLER_SETTINGS,f'Unknown sampler: {sampler}'
    hashes=proof['source_hashes']
    if not args.resume:
        for name,expected in hashes.items():
            assert sha256(ROOT/name)==expected,f'Verified source changed: {name}'
    snapshot=BASE/'source-snapshot'
    if not args.resume:
        for name in hashes:
            dest=snapshot/name
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/name,dest)
        # A namespace package has no __init__; preserve one if the project adds it.
        init=ROOT/'src/__init__.py'
        if init.exists():
            shutil.copy2(init,snapshot/'src/__init__.py')
        extra=[Path(__file__),ROOT/'scripts/tools/prepare_imagenet_scaled_stage2.py',
               ROOT/'scripts/tools/fit_imagenet_rq8_levels.py',
               ROOT/'scripts/tools/screen_imagenet_tokenizer_fidelity.py',
               ROOT/'scripts/tools/write_imagenet_fidelity_gate.py',
               ROOT/'tests/test_imagenet_scaled_stage2.py',ROOT/'tests/test_tokenizer_fidelity.py']
        for source in extra:
            name=str(source.relative_to(ROOT))
            hashes[name]=sha256(source)
            dest=snapshot/name
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,dest)
        (BASE/'source-manifest.json').write_text(json.dumps(hashes,indent=2)+'\n')
    else:
        hashes=json.loads((BASE/'source-manifest.json').read_text())
    for name,expected in hashes.items():
        assert sha256(snapshot/name)==expected,f'Snapshot changed: {name}'
    python=ROOT/'.venv-imagenet-stage2/bin/python'
    command=[str(python),'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
        str(snapshot/'scripts/tools/train_imagenet_scaled_stage2.py'),
        '--preparation',str(BASE),'--checkpoint',str(BASE/'assets/best_rfid_slot1_model.pt'),
        '--codebook',str(BASE/'scaled-atom-codebook.pt'),'--cache',str(BASE/'cache'),
        '--output',str(BASE/'train'),'--run-id',RUN_ID,'--batch-size',str(proof['microbatch_per_gpu']),
        '--cache-batch-size','64','--views','2','--workers','8',
        '--levels',str(proof['coefficient_levels']),'--fidelity-report',str(BASE/'fidelity-gate.json'),
        '--max-rfid-drift',str(args.max_rfid_drift),'--sampler',sampler]
    if args.reuse_cache:
        command.extend(['--reuse-cache',str(args.reuse_cache.resolve())])
    if args.resume:
        command.append('--resume')
    if args.sample_grid_on_start:
        command.append('--sample-grid-on-start')
    env={**os.environ,'LASER_PROJECT_ROOT':str(ROOT),'CUDA_VISIBLE_DEVICES':'0,1,2,3',
         'NCCL_NVLS_ENABLE':'0','OMP_NUM_THREADS':'8','OPENBLAS_NUM_THREADS':'8','MKL_NUM_THREADS':'8',
         'TORCH_HOME':'/workspace/tmp/official-rqvae-eval-cache','PYTHONUNBUFFERED':'1'}
    for name in ('WANDB_SERVICE','_WANDB_SERVICE'):
        env.pop(name,None)
    # Authentication uses the user-provided credential already installed in ~/.netrc.
    with (BASE/'production.log').open('ab') as log:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt=dict(torchrun_pid=process.pid,command=command,run_id=RUN_ID,
        url=f'https://wandb.ai/helloimlixin-rutgers/laser/runs/{RUN_ID}',started_unix=time.time(),
        production_from_scratch=not args.resume,preflight_weights_loaded=False,
        source_files_verified=len(hashes),workflow='full validation cache and rFID audit; two-view training cache; 100-epoch class-conditional RQ training',
        resume=args.resume,expected_initial_weights_sha256=proof['initial_weights_sha256'],
        generation_sampler=sampler,
        matched_rfid_gate=quality)
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':
    main()
