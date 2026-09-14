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
    parser.add_argument('--online-images',action=argparse.BooleanOptionalAction,default=None)
    parser.add_argument('--refresh-verified-source',action='store_true',
        help='Resume a paused run with newly preflight-verified source, preserving its prior snapshot')
    args=parser.parse_args()
    BASE=args.base.resolve();RUN_ID=args.run_id
    receipt_path=BASE/'launch.json'
    previous=json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    verification=(args.verification or Path(previous.get('verification',BASE/'preflight-verified/verification.json'))).resolve()
    proof=json.loads(verification.read_text())
    assert not args.refresh_verified_source or (args.resume and args.verification)
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
    if receipt_path.exists():
        proc=Path(f'/proc/{previous["torchrun_pid"]}/cmdline')
        if proc.exists() and str(BASE).encode() in proc.read_bytes():
            raise RuntimeError('The ImageNet workflow is already active')
        assert args.resume,'A previous launch exists; use --resume after inspecting its status'
    else:
        assert not (BASE/'train').exists(),'Unexpected existing production directory'
    sampler=args.sampler or previous.get('generation_sampler','original')
    assert sampler in SAMPLER_SETTINGS,f'Unknown sampler: {sampler}'
    online_images=(args.online_images if args.online_images is not None else
        proof.get('training_data_mode')=='online-images' if args.refresh_verified_source or not args.resume else
        previous.get('training_data_mode')=='online-images')
    if args.refresh_verified_source:
        assert proof.get('training_data_mode')==('online-images' if online_images else 'cached-latents')
        assert proof.get('generation_sampler')==sampler
        assert json.loads((BASE/'train/status.json').read_text())['phase']=='paused'
    hashes=proof['source_hashes']
    if not args.resume or args.refresh_verified_source:
        for name,expected in hashes.items():
            assert sha256(ROOT/name)==expected,f'Verified source changed: {name}'
    snapshot=BASE/'source-snapshot'
    revision_archive=None
    if args.refresh_verified_source:
        revision_archive=BASE/'source-revisions'/f'before-{int(time.time())}'
        revision_archive.mkdir(parents=True,exist_ok=False)
        old_hashes=json.loads((BASE/'source-manifest.json').read_text())
        for name,expected in old_hashes.items():
            assert sha256(snapshot/name)==expected,f'Previous snapshot changed: {name}'
        for name in ('launch.json','source-manifest.json','train/config.yaml','train/initialization.json','train/status.json'):
            dest=revision_archive/name
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(BASE/name,dest)
        os.link(BASE/'train/last.pt',revision_archive/'last.pt')
        snapshot.rename(revision_archive/'source-snapshot')
    if not args.resume or args.refresh_verified_source:
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
        if proof.get('coefficient_construction') in ('atom-specific','depth-and-atom-specific'):
            extra.extend([ROOT/'scripts/tools/prepare_imagenet_compact_rq.py',
                ROOT/'scripts/tools/assemble_imagenet_compact_run.py',
                ROOT/'scripts/tools/switch_imagenet_compact_rq.py',
                ROOT/'scripts/tools/calibrate_scaled_atom_temperature.py',
                ROOT/'tests/test_adaptive_scaled_atom_rq.py',ROOT/'tests/test_compact_rq_training.py',
                ROOT/'tests/test_scaled_atom_sampling.py'])
            extra.extend([ROOT/'scripts/tools/fit_depth_compact_rq.py',ROOT/'tests/test_depth_compact_rq.py',
                ROOT/'scripts/tools/audit_imagenet_rq_recipe.py'])
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
    if proof.get('coefficient_construction') in ('atom-specific','depth-and-atom-specific'):
        command.append('--compact')
    if args.resume:
        command.append('--resume')
    if args.sample_grid_on_start or args.refresh_verified_source:
        command.append('--sample-grid-on-start')
    if online_images:
        command.append('--online-images')
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
        source_files_verified=len(hashes),workflow=('validated frozen tokenizer; 100-epoch class-conditional RQ training; '
            + ('fresh ImageNet augmentations' if online_images else 'two-view training cache')),
        resume=args.resume,expected_initial_weights_sha256=proof['initial_weights_sha256'],
        generation_sampler=sampler,
        training_data_mode='online-images' if online_images else 'cached-latents',
        verification=str(verification),refreshed_verified_source=args.refresh_verified_source,
        previous_source_archive=str(revision_archive) if revision_archive else None,
        matched_rfid_gate=quality)
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':
    main()
