#!/usr/bin/env python3
"""Verify immutable recovery, resume the native RQ core, and preserve the queue."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import zipfile

ROOT=Path('/workspace/Projects/laser')
BASE=Path('/tmp/laser-fid14-continue100-20261016')
OUT=ROOT/'outputs/imagenet-fid14-continue100-4h100-20261016'
SOURCE=BASE/'chunk-cache/baae6b09483c493287a64e64b7857781c6563f483fa9f102eaaf99be3a98fa52.pt'


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,default=str)+'\n');temporary.replace(path)


def digest(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def main():
    import yaml
    import torch
    from launch_fid14_to100_then_support_order import environment
    lock=(BASE/'native-sequence.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert json.loads((OUT/'native-core-verification.json').read_text())['passed']
    plan=json.loads((OUT/'prepared.json').read_text())
    child=None;cancelled=False
    def status(phase,**values):
        write(OUT/'launch-status.json',dict(phase=phase,pid=os.getpid(),native_upstream_core=True,
            target_epoch=100,target_step=62600,time=time.time(),**values))
    def stop(*_):
        nonlocal cancelled
        cancelled=True
        if child is not None and child.poll() is None:child.send_signal(signal.SIGINT)
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    def launch(script,phase,env,args):
        nonlocal child
        with (BASE/(phase+'.log')).open('a') as log:
            child=subprocess.Popen([sys.executable,'-m','torch.distributed.run','--standalone',
                '--nproc-per-node=4',str(script),*args],cwd=BASE,env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT)
            status(phase,child_pid=child.pid,log=str(BASE/(phase+'.log')))
            code=child.wait()
        if cancelled:
            status('cancelled',returncode=code);raise SystemExit(code or 130)
        return code
    config=yaml.safe_load((BASE/'train.yaml').read_text())
    options=config['options']
    options.update(resume_checkpoint=str(SOURCE),checkpoint_dir=str(OUT/'native-train/checkpoints'),
        wandb_name='Historical LASER FID14.87 | original RQ core | continue to100 | 4H100',
        output=str(BASE/'native-production/train'))
    (BASE/'native-train.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    smoke=yaml.safe_load(yaml.safe_dump(config))
    smoke['options'].update(max_optimizer_steps=2,wandb_mode='disabled',fid_every=0,
        checkpoint_dir=str(OUT/'native-smoke/checkpoints'),output=str(BASE/'native-smoke/train'))
    (BASE/'native-smoke.yaml').write_text(yaml.safe_dump(smoke,sort_keys=False))
    env=environment('native-smoke');env.update(LASER_BENCHMARK='1',LASER_BENCHMARK_SAVE='1')
    code=launch(BASE/'entry-native.py','native_smoke',env,['--config',str(BASE/'native-smoke.yaml')])
    if code:status('failed',failed_phase='native_smoke',returncode=code);raise SystemExit(code)
    receipts=[json.loads(p.read_text()) for p in (OUT/'native-smoke/checkpoints/.checkpoint-data').glob('*.json')]
    assert len(receipts)==1 and receipts[0]['global_step']==48830
    receipt=receipts[0]
    status('verifying_native_checkpoint_persistence',checkpoint=receipt['persistent'])
    assert digest(receipt['persistent'])==receipt['sha256']
    saved=torch.load(receipt['local'],map_location='cpu',mmap=True,weights_only=False)
    assert saved['global_step']==48830 and len(saved['optimizer']['state'])==782
    assert {int(v['step']) for v in saved['optimizer']['state'].values()}=={14400}
    assert saved['execution_core']['implementation']=='original rqvae.models.rqtransformer'
    del saved
    for rank in range(4):
        initial=json.loads((OUT/f'verification/native-smoke/startup-rank{rank}.json').read_text())
        after=json.loads((OUT/f'verification/native-smoke/step1-rank{rank}.json').read_text())
        core=json.loads((OUT/f'verification/native-smoke/native-core-rank{rank}.json').read_text())
        assert initial['adam_step']==14398 and initial['optimizer_parameters']==782
        assert after['finite'] and after['adam_step']==14399 and core['no_upstream_core_edits']
        assert json.loads((OUT/f'verification/native-smoke/resume-rng-rank{rank}.json').read_text())['restored_exactly']
    write(OUT/'native-smoke-verification.json',dict(passed=True,updates_discarded=2,
        original_adam_age=14398,after_disposable_updates=14400,world_size=4,
        all_rank_native_core_verified=True,full_persistent_checkpoint_sha256_verified=True,
        serialization_roundtrip_verified=True,source_untouched=True))
    started=time.monotonic()
    while not (OUT/'native-source-import.json').exists():
        if cancelled:raise SystemExit(130)
        assert time.monotonic()-started<1800,'Epoch78 persistent import has not finished'
        status('waiting_for_verified_native_source_import');time.sleep(5)
    source=json.loads((OUT/'native-source-import.json').read_text());assert source['passed']
    files=[BASE/'entry-native.py',BASE/'native-train.yaml',BASE/'native-smoke.yaml']
    files.extend(BASE/'support'/name for name in ['fid14_continuation_runtime.py','fid14_native_checkpoint_io.py',
        'fid14_native_rq_runtime.py','native_physical_pair_adapter.py','refinement_schedule.py'])
    manifest={str(p.relative_to(BASE)):digest(p) for p in files}
    write(OUT/'reproducibility/native-code-manifest.json',manifest)
    with zipfile.ZipFile(OUT/'reproducibility/native-runtime.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in files:archive.write(p,p.relative_to(BASE))
        archive.write(Path(__file__),'controller.py')
    for attempt in range(3):
        env=environment('native-production')
        code=launch(BASE/'entry-native.py','native_training_to_epoch100',env,['--config',str(BASE/'native-train.yaml')])
        if code==0:break
        receipts=[json.loads(p.read_text()) for p in (OUT/'native-train/checkpoints/.checkpoint-data').glob('*.json')]
        candidates=[r for r in receipts if Path(r['local']).is_file() and Path(r['persistent']).is_file()]
        latest=max(candidates,key=lambda r:r['global_step']) if candidates else dict(local=str(SOURCE),global_step=48828)
        if attempt==2:status('failed',failed_phase='native_training_to_epoch100',returncode=code);raise SystemExit(code)
        options['resume_checkpoint']=latest['local']
        (BASE/'native-train.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
        status('retrying_native_training',attempt=attempt+2,resume_step=latest['global_step'])
    final=json.loads((OUT/'trial-checkpoint-ready.json').read_text())
    assert final['global_step']==62600 and final['epoch']==100 and not final['stopping'],final
    write(OUT/'completed-to100.json',dict(passed=True,global_step=62600,epoch=100,
        native_upstream_core=True,time=time.time()))
    for order in ('interleaved','support_first'):
        item=plan['next_trials'][order];base=Path(item['base'])
        for relative,expected in item['code_manifest'].items():assert digest(base/relative)==expected,relative
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        env=dict(environment(order),LASER_RQ_BASE=str(base),TORCH_HOME=str(base/'torch-cache'),
            WANDB_DIR=str(base/'wandb'),LASER_CHECKPOINT_STAGING_DIR=str(base/'checkpoint-staging'),
            LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(base/'checkpoint-cache'))
        code=launch(base/'train.py','new_decoder_'+order,env,[])
        if code:status('failed',failed_phase='new_decoder_'+order,returncode=code);raise SystemExit(code)
        done=json.loads((ROOT/'outputs'/item['run']/'completed.json').read_text())
        assert done['epochs']==2 and done['global_step']==1252,done
    hashes=[json.loads((Path(plan['next_trials'][order]['base'])/'initial-weight-hash.json').read_text())['sha256']
        for order in ('interleaved','support_first')]
    assert len(set(hashes))==1,hashes
    status('completed',historical_epoch=100,new_trials_epochs_per_arm=2,matched_initial_sha256=hashes[0])


if __name__=='__main__':main()
