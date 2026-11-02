#!/usr/bin/env python3
"""Prepare and supervise the FID16.9268 epoch68 continuation on eight 96GB GPUs."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import uuid

import yaml

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('/tmp/laser-imagenet-epoch68-rtx6000-20261019')
RUN = 'imagenet-q995-epoch68-to100-8rtx6000-20261019'
OUT = ROOT/'outputs'/(RUN+'-execution')
PARENT = 'imagenet-rfid421-ffhq-q995-clipped-sigma42p65-scratch-4h200-20261009'
SOURCE = ROOT/'outputs'/PARENT
REFERENCE_SHA = '3f9c92d15755e76ec312964a819e3e19b9cf3aadc618a7ae99f5e8aa96501260'


def write(path, value):
    if path.is_relative_to(OUT):
        local=BASE/'metadata'/path.relative_to(OUT)
        local.parent.mkdir(parents=True,exist_ok=True)
        temporary=local.with_suffix('.tmp')
        temporary.write_text(json.dumps(value,indent=2,default=str)+'\n')
        temporary.replace(local)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,default=str)+'\n')
    temporary.replace(path)


def copy_atomic(source, target):
    temporary=target.with_name(target.name+'.'+uuid.uuid4().hex+'.tmp')
    shutil.copyfile(source,temporary)
    temporary.replace(target)


def prepare():
    import torch
    source = BASE/'inputs/source-epoch068.pt'
    assert json.loads(source.with_suffix('.verified.json').read_text())['verified']
    payload = torch.load(source,map_location='cpu',mmap=True,weights_only=False)
    assert payload['epoch'] == 68 and payload['global_step'] == 42568
    assert abs(payload['fid']-16.926755510053226)<1e-10
    assert {int(v['step']) for v in payload['optimizer']['state'].values()} == {42568}
    assert len(payload['optimizer']['state']) == 837
    with (BASE/'inputs/imagenet_256_train.npz').open('rb') as stream:
        assert hashlib.file_digest(stream,'sha256').hexdigest() == REFERENCE_SHA
    checkpoints = OUT/'train/checkpoints'
    checkpoints.mkdir(parents=True,exist_ok=True)
    for name in ['last.pt','resume-epoch068.pt',*[Path(p).name for key in ('best_fid','best_inception')
                                              for _,p in payload[key]]]:
        alias = checkpoints/name
        if not alias.exists():alias.symlink_to(source)
    frozen = BASE/'source/src/training'
    for name in ('fid_adaptive_schedule.py','portable_imagenet_recovery.py','rq_imagenet_runtime.py','checkpoint_upload_queue.py'):
        shutil.copyfile(ROOT/'src/training'/name,frozen/name)
    shutil.copyfile(ROOT/'scripts/tools/verified_wandb_checkpoint_upload.py',BASE/'verified_wandb_checkpoint_upload.py')
    # SciPy 1.18 removed sqrtm(disp=False); the maintained backend supports both
    # signatures and preserves the same upstream Inception/FID equations.
    shutil.copyfile(ROOT/'src/rqvae_metrics.py',BASE/'source/src/rqvae_metrics.py')
    shutil.copyfile(ROOT/'scripts/tools/imagenet_epoch68_runtime.py',BASE/'imagenet_epoch68_runtime.py')
    prior = SOURCE/'resume-best-fid-refine-4h200-20261012/chunked_checkpoint_io.py'
    shutil.copyfile(prior,BASE/'chunked_checkpoint_io.py')
    shutil.copyfile(ROOT/'outputs/imagenet-q995-epoch68-to100-4h200-20261019/reproducibility/sample_panels.py',
                    BASE/'sample_panels.py')
    entry = (BASE/'entry.py').read_text()
    for name in ('loss-balance-policy.json','coefficient-support-policy.json','original-fid-only-policy.json'):
        entry=entry.replace("OUT/'"+name+"'","BASE/'"+name+"'")
    if 'imagenet_epoch68_runtime' not in entry:
        old = "assert world == 4 and backend == 'ddp'"
        assert entry.count(old)==1
        entry = entry.replace(old,"assert world == int(os.environ['LASER_EXPECTED_WORLD_SIZE']) and backend == 'ddp'")
        old = "if __name__ == '__main__':"
        assert entry.count(old)==1
        entry = entry.replace(old,"from imagenet_epoch68_runtime import install\ninstall(globals())\n\n"+old)
    (BASE/'entry.py').write_text(entry)
    # Workers stay on the local JPEG archive and overlap CPU augmentation.
    trainer = frozen/'rqtransformer.py'
    text = trainer.read_text()
    text = text.replace('batch_sampler=sampler, num_workers=4,',
        'batch_sampler=sampler, num_workers=int(os.environ.get("LASER_DATA_WORKERS", "8")),')
    # The fixed preview partitions its labels across the whole machine.
    for old,new in (
        ('if rank() == 0:\n                    for setting in preview_sampling_settings(args):',
         'if True:  # distributed fixed-class preview\n                    for setting in preview_sampling_settings(args):'),
        ('if rank() == 0:\n                                devices = [device.index] if device.type == "cuda" else []',
         'if True:  # distributed fixed-class preview\n                                devices = [device.index] if device.type == "cuda" else []')):
        text=text.replace(old,new)
    # Metric reductions and checkpoint publication can take longer than a step.
    trainer.write_text(text)
    weights = BASE/'torch-cache/hub/checkpoints'
    weights.mkdir(parents=True,exist_ok=True)
    for name in ('weights-inception-2015-12-05-6726825d.pth','pt_inception-2015-12-05-6726825d.pth'):
        target = weights/name
        if not target.exists():target.symlink_to(BASE/'inputs/weights-inception-2015-12-05-6726825d.pth')
    for name in ('checkpoint-staging','checkpoint-upload-cache','chunk-cache','wandb'):
        (BASE/name).mkdir(exist_ok=True)
    config = yaml.safe_load((SOURCE/'train.yaml').read_text())
    config['options'].update(checkpoint=str(BASE/'inputs/stage1-tokenizer.pt'),
        data=str(BASE/'data'),output=str(OUT/'train'),checkpoint_dir=str(checkpoints),
        resume_checkpoint=str(checkpoints/'resume-epoch068.pt'),
        fid_reference_stats=str(BASE/'inputs/imagenet_256_train.npz'),
        batch_size=128,total_batch_size=2048,epochs=100,lr_schedule_epochs=100,min_lr=0.,
        fid_every=2,fid_batch_size=128,save_ckpt_freq=1,save_step_freq=250,
        sample_grid_every=626,sample_grid_size=512,sample_grid_samples_per_class=64,
        sample_grid_on_start=True,sample_grid_batch_size=32,
        wandb_id=RUN,wandb_name='ImageNet K4 | epoch68 FID16.9268 to100 | adaptive cosine to zero | 8 RTX PRO 6000',
        wandb_mode='online',upload_checkpoints=True,checkpoint_upload_mode='files',
        model_only_best_checkpoints=False,max_optimizer_steps=0)
    (BASE/'train.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    if not (OUT/'train.yaml').exists():copy_atomic(BASE/'train.yaml',OUT/'train.yaml')
    canary = json.loads(json.dumps(config))
    canary['options'].update(wandb_mode='disabled',upload_checkpoints=False,
        sample_grid_on_start=True,sample_grid_every=0,max_optimizer_steps=2)
    (BASE/'canary.yaml').write_text(yaml.safe_dump(canary,sort_keys=False))
    write(OUT/'checkpoint-verification.json',dict(passed=True,source=str(source),
        source_md5=json.loads(source.with_suffix('.verified.json').read_text())['md5'],
        epoch=68,step=42568,fid=payload['fid'],inception_score=payload['inception_score'],
        adam_parameters=837,adam_step=42568,reference_sha256=REFERENCE_SHA))
    write(OUT/'plan.json',dict(run=RUN,parent=PARENT,source_epoch=68,source_step=42568,
        source_fid=payload['fid'],target_epoch=100,target_step=62600,remaining_updates=20032,
        hardware=dict(gpus=8,name='RTX PRO 6000 Blackwell 96GB',microbatch=128,
            accumulation=2,global_batch=2048,precision='bf16',frozen_encoder='fp32'),
        schedule=dict(anchor_lr=payload['optimizer']['param_groups'][0]['lr'],minimum=0.,
            kind='remaining-interval cosine with FID plateau reductions',patience=3,
            factor=.5,min_delta=.05,cooldown=1),
        evaluation=dict(samples=50000,real_split='full_train',real_images=1281167,
            every_epochs=2,always_final=True),
        previews=dict(classes=['cheeseburger','teapot','samoyed','bald eagle','red fox','hotpot','wombat','timber wolf'],
            samples_per_class=64,overview_per_class=10,seed=261001,every_epochs=1),
        sampler='retained measured epoch68 ancestral nucleus; atom T.9/P.9; coefficient T1/P.85',
        checkpoints=dict(best_fid=1,best_inception=1,last=1,full_optimizer=True,upload='online W&B run files'),
        training_launched=False,time=time.time()))
    # Preserve the exact executed files, including local runtime patches.
    with tarfile.open(BASE/'runtime-source.tar.gz','w:gz') as archive:
        for name in ('entry.py','imagenet_epoch68_runtime.py','sample_panels.py','chunked_checkpoint_io.py','train.yaml',
                     'loss-balance-policy.json','coefficient-support-policy.json','original-fid-only-policy.json','source'):
            archive.add(BASE/name,arcname=name,filter=lambda m: None if '__pycache__' in m.name else m)
    copy_atomic(BASE/'runtime-source.tar.gz',OUT/('runtime-source-'+str(time.time_ns())+'.tar.gz'))
    print('Frozen runtime and launch configurations prepared.',flush=True)


def environment(mode):
    env = dict(os.environ,
        WANDB_API_KEY=Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip(),
        LASER_RUN_BASE=str(BASE),LASER_PERSISTENT_BASE=str(OUT),LASER_PHASE='production',
        LASER_EXPECTED_WORLD_SIZE='8',LASER_ACCUMULATION='2',
        LASER_FFHQ_DECODER_OBJECTIVE='1',LASER_SUB_BIN_TARGETS='0',
        LASER_COEFFICIENT_SIGMA_BINS=str(.125/(6/2047)),
        LASER_PARENT_RUN='helloimlixin-rutgers/laser/'+PARENT,LASER_ONLINE_RUN=RUN,
        LASER_REFERENCE_SHA256=REFERENCE_SHA,
        LASER_CHECKPOINT_STAGING_DIR=str(BASE/'checkpoint-staging'),
        LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(BASE/'checkpoint-upload-cache'),
        LASER_CHECKPOINT_IMMUTABLE_FILES='1',LASER_DATA_WORKERS='8',LASER_COMPILE_BLOCKS='1',
        LASER_LOSS_BALANCE_VERIFY_DIR=str(BASE/'verification'/mode),
        LASER_LAYOUT_MIGRATION_DIR=str(BASE/'verification'/mode/'layout'),
        CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
        PYTHONPATH=str(BASE)+':'+str(BASE/'source')+':'+str(BASE/'source/runtime'),
        TORCH_HOME=str(BASE/'torch-cache'),TORCHINDUCTOR_CACHE_DIR=str(BASE/'inductor-cache'),
        TORCHINDUCTOR_COMPILE_THREADS='4',WANDB_DIR=str(BASE/'wandb'),
        WANDB_CACHE_DIR=str(BASE/'wandb-cache'),WANDB_DATA_DIR=str(BASE/'wandb-data'),
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',NCCL_NVLS_ENABLE='0',PYTHONUNBUFFERED='1')
    if mode=='canary':env['LASER_LOSS_BALANCE_SMOKE_CHECKPOINT']=str(BASE/'canary-endpoint.pt')
    return env


def run(mode):
    lock=(BASE/'training.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    config = BASE/('canary.yaml' if mode=='canary' else 'train.yaml')
    ready = json.loads((BASE/'data/ready.json').read_text())
    assert ready['ready'] and ready['training_images']==1281167
    assert json.loads((BASE/'metadata/checkpoint-verification.json').read_text())['passed']
    if mode=='production':
        assert json.loads((BASE/'metadata/canary-verification.json').read_text())['passed']
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=8',
             str(BASE/'entry.py'),'--config',str(config)]
    log=BASE/(mode+'.log')
    for attempt in range(1 if mode=='canary' else 4):
        if mode=='production' and attempt:
            checkpoint=OUT/'train/checkpoints/last.pt'
            assert checkpoint.is_file()
            c=yaml.safe_load(config.read_text())
            c['options']['resume_checkpoint']=str(checkpoint)
            c['options']['sample_grid_on_start']=False
            config.write_text(yaml.safe_dump(c,sort_keys=False))
        with log.open('a') as stream:
            child=subprocess.Popen(command,cwd=BASE,env=environment(mode),stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT)
            write(OUT/(mode+'-status.json'),dict(state='running',supervisor_pid=os.getpid(),
                child_pid=child.pid,attempt=attempt+1,log=str(log),target_epoch=100,time=time.time()))
            code=child.wait()
        if code==0:
            if mode=='canary':
                import torch
                endpoint=torch.load(BASE/'canary-endpoint.pt',map_location='cpu',mmap=True,weights_only=False)
                assert endpoint['global_step']==42570 and endpoint['scheduler']['last_epoch']==42570
                assert {int(v['step']) for v in endpoint['optimizer']['state'].values()}=={42570}
                proof=[json.loads((BASE/'verification/canary'/f'progress-rank{i}.json').read_text()) for i in range(8)]
                assert all(p['passed'] and p['step']==42570 for p in proof)
                write(OUT/'canary-verification.json',dict(passed=True,updates=2,discarded=True,
                    step=42570,all_ranks=proof,scheduler=endpoint['scheduler'],time=time.time()))
            else:
                sys.path[:0]=[str(BASE),str(BASE/'source')]
                import chunked_checkpoint_io
                chunked_checkpoint_io.LOCAL=BASE/'chunk-cache'
                checkpoint=chunked_checkpoint_io.recover(OUT/'train/checkpoints/last.pt',lambda p:p)
                import torch
                endpoint=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
                assert endpoint['epoch']==100 and endpoint['global_step']==62600
                assert all(g['lr']==0. for g in endpoint['optimizer']['param_groups'])
                write(OUT/'completed.json',dict(passed=True,epoch=100,step=62600,lr=0.,time=time.time()))
            write(OUT/(mode+'-status.json'),dict(state='completed',returncode=0,time=time.time()))
            return
        write(OUT/(mode+'-status.json'),dict(state='failed',returncode=code,attempt=attempt+1,time=time.time()))
    raise SystemExit(code)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','canary','production'])
    args=parser.parse_args()
    if args.mode=='prepare':prepare()
    else:run(args.mode)
