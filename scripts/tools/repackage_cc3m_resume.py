"""Deploy checkpoint-upload fixes while preserving an existing full trajectory."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
from types import SimpleNamespace

from omegaconf import OmegaConf
import torch

from scripts.tools.prepare_cc3m_best_resume import assert_identical, digest
from src.training.cc3m_text import (config_digest, create_lr_scheduler,
    verify_checkpoint_progress, verify_resume_config)
from src.training import rqtransformer as rq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--source-local', type=Path, required=True)
    parser.add_argument('--local', type=Path, required=True)
    parser.add_argument('--archive-name', default='original-best-upload-fix-source-20261005')
    parser.add_argument('--report-name', default='original-best-upload-fix.json')
    parser.add_argument('--preserve-best-files', action='store_true',
                        help='Keep already resumable best files unchanged during a runtime-only repair')
    args = parser.parse_args()
    base, previous, local = args.base.resolve(), args.source_local.resolve(), args.local.resolve()
    root = Path(__file__).resolve().parents[2]
    source = torch.load(previous/'checkpoints/last.pt',map_location='cpu',weights_only=True,mmap=True)
    saved = source['config']
    updates = saved['train_items']//saved['total_batch_size']
    verify_checkpoint_progress(source,updates,saved['accumulation'],8)
    assert not local.exists()
    runtime = local/'runtime';runtime.mkdir(parents=True)
    with tarfile.open(base/'runtime.tar.gz') as archive:
        archive.extractall(runtime,filter='data')
    manifest = dict(saved['runtime_sha256'])
    for name, expected in manifest.items():
        assert digest(runtime/name)==expected,name
    for name in ['src/training/cc3m_text.py','src/training/checkpoint_upload_queue.py','scripts/tools/prepare_cc3m_best_resume.py',
                 'scripts/tools/repackage_cc3m_resume.py']:
        dest=runtime/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(root/name,dest);manifest[name]=digest(dest)
    subprocess.run([sys.executable,'-c','import src.training.cc3m_text; import train'],
        cwd=runtime,env=dict(os.environ,PYTHONPATH=f'{runtime/"runtime"}:{runtime}'),check=True)
    (local/'assets').mkdir()
    for name in ['stage1.pt','train.pt','validation.pt']:
        (local/'assets'/name).hardlink_to(previous/'assets'/name)
    for name in ['torch','inductor','triton']:
        (local/name).symlink_to(previous/name,target_is_directory=True)
    target=copy.deepcopy(saved)
    target.update(runtime_sha256=manifest,checkpoint_on_resume=False,resume_checkpoint=None,
        checkpoint=str(local/'assets/stage1.pt'),token_cache=str(local/'assets/train.pt'),
        validation_cache=str(local/'assets/validation.pt'),local_checkpoints=str(local/'checkpoints'),
        runtime_migration_previous_manifest_sha256=config_digest(saved['runtime_sha256']))
    assert target['fid_lr_policy']==saved['fid_lr_policy']
    (local/'checkpoints').mkdir()
    groups={};records=[]
    for slot in ['last.pt','best-fid.pt','best-clip.pt']:
        path=previous/'checkpoints'/slot
        identity=(path.stat().st_dev,path.stat().st_ino)
        if identity in groups:
            (local/'checkpoints'/slot).hardlink_to(groups[identity])
            records.append(dict(slot=slot,shared_with=groups[identity].name));continue
        state=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        verify_checkpoint_progress(state,updates,saved['accumulation'],8)
        verify_resume_config(state['config'],target)
        optimizer=SimpleNamespace(param_groups=copy.deepcopy(state['optimizer']['param_groups']))
        scheduler=create_lr_scheduler(optimizer,target,updates,state['global_step'],
            state['scheduler'],state['config'])
        assert_identical(scheduler.state_dict(),state['scheduler'])
        assert_identical(optimizer.param_groups,state['optimizer']['param_groups'])
        destination=local/'checkpoints'/slot
        if args.preserve_best_files and slot != 'last.pt':
            destination.hardlink_to(path)
            payload=state
        else:
            payload=dict(state,config=target)
            rq.atomic_torch_save(payload,destination)
        restored=torch.load(destination,map_location='cpu',weights_only=True,mmap=True)
        verify_checkpoint_progress(restored,updates,saved['accumulation'],8)
        verify_resume_config(restored['config'],target)
        for field in ['model','optimizer','scheduler','rng_state_by_rank','epoch',
                      'next_microbatch','global_step','metrics','best_fid','best_clip']:
            assert_identical(state[field],restored[field])
        groups[identity]=destination
        records.append(dict(slot=slot,step=state['global_step'],epoch=state['epoch'],
            bytes=destination.stat().st_size,md5=digest(destination,'md5'),
            all_training_state_exact=True))
        print(json.dumps(records[-1]),flush=True)
        del restored,payload,state
    archive=base/args.archive_name;archive.mkdir()
    for name in ['recipe.yaml','runtime.tar.gz','runtime-manifest.json','resume.py']:
        shutil.copyfile(base/name,archive/name)
    cfg=OmegaConf.load(base/'recipe.yaml');cfg.options=target
    OmegaConf.save(cfg,base/'recipe.yaml')
    OmegaConf.save(cfg,root/'configs/stage2'/(target['wandb_id']+'.yaml'))
    with tarfile.open(base/'runtime.tar.gz','w:gz') as archive_out:
        for name in manifest:archive_out.add(runtime/name,arcname=name)
    (base/'runtime-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    report=dict(timestamp=time.time(),source_step=source['global_step'],
        source_epoch=source['epoch'],lr_before=source['optimizer']['param_groups'][0]['lr'],
        lr_after=source['optimizer']['param_groups'][0]['lr'],
        lr_policy_unchanged=True,all_training_state_exact=True,checkpoints=records,
        local=str(local),fixes=['recovered best-only states do not replace the active last slot'])
    (base/args.report_name).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
