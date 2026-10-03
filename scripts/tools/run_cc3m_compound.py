"""Supervise CC3M cache preparation and launch the validated online stage-2 run."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.tools.build_cc3m_compound_cache import STAGE1_SHA, write_json


def alive(pid):
    try:
        os.kill(pid,0)
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except (ProcessLookupError,FileNotFoundError):return False


def merge_cache(base,options,*,expected_shards=None):
    import torch
    torch.set_num_threads(8)
    manifests={}
    expected_shards=expected_shards or {'train':576,'validation':16}
    for split,expected in [('train',options['train_items']),('validation',options['validation_items'])]:
        receipts=sorted((base/'cache').glob(f'cc3m-{split}-*.json'))
        if len(receipts)!=expected_shards[split]:raise ValueError('Incomplete cache shards')
        parts=[]
        for path in receipts:
            meta=json.loads(path.read_text())
            if meta['stage1_sha256']!=STAGE1_SHA or meta['clip_coefficients']:raise ValueError('Invalid cache source')
            data=torch.load(path.with_suffix('.pt'),weights_only=True,map_location='cpu',mmap=True)
            assert data['meta']==meta and len(data['captions'])==meta['items']
            assert data['atoms'].shape==(meta['items'],8,8,4)
            assert data['coeffs'].dtype==torch.float32 and torch.isfinite(data['coeffs']).all()
            assert int(data['atoms'].min())>=0 and int(data['atoms'].max())<16384
            sorted_atoms=data['atoms'].sort(-1).values
            assert (sorted_atoms[...,1:]!=sorted_atoms[...,:-1]).all()
            parts.append(data)
        coeffs=torch.cat([p['coeffs'] for p in parts])
        if split=='train':scales=coeffs.abs().reshape(-1,4).amax(0)/3
        assert torch.isfinite(scales).all() and (scales>0).all()
        coeffs.div_(scales)
        payload=dict(atoms=torch.cat([p['atoms'] for p in parts]),coeffs=coeffs,
            text_ids=torch.cat([p['text_ids'] for p in parts]),
            captions=[c for p in parts for c in p['captions']])
        assert len(payload['atoms'])==expected
        payload['meta']=dict(format='laser_cc3m_compound_pairs_v1',stage1_sha256=STAGE1_SHA,
            items=expected,split=split,shape=[8,8,4],clip_coefficients=False,
            coeff_scales=scales.tolist(),coeff_vocab_size=2048,coeff_max=3.,
            coefficient_storage='fp32',encoder_precision='fp32',dataset_revision=options['dataset_revision'],
            transform='resize256_center_crop256',text_tokenizer='bpe16k_huggingface',text_length=32,
            source_shards=[p.name for p in receipts])
        local=Path(options['token_cache'] if split=='train' else options['validation_cache'])
        local.parent.mkdir(parents=True,exist_ok=True)
        torch.save(payload,local)
        persistent=base/'assets'/local.name;persistent.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(local,persistent)
        manifests[split]=payload['meta']
        print(f'Merged and persisted {split}: {expected:,} images',flush=True)
        del payload,parts,coeffs
    write_json(base/'assets/cache-verification.json',dict(passed=True,**manifests))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--base',type=Path,required=True)
    args=p.parse_args();base=args.base.resolve();base.mkdir(parents=True,exist_ok=True)
    lock=open('/tmp/laser-cc3m-compound-supervisor.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prior_pid=base/'training.pid'
    if prior_pid.is_file() and alive(int(prior_pid.read_text())):
        raise RuntimeError('This run already has an active training process')
    from omegaconf import OmegaConf
    options=OmegaConf.to_container(OmegaConf.load(args.config).options,resolve=True)
    import wandb
    wb=wandb.init(entity=options['wandb_entity'],project=options['wandb_project'],id=options['wandb_id'],
        name=options['wandb_name'],mode='online',resume='allow',config=options,allow_val_change=True)
    wb.summary.update({'pipeline/phase':'cache','stage1/rfid':4.210914134979248,
        'stage1/sha256':STAGE1_SHA,'recipe/ffhq_reference':'ffhqcmp0804205803','recipe/ffhq_reference_fid':8.174392700195312,
        'recipe/official_cc3m':'https://github.com/kakaobrain/rq-vae-transformer/blob/main/configs/cc3m/cc3m-rqtransformer-8x8x4-650M.yaml'})
    write_json(base/'wandb.json',dict(url=wb.url,run_id=wb.id))
    for source in [args.config,base/'assets/stage1-provenance.json']:
        if source.is_file():wb.save(str(source),base_path=str(source.parent),policy='now')

    def status(phase,**values):
        write_json(base/'status.json',dict(phase=phase,pid=os.getpid(),timestamp=time.time(),**values))
        wb.summary['pipeline/phase']=phase
        wb.log({f'pipeline/{k}':v for k,v in values.items() if isinstance(v,(int,float))})
        print(json.dumps(dict(phase=phase,**values)),flush=True)

    def command(name,cmd):
        with (base/f'{name}.log').open('a') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            (base/f'{name}.pid').write_text(str(child.pid)+'\n')
            while child.poll() is None:time.sleep(5)
            if child.returncode:raise RuntimeError(f'{name} failed with {child.returncode}')

    try:
        attempts=0
        uploaded_reports=set()
        while True:
            receipts=list((base/'cache').glob('cc3m-*.json'))
            items=sum(json.loads(x.read_text())['items'] for x in receipts)
            status('cache',completed_shards=len(receipts),total_shards=592,cached_images=items,
                total_images=options['train_items']+options['validation_items'])
            for name in ('complete.json','ddp-complete.json','online-transport.json','persistence-test.json'):
                report=base/'preflight'/name
                if report.is_file() and name not in uploaded_reports:
                    wb.save(str(report),base_path=str(base),policy='now')
                    wb.summary['preflight/'+name.removesuffix('.json')]=json.loads(report.read_text())
                    uploaded_reports.add(name)
            pid_path=base/'cache.pid'
            if len(receipts)==592 and not (pid_path.is_file() and alive(int(pid_path.read_text()))):break
            if not pid_path.is_file() or not alive(int(pid_path.read_text())):
                if attempts>=3:raise RuntimeError('Cache retries exhausted; inspect cache.log')
                attempts+=1
                with (base/'cache.log').open('a') as log:
                    child=subprocess.Popen([sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=6',
                        str(ROOT/'scripts/tools/build_cc3m_compound_cache.py'),'--checkpoint',options['checkpoint'],
                        '--data',options['data'],'--output',str(base/'cache'),'--batch-size','32','--num-workers','3'],
                        cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                pid_path.write_text(str(child.pid)+'\n')
            time.sleep(30)
        status('merging_cache')
        merge_cache(base,options)
        checks = [
            ('preflight', base/'preflight/complete.json', [sys.executable,
                str(ROOT/'scripts/tools/preflight_cc3m_compound.py'),'--config',str(args.config),
                '--cache-shard',str(base/'cache/cc3m-train-0000.pt'),'--output',str(base/'preflight')]),
            ('ddp-preflight', base/'preflight/ddp-complete.json', [sys.executable,'-m','torch.distributed.run',
                '--standalone','--nproc-per-node=6',str(ROOT/'scripts/tools/preflight_cc3m_ddp.py'),
                '--config',str(args.config),'--cache-shard',str(base/'cache/cc3m-train-0000.pt'),
                '--output',str(base/'preflight/ddp-complete.json')]),
        ]
        for name, report, argv in checks:
            pid_file=base/f'{name}.pid'
            while not report.is_file() and pid_file.is_file() and alive(int(pid_file.read_text())):
                status('waiting_for_'+name);time.sleep(15)
            if not report.is_file():command(name,argv)
            if not json.loads(report.read_text())['passed']:raise RuntimeError(f'{name} failed')
        preflight=json.loads((base/'preflight/complete.json').read_text())
        if not preflight['passed']:raise RuntimeError('Preflight failed')
        with Path(options['checkpoint']).open('rb') as source:
            if hashlib.file_digest(source,'sha256').hexdigest()!=STAGE1_SHA:
                raise RuntimeError('The frozen stage-1 checkpoint changed')
        artifact=wandb.Artifact(options['wandb_id']+'-launch',type='run-config')
        for relative,digest in json.loads((base/'runtime-manifest.json').read_text()).items():
            if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=digest:
                raise RuntimeError(f'Runtime changed since validation: {relative}')
        for path in [args.config,base/'assets/cache-verification.json',base/'preflight/complete.json',
                     base/'preflight/ddp-complete.json',base/'runtime-manifest.json']:
            artifact.add_file(str(path),name=path.name)
        wb.log_artifact(artifact).wait()
        status('starting_training')
        wb.finish();wb=None
        command('training',[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=6',
            str(ROOT/'train.py'),'--config',str(args.config)])
        write_json(base/'status.json',dict(phase='completed',timestamp=time.time()))
    except BaseException as error:
        write_json(base/'status.json',dict(phase='failed',error=str(error),timestamp=time.time()))
        if wb is not None:wb.summary.update({'pipeline/phase':'failed','pipeline/error':str(error)});wb.finish(exit_code=1)
        raise


if __name__=='__main__':main()
