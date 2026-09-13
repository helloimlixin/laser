#!/usr/bin/env python3
"""Queue the new paired prior campaign after the existing codec jobs finish."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from src.tts_pairing import file_sha


def alive(pid):
    try:
        state=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0]
        return state != 'Z'
    except FileNotFoundError:
        return False


def preceding_campaign_done(path):
    if not path.exists(): return False
    state=json.loads(path.read_text())
    if state['status']=='running': return False
    return not any(j.get('pid') and alive(j['pid']) for j in state['jobs'])


def gpus_idle():
    output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
    return not output.strip()


def verify_preflights(root):
    results=[json.loads((root/f'preflight_{a}/completion.json').read_text()) for a in ('laser','rvq')]
    assert all(r['status']=='smoke_complete' and r['step']==2 for r in results)
    assert results[0]['data_chain']==results[1]['data_chain']
    assert results[0]['seen_frames']==results[1]['seen_frames']
    for arm in ('laser','rvq'):
        assert list((root/f'preflight_{arm}/samples').glob('epoch_*/*.wav'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('outputs/mdctcodec_tts_paired'))
    p.add_argument('--resume',action='store_true')
    args=p.parse_args(); os.chdir(REPO); root=args.root.resolve()
    lock=(root/'supervisor.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    previous=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else None
    if previous and not args.resume: raise RuntimeError('Existing queue; use --resume after inspecting its status')
    plan=json.loads((root/'plan.json').read_text())
    python=sys.executable
    jobs=[{'name':f'cache_rvq_{i}','gpu':i,'deps':[],
        'command':[python,'-u','scripts/tools/cache_mdctcodec_tts.py','--output',str(root/'rvq_cache_gpu'),
                   '--device','cuda:0','--worker-index',str(i),'--worker-count','2']} for i in range(2)]
    jobs += [
        {'name':'merge','gpu':None,'deps':['cache_rvq_0','cache_rvq_1'],'command':[python,'-u','scripts/tools/cache_mdctcodec_tts.py','--merge','--output',str(root/'rvq_cache_gpu')]},
        {'name':'finalize','gpu':None,'deps':['merge'],'command':[python,'-u','scripts/tools/prepare_mdctcodec_tts_pair.py','--finalize','--root',str(root)]}]
    for arm,gpu in [('laser',0),('rvq',1)]:
        jobs.append({'name':f'preflight_{arm}','gpu':gpu,'deps':['finalize'],
            'command':[python,'-u','scripts/tools/train_mdctcodec_tts.py','--config',str(root/f'{arm}.yaml'),
                '--output',str(root/f'preflight_{arm}'),'--smoke-steps','2','--smoke-previews','--mode','disabled']})
    for arm,gpu in [('laser',0),('rvq',1)]:
        jobs.append({'name':f'train_{arm}','gpu':gpu,'deps':['preflight_laser','preflight_rvq'],
            'command':[python,'-u','scripts/tools/train_mdctcodec_tts.py','--config',str(root/f'{arm}.yaml')]})
    jobs.append({'name':'benchmark','gpu':0,'deps':['train_laser','train_rvq'],
        'command':[python,'-u','scripts/tools/evaluate_mdctcodec_tts_pair.py','--root',str(root)]})
    for job in jobs: job['status']='queued'
    used=previous['active_gpu_hours']*3600 if previous else plan.get('preflight_gpu_seconds',0.)
    completed=set()
    if previous:
        for job in jobs:
            old=next(j for j in previous['jobs'] if j['name']==job['name'])
            if old.get('pid') and alive(old['pid']): raise RuntimeError(f'Orphan job still alive: {old["pid"]}')
            if old['status']=='complete': job.update(old); completed.add(job['name'])
            elif job['name'].startswith('train_'):
                checkpoint=root/job['name'].removeprefix('train_')/'checkpoints/last.pt'
                if checkpoint.exists(): job['command']+=['--resume',str(checkpoint)]
            elif job['name'].startswith('preflight_') and (root/job['name']/'run.json').exists():
                raise RuntimeError('Failed preflight output exists; inspect and choose a fresh preflight directory before retrying')
    import wandb
    run=wandb.init(entity='helloimlixin-rutgers',project='laser',name='mdctcodec-6kbps-paired-rqtransformer-campaign',
        group=plan['name'],job_type='paired-tts-supervisor',dir=str(root),config=plan,
        id=previous.get('wandb_id') if previous else None,resume='must' if previous else None)
    artifact=wandb.Artifact('mdctcodec-paired-rqtransformer-protocol',type='experiment-protocol')
    for path in [root/'plan.json',root/'benchmark/manifest.json',root/'laser_initial.pt',root/'rvq_initial.pt',
                 root/'gpu_preflight.json',root/'plan_before_gpu_cache_amendment.json',
                 REPO/'docs/mdctcodec-paired-rqtransformer-2026-09-13.md',
                 Path(__file__),REPO/'scripts/tools/prepare_mdctcodec_tts_pair.py',
                 REPO/'scripts/tools/evaluate_mdctcodec_tts_pair.py',REPO/'src/models/laser_tts.py',
                 REPO/'src/training/mdctcodec_tts.py',REPO/'src/tts_pairing.py']:
        if not path.exists(): continue
        artifact.add_file(str(path),name=path.name)
    run.log_artifact(artifact,aliases=['latest','frozen-plan']).wait()
    processes={}; stopped=False; failure=None; last=time.monotonic(); last_log=last
    previous_path=REPO/'outputs/mdctcodec_long_campaign/status.json'
    resources_released=bool(previous and previous.get('resources_released'))
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    try:
        while True:
            now=time.monotonic(); used+=(now-last)*sum(j['gpu'] is not None for _,j,_ in processes.values()); last=now
            for name,(process,job,log) in list(processes.items()):
                rc=process.poll()
                if rc is None: continue
                job.update(status='complete' if rc==0 else 'failed',returncode=rc,
                           finished_utc=datetime.now(timezone.utc).isoformat())
                log.close(); del processes[name]
                if rc==0:
                    if name=='finalize':
                        verified=wandb.Artifact('mdctcodec-paired-rqtransformer-protocol',type='experiment-protocol',
                            metadata={'protocol_sha256':file_sha(root/'protocol.json'),'cache_alignment_verified':True})
                        for path in [root/'protocol.json',root/'laser.yaml',root/'rvq.yaml',root/'benchmark/manifest.json']:
                            verified.add_file(str(path),name=path.name)
                        run.log_artifact(verified,aliases=['latest','verified-caches']).wait()
                    if name.startswith('train_'):
                        result=json.loads((root/name.removeprefix('train_')/'completion.json').read_text())
                        if result['completed_epochs']!=plan['train']['epochs']:
                            job['status']='incomplete'; stopped=True; failure=f'{name} stopped before the common epoch target'
                    if job['status']=='complete': completed.add(name)
                else: stopped=True; failure=f'{name} exited {rc}; see {root/(name+".log")}'
            cache_ready=all((root/'rvq_cache_gpu'/f'worker_{i}_complete.json').exists() for i in range(2))
            if not resources_released and preceding_campaign_done(previous_path) and gpus_idle(): resources_released=True
            if used>=plan['compute_budget_gpu_hours']*3600: stopped=True
            if stopped:
                for process,job,_ in processes.values():
                    if job.get('stop_sent'): continue
                    if job['name'].startswith(('train_','preflight_')): process.send_signal(signal.SIGTERM)
                    else: os.killpg(process.pid,signal.SIGTERM)
                    job['stop_sent']=True
            else:
                for job in jobs:
                    if job['status']!='queued' or not set(job['deps'])<=completed: continue
                    if job['name']=='merge' and not cache_ready: continue
                    if job['gpu'] is not None and not resources_released: continue
                    if any(other['gpu']==job['gpu'] for _,other,_ in processes.values()): continue
                    if job['name'].startswith('train_'): verify_preflights(root)
                    env={**os.environ,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','PYTHONPATH':str(REPO),
                         'HF_HOME':'/workspace/tts-benchmark-models','HF_HUB_DISABLE_XET':'1'}
                    if job['gpu'] is not None: env['CUDA_VISIBLE_DEVICES']=str(job['gpu'])
                    log=(root/(job['name']+'.log')).open('a')
                    process=subprocess.Popen(job['command'],stdout=log,stderr=subprocess.STDOUT,
                        env=env,start_new_session=True)
                    job.update(status='running',pid=process.pid,started_utc=datetime.now(timezone.utc).isoformat())
                    processes[job['name']]=(process,job,log)
            done=not processes and (stopped or len(completed)==len(jobs))
            status={'status':'stopped_budget_or_failure' if done and stopped else 'complete' if done else
                    'running' if any(j['gpu'] is not None for _,j,_ in processes.values()) else 'queued',
                'pid':os.getpid(),'wandb_id':run.id,'url':run.url,'active_gpu_hours':used/3600,
                'budget_gpu_hours':plan['compute_budget_gpu_hours'],'cache_workers_complete':cache_ready,
                'resources_released':resources_released,'waiting_for':None if resources_released else str(previous_path),
                'updated_utc':datetime.now(timezone.utc).isoformat(),'jobs':jobs,'failure':failure}
            temp=root/'status.tmp'; temp.write_text(json.dumps(status,indent=2)); temp.replace(root/'status.json')
            if now-last_log>=60 or done:
                run.summary.update({k:v for k,v in status.items() if k!='jobs'})
                run.log({'campaign/active_gpu_hours':used/3600,'campaign/completed_jobs':len(completed)})
                last_log=now
            if done: break
            time.sleep(15)
    except BaseException as error:
        run.summary.update(status='supervisor_failed',failure=str(error))
        failure_status={'status':'supervisor_failed','pid':os.getpid(),'wandb_id':run.id,'url':run.url,
            'active_gpu_hours':used/3600,'budget_gpu_hours':plan['compute_budget_gpu_hours'],
            'resources_released':resources_released,'jobs':jobs,'failure':str(error),
            'updated_utc':datetime.now(timezone.utc).isoformat()}
        temporary=root/'status.tmp';temporary.write_text(json.dumps(failure_status,indent=2));temporary.replace(root/'status.json')
        raise
    finally:
        for process,_,_ in processes.values():
            if process.poll() is None: process.send_signal(signal.SIGTERM)
        run.finish()


if __name__=='__main__': main()
