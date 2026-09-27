"""Verified state relocation and run monitoring; no changes to the compound loss."""
import hashlib,json,os,signal,time
from pathlib import Path
import torch
import torch.distributed as dist

_STOP=False

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)

def install_stop_handler():
    def request_stop(signum,frame):
        global _STOP
        _STOP=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)

def should_stop(device):
    value=torch.tensor(int(_STOP or time.time()>=float(os.environ.get('CHURCH_STOP_TIME','inf'))),device=device)
    if dist.is_initialized():dist.all_reduce(value,op=dist.ReduceOp.MAX)
    return bool(value.item())

def record_progress(payload):
    base=os.environ.get('CHURCH_BASE')
    if base:
        record=dict(payload,updated_unix=time.time(),job_id=os.environ.get('SLURM_JOB_ID'))
        atomic_json(Path(base)/'train/status.json',record)
        print(json.dumps(dict(phase='training',**record)),flush=True)

def continuation_identity():
    base=Path(os.environ['CHURCH_BASE'])
    preflight=json.loads((base/'preflight.json').read_text())
    cache=json.loads((base/'prepared/cache-ready.json').read_text())
    return dict(run_id='church-laser-rfid421-ft3-compound-scratch90-20260918',
        tokenizer_sha256=preflight['tokenizer']['export_sha256'],
        reference_sha256=preflight['reference_sha256'],cache_sha256=cache['cache_sha256'],
        runtime_manifest_sha256=sha(base/'runtime-sha256.txt'))

def validate_recovered_checkpoint(payload,path):
    """Only the original checkpoint or this frozen continuation may be resumed."""
    base=Path(os.environ['CHURCH_BASE'])
    preflight=json.loads((base/'preflight.json').read_text())
    digest=sha(path)
    if digest!=preflight['stage2_sha256']:
        identity=payload['config'].get('amarel_continuation')
        accepted=[continuation_identity(),*preflight.get('accepted_continuation_identities',[])]
        if identity not in accepted:
            raise ValueError('Latest checkpoint was not produced by this validated continuation')
    step=int(payload['global_step']);epoch=int(payload['epoch']);batch=int(payload.get('batch_idx',0))
    world=int(payload['checkpoint_world_size']);microbatch=int(payload['config']['batch_size'])
    assert world in (4,8) and microbatch==16
    assert 36400<=step<=88740 and batch*world*microbatch%128==0
    assert step==epoch*986+batch*world*microbatch//128
    assert payload['scheduler']['last_epoch']==step and payload['scheduler']['T_max']==88740
    assert len(payload['optimizer']['state'])==517
    assert all(float(state['step'])==step for state in payload['optimizer']['state'].values())
    assert len(payload['rng_state_by_rank'])==world
    return dict(sha256=digest,step=step,epoch=epoch,batch_idx=batch,world_size=world)

def record_completion(step):
    base=Path(os.environ['CHURCH_BASE']);job=os.environ['SLURM_JOB_ID']
    record=dict(job_id=job,step=int(step),target_steps=88740,complete=step>=88740,
                wandb_finish_returned=True,updated_unix=time.time())
    atomic_json(base/'train'/f'completion-{job}.json',record)
    print(json.dumps(dict(phase='allocation_finished',**record)),flush=True)

def adapt_resume_payload(payload,args):
    """Validate all saved training knobs, then relocate paths after hash checks."""
    saved=payload['config']
    allowed={'checkpoint','data','token_cache','resume_checkpoint','output','checkpoint_dir',
             'fid_reference_stats','upload_token_cache','max_optimizer_steps','wandb_mode',
             'sample_grid_on_start','save_step_freq','smoke_test'}
    for key,value in vars(args).items():
        if key in saved and key not in allowed and value!=saved[key]:
            raise ValueError(f'Resume setting changed: {key}: {saved[key]!r} -> {value!r}')
    world=int(os.environ.get('WORLD_SIZE','8'))
    old_world=payload['checkpoint_world_size']
    if world not in (4,8) or old_world not in (4,8) or (old_world==4 and world==8):
        raise ValueError('Supported layouts are eight-to-eight, eight-to-four, and four-to-four')
    if world!=old_world:
        states=payload['rng_state_by_rank']
        if len(states)!=old_world:raise ValueError('Incomplete saved rank RNG state')
        payload['rng_state_by_rank']=states[:world]
        payload['amarel_world_size_change']=dict(old=old_world,new=world,
            global_batch_preserved=128,accumulation_steps=2,
            rng_policy='restore first four saved rank streams; global stochastic trajectory changes')
    base=Path(os.environ['CHURCH_BASE'])
    preflight=json.loads((base/'preflight.json').read_text())
    assert sha(args.checkpoint)==preflight['tokenizer']['export_sha256']
    assert sha(args.fid_reference_stats)==preflight['reference_sha256']
    if saved['coeff_scales']!=preflight['coeff_scales']:
        raise ValueError('Cannot replace the saved coefficient normalization')
    payload['config']=dict(saved,fid_reference_stats=str(args.fid_reference_stats))
    root=args.checkpoint_dir or args.output/'checkpoints'
    relocated=[]
    for value,old in payload.get('best_fid',[]):
        path=Path(root)/Path(old).name
        if not path.is_file():raise FileNotFoundError(f'Missing retained checkpoint: {path}')
        relocated.append((value,str(path)))
    payload['best_fid']=relocated
    return payload
