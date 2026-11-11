"""Freeze the strongest available EMA, search it, then transfer finalists forward."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import py_compile
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
TRAIN=ROOT/'outputs/imagenet-rq24576-refine80-lowLR-noise-ema-8h100-20261110'
BASE=Path('/tmp/laser-rq24576-refine80-8h100-20261110')
LOCAL=Path('/tmp/laser-rq24576-ema-sampler-search-20261110')
OUT=ROOT/'outputs/imagenet-rq24576-ema-sampler-search-20261110'


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_name(path.name+f'.{os.getpid()}.pending');pending.write_text(json.dumps(value,indent=2)+'\n');pending.replace(path)


def policies():
    k=[1536,2048,3072,6144];wide=[2048,2560,4096,8192];p=[.95,.95,.98,.99]
    def setting(t,ks=None,ps=None):return dict(temperature=t,top_k=ks or list(k),top_p=ps or list(p))
    presets={'current':setting(.975)}
    for label,t in [('0900',.9),('0925',.925),('0950',.95),('1000',1.),('1025',1.025),('1050',1.05)]:
        presets['t'+label]=setting(t)
    for label,t in [('0950',.95),('0975',.975),('1000',1.)]:presets['wide_t'+label]=setting(t,wide)
    presets.update(depth_ramp=setting([.925,.95,.975,1.],wide),
        depth_ramp_warm=setting([.95,.975,1.,1.025],wide),
        depth_cool_front=setting([.925,.925,.975,.975],wide),
        tighter_tail=setting(.975,ps=[.95]*4),
        broad_nucleus=setting(.975,wide,[.97,.97,.99,.995]),
        narrow_semantic=setting(.95,[1024,1536,3072,6144],[.93,.95,.98,.99]),
        warm_tail=setting([.95,.975,1.025,1.075],[2048,4096,8192,16384],[.95,.97,.99,.995]),
        open_tail=setting([.975,1.,1.025,1.05],[2048,4096,8192,24576],[.95,.97,.99,1.]),
        reverse_depth_ramp=setting([1.,.975,.95,.925],wide),
        native_k256=setting(1.,[256]*4,[.95]*4))
    assert len(presets)==20
    for s in presets.values():
        ts=s['temperature'] if isinstance(s['temperature'],list) else [s['temperature']]*4
        assert len(ts)==len(s['top_k'])==len(s['top_p'])==4
        assert all(t>0 for t in ts) and all(0<p<=1 for p in s['top_p'])
        assert all(1<=n<=24576 for n in s['top_k'])
    assert len({json.dumps(s,sort_keys=True) for s in presets.values()})==20
    return presets


def best_available():
    index=read(TRAIN/'epoch-checkpoints.json');rows=[]
    for path in (TRAIN/'evaluations').glob('epoch*.json'):
        row=read(path);epoch=row['train/epoch'];state=index.get(str(epoch*626))
        if state is None:continue
        assert state['verified_bytes']==state['bytes'] and state['persistent_state']=='COMPLETE'
        assert row['generated_images']==50000 and row['real_split']=='train' and row['real_images']==1281167
        assert row['ema']['weights']=='ema' and not row['rejection_sampling']
        rows.append((row['ema']['fid'],epoch,row,state))
    assert rows
    _,epoch,evaluation,receipt=min(rows,key=lambda r:r[0])
    return epoch,evaluation,receipt


def build_plan(local,out,epoch,evaluation,receipt,kind,primary=None):
    local.mkdir(exist_ok=True);out.mkdir(exist_ok=True)
    source=Path(receipt['path'])
    if not source.exists():source=Path(receipt['persistent_path'])
    checkpoint=local/'frozen-checkpoint.pt'
    if not checkpoint.exists():os.link(source,checkpoint)
    assert checkpoint.stat().st_size==receipt['bytes']
    with checkpoint.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    assert digest==receipt['sha256']
    presets=policies();all_policies=dict(presets);seeds={};routing={}
    for name,sampler in presets.items():
        seeds[name]=26111086;routing[name]=name
        for seed in [261001,271001,281001]:
            key=name+'__s'+str(seed);all_policies[key]=sampler;seeds[key]=seed;routing[key]=name
    classes=[dict(name=n,imagenet_id=c) for n,c in [('cheeseburger',933),('teapot',849),('samoyed',258),
        ('bald eagle',22),('red fox',277),('hotpot',926),('wombat',106),('timber wolf',269)]]
    ema=evaluation['ema']
    assert ema['sampler']==presets['current']
    plan=dict(base=str(BASE),local=str(local),output=str(out),source_training_output=str(TRAIN),source_run=TRAIN.name,
        source_epoch=epoch,source_step=epoch*626,checkpoint=str(checkpoint),checkpoint_bytes=receipt['bytes'],
        checkpoint_sha256=digest,weights='ema',state_dict_key='state_dict',ema_decay=ema['ema_decay'],ema_updates=ema['ema_updates'],
        policies=all_policies,policy_seeds=seeds,case_base_policy=routing,screen_policies=list(presets) if primary is None else [],
        prespecified_primary_policies=primary,screen_images=10000,screen_seed=26111086,
        confirm_images=50000,confirm_seed=261001,is_shuffle_seed=261002,generation_seeds=[261001,271001,281001],replication_seeds=[271001,281001],
        world_size=8,batch=64,decode_batch=8,memory_gib=12,study_kind=kind,screening_configurations=20 if primary is None else 0,
        reference_images=1281167,reference_split='train',reference_sha256='3f9c92d15755e76ec312964a819e3e19b9cf3aadc618a7ae99f5e8aa96501260',
        active_recipe_sha256=hashlib.sha256((BASE/'inputs/recipe.yaml').read_bytes()).hexdigest(),
        logged_baseline=ema,training_continues=True,no_extra_hardware=True,active_training_sampler_changed=False,
        inference_sampler_precision='native FP16',decoder_precision='FP32',inception_precision='FP32',
        rejection_sampling=False,images_logged=True,image_logging_required=True,
        adapter_probe=presets['depth_ramp'],published_target_fid=11.56,earlier_original_best_fid=12.65257290553177,
        teacher_temperature_at_checkpoint=.25,training_teacher_temperature_annealed=True,
        screen_metric='FID10k (biased diagnostic, not a benchmark score) plus unbiased normalized moment MMD; confirm with full50k.',
        selection='Current control plus top four FID10k candidates, best moment-MMD, best IS10k, and depth ramp; at most eight full50k primary cases.',
        recommendation_rule='FID-first: at least0.05 fresh-seed mean improvement, lower FID on both fresh seeds and lower three-seed mean. Report IS tradeoffs separately.',
        previews=dict(classes=classes,images_per_class=8,seed_offset=86000000,primary_seed=261001,
            unfiltered=True,excluded_from_metric_images=True,log_every_screen_config=True,log_every_confirm_config=True,
            screen_uses_matched_primary_preview_seed=True),created_at=time.time())
    retention=dict(passed=True,weights='ema',source_epoch=epoch,source_checkpoint_sha256=digest,
        bytes=receipt['bytes'],verified_bytes=receipt['verified_bytes'],pinned_local_checkpoint=str(checkpoint),
        canonical_persistent_checkpoint=receipt['persistent_path'],all_evaluated_training_checkpoints_retained=True)
    for root in [local,out]:write(root/'plan.json',plan);write(root/'checkpoint-retention.json',retention)
    return plan


def run_study(plan):
    local=Path(plan['local']);out=Path(plan['output'])
    write(out/'queue-status.json',dict(phase='memory_preflight',source_epoch=plan['source_epoch'],weights='ema',time=time.time()))
    for path in HERE.glob('*.py'):py_compile.compile(str(path),doraise=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    with (local/'preflight.log').open('ab',buffering=0) as log:
        result=subprocess.run([sys.executable,'-u',str(HERE/'worker.py'),'--plan',str(local/'plan.json'),
            '--rank','0','--preflight'],env=env,stdout=log,stderr=subprocess.STDOUT)
    assert result.returncode==0,'Inference memory preflight failed; training is independent'
    proof=read(local/'preflight-rank-0.json');assert proof['passed']
    assert proof['peak_reserved_gib']<12
    memory=dict(proof,passed=True,training_peak_allocated_gib=58.516,training_expected_reserved_gib=61,
        physical_device_memory_gib=79.648,combined_reserved_budget_gib=73,
        training_and_inference_processes_independent=True)
    for root in [local,out]:write(root/'memory-preflight.json',memory)
    with (local/'coordinator.log').open('ab',buffering=0) as log:
        child=subprocess.Popen([sys.executable,'-u',str(HERE/'sweep.py'),'--plan',str(local/'plan.json')],
            stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    write(local/'launch-pids.json',dict(coordinator=child.pid,orchestrator=os.getpid()))
    write(out/'launch.json',dict(coordinator=child.pid,orchestrator=os.getpid(),source_epoch=plan['source_epoch'],weights='ema'))
    write(out/'queue-status.json',dict(phase='search_running',source_epoch=plan['source_epoch'],coordinator=child.pid,time=time.time()))
    print(json.dumps(dict(event='study_launched',pid=child.pid,output=str(out),source_epoch=plan['source_epoch'])),flush=True)
    assert child.wait()==0,'Sampler coordinator failed; training is independent'
    assert read(out/'status.json')['phase']=='completed'
    return read(out/'report.json')


def main():
    lock=(LOCAL/'prepare.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(OUT/'queue-status.json',dict(phase='selecting_best_available_ema',time=time.time()))
    progress_path=TRAIN/'generation-progress.json'
    if progress_path.exists():
        pending_epoch=read(progress_path)['epoch']
        deadline=time.monotonic()+300
        while str(pending_epoch*626) not in read(TRAIN/'epoch-checkpoints.json') and time.monotonic()<deadline:
            write(OUT/'queue-status.json',dict(phase='waiting_for_in_progress_evaluation',pending_epoch=pending_epoch,time=time.time()))
            time.sleep(5)
    epoch,evaluation,receipt=best_available()
    plan=build_plan(LOCAL,OUT,epoch,evaluation,receipt,'20-config screen and fresh-seed confirmation')
    report=run_study(plan)
    # Re-evaluate finalists on the strongest later EMA rather than leaving a stale study.
    next_epoch,next_eval,next_receipt=best_available()
    if next_epoch>epoch and next_eval['ema']['fid']<evaluation['ema']['fid']-.02:
        primary=read(LOCAL/'primary-summary.json')
        names=list(dict.fromkeys(['current',primary['lowest_fid']['base_policy'],primary['strongest_balance']['base_policy']]))
        local=Path(f'/tmp/laser-rq24576-ema-sampler-transfer-epoch{next_epoch:03d}-20261110')
        out=ROOT/'outputs'/f'imagenet-rq24576-ema-sampler-transfer-epoch{next_epoch:03d}-20261110'
        transfer=build_plan(local,out,next_epoch,next_eval,next_receipt,'finalist transfer with fresh-seed confirmation',names)
        write(OUT/'follow-up.json',dict(source_epoch=next_epoch,output=str(out),policies=names,status='launching'))
        run_study(transfer)
        write(OUT/'follow-up.json',dict(source_epoch=next_epoch,output=str(out),policies=names,status='completed'))
    write(OUT/'queue-status.json',dict(phase='completed',initial_source_epoch=epoch,time=time.time()))


if __name__=='__main__':
    try:main()
    except Exception as error:
        write(OUT/'queue-status.json',dict(phase='failed',error=str(error),error_type=type(error).__name__,time=time.time()));raise
