"""Promote the full-FID-verified fixed-token decoder pilot without an idle gap."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time

HERE=Path(__file__).resolve().parent
OLD=Path('/tmp/laser-s1-lr1e9-b200-20261105')
PILOT=Path('/tmp/laser-decoder-only-pilot-20261105')
ROOT=HERE.parents[1]
OLD_OUT=ROOT/'outputs/imnet-s1-laser-best2465-lr1e9-continuous-1b200-20261105'
RUN='imnet-s1-laser-decoder-refine-best24608-continuous-1b200-20261105'
BASE=Path('/tmp/laser-decoder-refinement-b200-20261105')
OUT=ROOT/'outputs'/RUN


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.pending')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    import torch
    import wandb
    torch.set_num_threads(4)
    OUT.mkdir(parents=True,exist_ok=True);BASE.mkdir(exist_ok=True)
    lock=(OUT/'supervisor.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'launch.json').exists(),'Decoder refinement is already staged/running'
    result=json.loads((PILOT/'pilot-result.json').read_text())
    assert result['improved'] and result['full_validation_images']==50000
    assert result['training_proof']['encoder_quantizer_parameters_unchanged']
    assert result['training_proof']['probe_token_changes']==0
    best=json.loads((PILOT/'run-output/best-reconstruction.json').read_text())
    source=PILOT/'run-output/tokenizer-last.pt'
    assert sha(source)==best['checkpoint']['checkpoint_sha256']
    assert abs(best['rfid']-result['candidate_rfid'])<1e-12
    assert result['candidate_rfid']<json.loads((OLD/'run-output/best-reconstruction.json').read_text())['rfid']
    # Publish the independently evaluated pilot before moving the GPU. The
    # existing continuous job remains active throughout this upload.
    stage=wandb.init(entity='helloimlixin-rutgers',project='laser',id=RUN,
        name='ImageNet stage1 | fixed tokenizer decoder refinement | rFID2.46084 | B200',
        resume='allow',dir=str(BASE),job_type='decoder-refinement',
        config={'verified_pilot':result,'continuous':True})
    stage.summary.update({'training/status':'staging_verified_candidate',
        'reconstruction/best_rfid':best['rfid']})
    artifact=wandb.Artifact(RUN+'-pilot',type='model',metadata={
        **best['checkpoint'],'rfid':best['rfid'],'count':50000,
        'encoder_quantizer_frozen':True,'source_run':OLD_OUT.name})
    artifact.add_file(str(source),name='checkpoint.pt',policy='immutable')
    artifact.add_file(str(PILOT/'run-output/config.json'),name='config.json')
    artifact.add_file(str(PILOT/'pilot-result.json'),name='full-evaluation.json')
    artifact.add_file(str(PILOT/'training-verification.json'),name='frozen-parameter-verification.json')
    logged=stage.log_artifact(artifact,aliases=['best-rfid','verified-pilot']);logged.wait()
    best['checkpoint']['artifact']=logged.qualified_name
    stage.summary['reconstruction/best_artifact']=logged.qualified_name
    stage.finish()
    runtime=BASE/'runtime'
    shutil.copytree(PILOT/'runtime',runtime,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    driver=(OLD/'runtime/continuous.py').read_text()
    marker='    base_init=trainer.Experiment.__init__'
    assert driver.count(marker)==1
    freeze="""    factory=trainer.FourScaleJointVAE
    class FrozenTokenizerVAE(factory):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.encoder.requires_grad_(False)
            self.quant_conv.requires_grad_(False)
            self.quantize.requires_grad_(False)
    trainer.FourScaleJointVAE=FrozenTokenizerVAE
"""
    driver=driver.replace(marker,freeze+marker)
    driver=driver.replace("'training/continuous':True,", "'training/continuous':True,'tokenizer/encoder_frozen':True,'tokenizer/quantizer_frozen':True,")
    (runtime/'continuous.py').write_text(driver)
    path=runtime/'train_matched_stage1.py';text=path.read_text()
    old="for group in self.d_optimizer.param_groups:group['lr']=cfg['discriminator_lr']"
    assert text.count(old)==1
    path.write_text(text.replace(old,"for group in self.d_optimizer.param_groups:group['lr']=max(cfg['discriminator_lr_floor'],cfg['discriminator_lr']*self.learning_dynamics['lr_factor'])"))
    for name in ['decoder_only_pilot.py','evaluate_decoder_pilot.py']:
        shutil.copyfile(HERE/name,runtime/name)
    cfg=json.loads((PILOT/'run-output/config.json').read_text())
    cfg.pop('decoder_only_pilot',None)
    local=BASE/'run-output';local.mkdir(exist_ok=True)
    cfg.update(output=str(local),source_bundle=str(BASE/'source.tar.gz'),
        microbatch=64,accumulation=2,workers=8,lr=1e-7,
        discriminator_lr=1e-9,discriminator_lr_floor=1e-11,
        wandb=dict(id=RUN,name='ImageNet stage1 | fixed tokenizer decoder refinement | rFID2.46084 | B200'))
    cfg['decoder_refinement']=dict(frozen_prefixes=['encoder.','quant_conv.','quantize.'],
        pilot_result=result,ema=False,inference_precision_and_fid_protocol_unchanged=True)
    policy=cfg['continuation']
    policy.update(source_run='helloimlixin-rutgers/laser/'+OLD_OUT.name,
        source_artifact=logged.qualified_name,source_step=best['step'],
        source_rfid=best['rfid'],source_checkpoint_sha256=best['checkpoint']['checkpoint_sha256'],
        durable=str(OUT),minimum_improvement=.0005)
    policy['learning_rate'].update(maximum=1e-7,minimum=1e-7,absolute_floor=1e-9,
        start_step=best['step'],kind='decoder_constant_with_plateau_and_recovery_reductions')
    cfg['runtime_sha256']={name:sha(runtime/name) for name in cfg['runtime_sha256']}
    cfg['runtime_sha256']['evaluate_decoder_pilot.py']=sha(runtime/'evaluate_decoder_pilot.py')
    contract={k:v for k,v in cfg.items() if k not in {'output','wandb','workers','checkpoint_every','preview_every','source_bundle'}}
    contract_sha=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    state=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
    progress=dict(state['progress']);assert progress['batch']%4==0;progress['batch']//=4
    payload=dict(state,progress=progress,contract_sha256=contract_sha,
        learning_dynamics=dict(lr_factor=1.,nonimproving_checks=0,evaluations=0,reductions=0,recoveries=0))
    torch.save(payload,local/'tokenizer-last.pt')
    restored=torch.load(local/'tokenizer-last.pt',map_location='cpu',weights_only=False,mmap=True)
    sys.path.insert(0,str(runtime))
    from migrate_stage1_world_size import assert_equal,global_indices
    for key in ['model','discriminator','optimizer','discriminator_optimizer','rng']:
        assert_equal(state[key],restored[key])
    assert global_indices(1281167,1,16,8,state['progress']['batch'],4,cfg['seed'])==global_indices(1281167,1,64,2,progress['batch'],4,cfg['seed'])
    ages={name:sorted({int(v['step']) for v in state[name]['state'].values() if 'step' in v})
        for name in ['optimizer','discriminator_optimizer']}
    assert ages==dict(optimizer=[43136,43166],discriminator_optimizer=[33166])
    os.link(source,local/'tokenizer-best-rfid.pt')
    proof=dict(passed=True,source_rfid=best['rfid'],source_artifact=logged.qualified_name,
        model_both_optimizers_rng_preserved=True,next_three_global_batches_identical=True,
        original_microbatch=16,new_microbatch=64,global_batch=128,
        source_progress=state['progress'],progress=progress,optimizer_ages=ages,
        frozen_parameter_training_proof=result['training_proof'])
    for name,value in [('config.json',cfg),('contract.json',contract),('initialization.json',proof),
        ('resume-verification.json',proof),('best-reconstruction.json',best),('pilot-result.json',result)]:
        write(local/name,value);write(OUT/name,value)
    write(local/'checkpoint.json',dict(progress=progress,checkpoint_sha256=sha(local/'tokenizer-last.pt'),
        contract_sha256=contract_sha,full_state=True))
    with tarfile.open(BASE/'source.tar.gz','w:gz') as archive:
        for path in runtime.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                archive.add(path,arcname='runtime/'+str(path.relative_to(runtime)))
    current=json.loads((OLD/'run-output/best-reconstruction.json').read_text())
    assert best['rfid']<current['rfid'],'The live job found a better checkpoint during staging'
    status=json.loads((OLD_OUT/'supervisor-status.json').read_text())
    parent=status['supervisor_pid'];child=status['child_pid']
    assert b'launch_smaller_lr.py' in Path(f'/proc/{parent}/cmdline').read_bytes()
    parents={}
    for process in Path('/proc').iterdir():
        if process.name.isdigit():
            try:parents[int(process.name)]=int((process/'stat').read_text().split()[3])
            except (OSError,ValueError):pass
    descendants={child}
    while True:
        expanded=descendants|{pid for pid,p in parents.items() if p in descendants}
        if expanded==descendants:break
        descendants=expanded
    write(OLD_OUT/'decoder-refinement-handoff.json',dict(time=time.time(),new_run=RUN,
        old_best=current['rfid'],new_best=best['rfid'],verified_pilot=result))
    os.kill(parent,signal.SIGTERM)
    for pid in sorted(descendants,reverse=True):
        try:
            if os.getpgid(pid)==pid:os.killpg(pid,signal.SIGKILL)
            else:os.kill(pid,signal.SIGKILL)
        except ProcessLookupError:pass
    write(OLD_OUT/'supervisor-status.json',dict(phase='superseded_verified_decoder_refinement',new_run=RUN,time=time.time()))
    command=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc-per-node=1',
        str(runtime/'continuous.py'),'--config',str(local/'config.json')]
    env={k:v for k,v in os.environ.items() if k not in ['WANDB_SERVICE','_WANDB_SERVICE']}
    env.update(LASER_VGG16_WEIGHTS='/tmp/laser-s1-best-b200/weights/vgg16-397923af.pth',
        OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',
        WANDB_DISABLE_GIT='true',WANDB_MODE='online',PYTHONUNBUFFERED='1',
        PYTORCH_ALLOC_CONF='expandable_segments:True')
    write(OUT/'launch.json',dict(run_id=RUN,time=time.time(),source=proof,
        generator_lr=1e-7,discriminator_lr=1e-9,frozen_encoder_quantizer=True,
        continuous=True,microbatch=64,accumulation=2,global_batch=128,
        first_full_evaluation_step=best['step']+100,log=str(BASE/'training.log')))
    failures=0
    while True:
        with (BASE/'training.log').open('ab',buffering=0) as log:
            process=subprocess.Popen(command,cwd=runtime,env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while process.poll() is None:
            write(OUT/'supervisor-status.json',dict(phase='training',continuous=True,
                supervisor_pid=os.getpid(),child_pid=process.pid,failures=failures,time=time.time()))
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:pass
        if process.returncode==0 or (local/'STOP').exists():return
        failures+=1
        write(OUT/'supervisor-status.json',dict(phase='restarting_after_error',failures=failures,time=time.time()))
        if failures>=5:raise RuntimeError('Repeated decoder refinement runtime failure')
        time.sleep(2)


if __name__=='__main__':main()
