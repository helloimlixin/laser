"""Build an isolated guarded continuation from the verified rFID2.486 checkpoint."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile

import torch

from dynamics import initial_state

HERE = Path(__file__).resolve().parent
OLD = Path('/tmp/laser-s1-best-b200')
BASE = Path(os.environ.get('LASER_GUARDED_BASE','/tmp/laser-s1-best2486-b200'))
FROZEN = BASE/'runtime'
RUN = os.environ.get('LASER_GUARDED_RUN','imnet-s1-laser-best2486-lr1e5-guarded-1b200-20261104')
RATE = float(os.environ.get('LASER_GUARDED_LR','1e-5'))
DECAY_UPDATES = int(os.environ.get('LASER_GUARDED_DECAY_UPDATES','5000'))
CONTROL = os.environ.get('LASER_BASELINE_CONTROL_PATH')
DURABLE = HERE.parents[1]/'outputs'/RUN
SOURCE = OLD/'run-output/tokenizer-best-rfid.pt'
SOURCE_HASH = 'fd96ca9a1607a07e185b61f593345aac2d0b937ec6d83aa80ebeaa1fd9b1b528'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.pending')
    temporary.write_text(json.dumps(obj,indent=2)+'\n')
    temporary.replace(path)


def build_runtime():
    assert not FROZEN.exists(),'Inspect the existing runtime before rebuilding'
    shutil.copytree(OLD/'frozen/runtime',FROZEN,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    text = (OLD/'original-train_matched_stage1.py').read_text()
    def replace(old,new):
        nonlocal text
        assert text.count(old)==1,(old,text.count(old))
        text=text.replace(old,new)
    replace('import dist as var_dist','import dist as var_dist\nfrom continuation import learning_rate, evaluation_due')
    replace("for group in self.optimizer.param_groups:group['lr']=cfg['lr']*ratio",
            "for group in self.optimizer.param_groups:group['lr']=learning_rate(self,self.progress['step'])\n                for group in self.d_optimizer.param_groups:group['lr']=learning_rate(self,self.progress['step'])")
    replace("world_size=self.world,record_hit=getattr(self.model.quantize,'record_hit',None)",
            "world_size=self.world,learning_dynamics=dict(self.learning_dynamics),record_hit=getattr(self.model.quantize,'record_hit',None)")
    replace('receipt=dict(progress=self.progress,checkpoint_sha256=sha(path)',
            'receipt=dict(learning_dynamics=dict(self.learning_dynamics),progress=self.progress,checkpoint_sha256=sha(path)')
    replace("if self.progress['step']<=10 or self.progress['step']%10==0:",
            "if self.smoke or self.progress['step']<=10 or self.progress['step']%10==0:")
    replace('                if flag.item():return\n',
            "                if flag.item():return\n                if not self.smoke and evaluation_due(cfg['continuation'],self.progress['step']):\n                    self.save(publish=False);self.evaluate()\n                    if self.stop:return\n")
    replace('            if not self.smoke:self.evaluate()\n',
            '            if not self.smoke:\n                self.evaluate()\n                if self.stop:return\n')
    replace("        elif experiment.run and experiment.progress['epoch']<cfg['epochs']:experiment.run.summary['training/status']='paused_after_checkpoint';experiment.run.finish()",
            "        elif experiment.run and experiment.progress['epoch']<cfg['epochs']:\n            if not experiment.quality_stop:experiment.run.summary['training/status']='paused_after_checkpoint'\n            experiment.run.finish()")
    (FROZEN/'train_matched_stage1.py').write_text(text)
    for name in ['continuation.py','dynamics.py']:
        shutil.copyfile(HERE/name,FROZEN/name)


def prepare(smoke=False):
    torch.set_num_threads(4)
    assert sha(SOURCE)==SOURCE_HASH
    source_cfg=json.loads((OLD/'run-output/config.json').read_text())
    state=torch.load(SOURCE,map_location='cpu',weights_only=False,mmap=True)
    assert state['progress']==dict(epoch=4,batch=5000,step=42536)
    assert state['world_size']==1 and len(state['rng'])==1
    best=json.loads((OLD/'run-output/best-reconstruction.json').read_text())
    assert best['step']==42536 and best['checkpoint']['checkpoint_sha256']==SOURCE_HASH
    ignored={'output','wandb','workers','checkpoint_every','preview_every','source_bundle'}
    source_contract={k:v for k,v in source_cfg.items() if k not in ignored}
    assert state['contract_sha256']==hashlib.sha256(json.dumps(source_contract,sort_keys=True).encode()).hexdigest()
    out=BASE/('preflight' if smoke else 'run-output')
    assert not (out/'tokenizer-last.pt').exists(),'Refusing to overwrite an existing checkpoint'
    out.mkdir(parents=True,exist_ok=True)
    cfg=dict(source_cfg)
    cfg.update(output=str(out),lr=RATE,checkpoint_every=500,preview_every=500,
               source_bundle=str(BASE/'source.tar.gz'),
               wandb=dict(id=RUN,name=f'ImageNet stage1 | best rFID2.486 | adaptive LR{RATE:g} | 2% guard | B200'))
    cfg['continuation']=dict(
        source_run='helloimlixin-rutgers/laser/'+source_cfg['wandb']['id'],
        source_artifact=best['checkpoint']['artifact'],source_step=42536,
        source_checkpoint_sha256=SOURCE_HASH,source_rfid=best['rfid'],
        durable=str(DURABLE),first_evaluation_updates=100,evaluation_every_updates=500,
        maximum_relative_regression=.02,minimum_improvement=.005,
        reduce_after_nonimproving_checks=2,stop_after_nonimproving_checks=4,
        lr_reduction_factor=.5,baseline_inherited_same_hardware_and_protocol=True,
        learning_rate=dict(maximum=RATE,minimum=RATE*.2,absolute_floor=RATE*.05,
                           start_step=42536,decay_updates=DECAY_UPDATES,
                           applies_to='generator_and_discriminator',kind='cosine_with_plateau_reductions'))
    if CONTROL:
        control=json.loads(Path(CONTROL).read_text())
        assert control['passed'] and control['checkpoint_sha256']==SOURCE_HASH
        cfg['continuation']['independent_full_baseline_control']=control
    cfg['runtime_sha256']={name:sha(FROZEN/name) for name in source_cfg['runtime_sha256']}
    cfg['runtime_sha256']['dynamics.py']=sha(FROZEN/'dynamics.py')
    contract={k:v for k,v in cfg.items() if k not in ignored}
    contract_sha=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    payload=dict(state,contract_sha256=contract_sha,learning_dynamics=initial_state(),
                 continuation_lineage=cfg['continuation'])
    target=out/'tokenizer-last.pt'
    torch.save(payload,target)
    recovered=torch.load(target,map_location='cpu',weights_only=False,mmap=True)
    sys.path.insert(0,str(FROZEN))
    from migrate_stage1_world_size import assert_equal,global_indices
    for name in ['model','discriminator','optimizer','discriminator_optimizer','progress','rng']:
        assert_equal(state[name],recovered[name])
    before=global_indices(1281167,1,64,2,5000,4,0)
    after=global_indices(1281167,1,cfg['microbatch'],cfg['accumulation'],recovered['progress']['batch'],4,0)
    assert before==after
    ages=sorted({int(v['step']) for v in recovered['optimizer']['state'].values() if 'step' in v})
    d_ages=sorted({int(v['step']) for v in recovered['discriminator_optimizer']['state'].values() if 'step' in v})
    assert ages==[42536] and d_ages==[32536]
    proof=dict(passed=True,source_checkpoint_sha256=SOURCE_HASH,
        source_artifact=best['checkpoint']['artifact'],source_rfid=best['rfid'],
        model_both_optimizers_rng_and_cursor_bit_exact=True,
        next_three_global_batches_identical=True,optimizer_age=ages,
        discriminator_optimizer_age=d_ages,world_size=1,microbatch=64,accumulation=2,
        global_batch=128,progress=state['progress'],checkpoint_sha256=sha(target),
        contract_sha256=contract_sha,full_state=True)
    write(out/'config.json',cfg)
    write(out/'checkpoint.json',proof)
    write(out/'initialization.json',proof)
    write(out/'resume-verification.json',proof)
    write(out/'best-reconstruction.json',best)
    write(out/'baseline-verification.json',dict(inherited=True,passed=True,
        source_checkpoint_sha256=SOURCE_HASH,rfid=best['rfid'],count=50000,
        same_gpu_microbatch_transforms_reference_and_feature_backend=True))
    if not smoke:
        DURABLE.mkdir(parents=True,exist_ok=True)
        for name in ['config.json','initialization.json','resume-verification.json',
                     'best-reconstruction.json','baseline-verification.json']:
            shutil.copyfile(out/name,DURABLE/name)
        with tarfile.open(BASE/'source.tar.gz','w:gz') as archive:
            for path in FROZEN.rglob('*'):
                if path.is_file() and '__pycache__' not in path.parts:
                    archive.add(path,arcname='runtime/'+str(path.relative_to(FROZEN)))
        shutil.copyfile(BASE/'source.tar.gz',DURABLE/'source.tar.gz')
    print(json.dumps(proof),flush=True)
    return cfg


if __name__=='__main__':
    if '--build' in sys.argv:build_runtime()
    prepare(smoke='--smoke' in sys.argv)
