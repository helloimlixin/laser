#!/usr/bin/env python3
"""Resume the protected historical branch, then run the matched fresh decoder trial."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import zipfile

ROOT = Path('/workspace/Projects/laser')
BASE = Path('/tmp/laser-fid14-continue100-20261016')
RUN = 'imagenet-fid14-continue100-4h100-20261016'
OUT = ROOT/'outputs'/RUN
HISTORICAL = ROOT/'outputs/imagenet-rfid421-lossdecay-warm200-lr1e5-3epoch-8h100-20261006/epoch77-continuation-4h200-20261013'
SOURCE = Path('/tmp/laser-compound512-native-imagenet-20261016')
ARMS = ('interleaved', 'support_first')
CHILDREN = {order:dict(base=BASE/('next-'+order),
    run='imagenet-native-'+order.replace('_','-')+'-scratch2-4h100-afterfid14-20261016') for order in ARMS}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str)+'\n')
    temporary.replace(path)


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def replace_once(text, old, new):
    assert text.count(old) == 1, (old[:120], text.count(old))
    return text.replace(old, new, 1)


def adapt_historical_entry(text):
    text=replace_once(text,'from preserved_cloud_upload import PreservedCloudUpload',
        'PreservedCloudUpload = VerifiedCloudUpload  # New W&B identity; historical cloud slots stay intact.')
    text=replace_once(text,"continuation_identity='resume original training run ID'",
        "continuation_identity='full-state continuation in a separate W&B record'")
    text=replace_once(text,"rng_initialization='first4 of8 saved CPU/CUDA streams retained; changed layout is not bitwise identical'",
        "rng_initialization='all four saved CPU/CUDA streams retained; smaller microbatches are not bitwise identical'")
    text=replace_once(text,"WB.summary['evaluation/matched_source_fid']", "WB.summary['historical/validation_source_fid']")
    text=replace_once(text,"WB.summary['evaluation/matched_source_inception_score']", "WB.summary['historical/source_inception_score']")
    text=replace_once(text," if UPDATES==20:ARGS.save_step_freq=1\n elif UPDATES==21:ARGS.save_step_freq=626\n",'')
    return text


def prepare():
    import yaml
    assert (BASE/'frozen/runtime').is_dir()
    assert not (OUT/'prepared.json').exists(), 'Existing launch must be inspected before preparing again'
    (BASE/'source').mkdir(exist_ok=True)
    (BASE/'source/runtime').symlink_to(BASE/'frozen/runtime', target_is_directory=True)
    shutil.copytree(BASE/'frozen/support', BASE/'support', dirs_exist_ok=True)
    for source, target in [
        (ROOT/'src/training/portable_imagenet_recovery.py',BASE/'support/portable_imagenet_recovery.py'),
        (ROOT/'src/training/rq_imagenet_runtime.py',BASE/'support/imagenet_tar.py'),
        (ROOT/'scripts/tools/fid14_continuation_runtime.py',BASE/'support/fid14_continuation_runtime.py'),
        (ROOT/'src/rqvae_metrics.py',BASE/'compatible_metrics.py'),
        (HISTORICAL/'refinement_schedule.py',BASE/'support/refinement_schedule.py'),
        (HISTORICAL/'smart_sampling.py',BASE/'support/smart_sampling.py'),
        (HISTORICAL/'source-control-baseline.json',BASE/'source-control-baseline.json'),
        (ROOT/'outputs/imagenet-epoch77-refine-4h200-20261013/chunked_checkpoint_io.py',BASE/'support/chunked_checkpoint_io.py')]:
        shutil.copyfile(source,target)
    (BASE/'data').symlink_to(SOURCE/'data', target_is_directory=True)
    (BASE/'torch-cache').symlink_to(SOURCE/'torch-cache', target_is_directory=True)
    (BASE/'inputs').mkdir(exist_ok=True)
    for source, name in [(SOURCE/'inputs/stage1-model.pt','resume-stage1-tokenizer.pt'),
        (SOURCE/'inputs/imagenet_256_train.npz','imagenet_256_train.npz'),
        (ROOT/'outputs/imagenet-epoch77-refine-4h200-20261013/imagenet_256_val.npz','imagenet_256_val.npz')]:
        target=BASE/'inputs'/name
        if source.stat().st_dev == target.parent.stat().st_dev:
            os.link(source,target)
        else:
            shutil.copyfile(source,target)
    text=adapt_historical_entry((HISTORICAL/'entry.py').read_text())
    # The saved continuation already has the requested 100-epoch policy. Keep
    # its model, target construction, objective, optimizer and scheduler code.
    text=replace_once(text,"  payload=dict(payload)\n", "  payload=dict(payload)\n  if payload.get('evaluation_protocol_version')!='fid50k-fulltrain-with-validation-controller-v1':\n   payload['best_fid']=[];payload['best_inception']=[]\n")
    text=replace_once(text," payload=dict(payload,objective_revision=dict(",
        " payload=dict(payload,evaluation_protocol_version='fid50k-fulltrain-with-validation-controller-v1',objective_revision=dict(")
    text=replace_once(text," WB.summary['evaluation/reference']='ImageNet validation50000 / original RQ Inception / generated50000 / ten IS splits'",
        " WB.summary['evaluation/reference']='ImageNet full training1281167 / original RQ Inception / generated50000 / ten IS splits'")
    text=replace_once(text,"from src.training.cli import main\n", "from fid14_continuation_runtime import install\ninstall(globals())\n\nfrom src.training.cli import main\n")
    compile(text,'historical-continuation-entry.py','exec')
    (BASE/'entry.py').write_text(text)
    config=yaml.safe_load((HISTORICAL/'train.yaml').read_text())
    options=config['options']
    options.update(checkpoint=str(BASE/'inputs/resume-stage1-tokenizer.pt'),data=str(BASE/'data'),
        output=str(BASE/'production/train'),checkpoint_dir=str(OUT/'train/checkpoints'),
        batch_size=64,epochs=100,max_optimizer_steps=0,wandb_id=RUN,
        wandb_name='Historical LASER FID14.87 | full-state continuation to100 | 4H100',
        fid_real_split='train',fid_reference_stats=str(BASE/'inputs/imagenet_256_train.npz'),
        fid_batch_size=128,fid_every=1,save_step_freq=626,
        resume_checkpoint=str(BASE/'inputs/resume-source.pt'))
    (BASE/'train.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    for name in ('wandb','wandb-cache','wandb-data','inductor-cache','checkpoint-staging','checkpoint-upload-cache'):
        (BASE/name).mkdir(exist_ok=True)
    (OUT/'reproducibility').mkdir(exist_ok=True)
    for name in ('entry.py','train.yaml'):
        shutil.copyfile(BASE/name,OUT/'reproducibility'/name)
    prepare_next_trials()
    write(OUT/'prepared.json',dict(run_id=RUN,source_checkpoint=str(HISTORICAL/'train/checkpoints/last.pt'),
        source_step=48747,source_epoch=77,source_adam_age=14317,target_epoch=100,target_step=62600,
        remaining_updates=13853,world_size=4,microbatch=64,accumulation=8,global_batch=2048,
        objective='saved equal atom/coefficient CE plus range-normalized CRPS0.05',
        coefficient_sigma=.075,lr_schedule='saved epoch77-original-run-finetune-to100-v2; no reset',
        fid=dict(generated=50000,reference='train',real_images=1281167,
            validation_diagnostic='same samples; retained for historical adaptive LR controller'),
        next_trials=CHILDREN,requires_epoch100_before_new_training=True,time=time.time()))


def prepare_next_trials():
    import yaml
    sys.path[:0]=[str(ROOT),str(ROOT/'runtime')]
    from runtime.rq_imagenet_entry import freeze_shared_runtime
    source_recipe=yaml.safe_load((SOURCE/'inputs/recipe.yaml').read_text())
    for order, item in CHILDREN.items():
        base=item['base'];base.mkdir(exist_ok=True)
        freeze_shared_runtime(base,ROOT,include_stochastic_teacher=False)
        (base/'upstream').symlink_to(SOURCE/'upstream',target_is_directory=True)
        (base/'data').symlink_to(SOURCE/'data',target_is_directory=True)
        (base/'torch-cache').symlink_to(SOURCE/'torch-cache',target_is_directory=True)
        (base/'inputs').mkdir(exist_ok=True)
        for name in ['stage2-training-config.yaml','stage2-archive-config.yaml','rq-upstream-core.json','imagenet_256_train.npz']:
            shutil.copyfile(SOURCE/'inputs'/name,base/'inputs'/name)
        os.link(SOURCE/'inputs/stage1-model.pt',base/'inputs/stage1-model.pt')
        recipe=dict(source_recipe,run_id=item['run'],name='Native RQ eight events | '+order+' | matched fresh2',
            epochs=2,microbatch=64,decoder='native_support_order',event_order=order,
            atom_ce_weight=1.,auxiliary_loss_terms=0)
        recipe['targets']=dict(recipe['targets'],encoder_batch_size=8)
        recipe['evaluation']=dict(recipe['evaluation'],batch_size=128,every_epochs=1,include_epoch_one=True)
        recipe.pop('continuation',None)
        recipe.pop('initialization',None)
        (base/'inputs/recipe.yaml').write_text(yaml.safe_dump(recipe,sort_keys=False))
        configuration=yaml.safe_load((base/'inputs/stage2-training-config.yaml').read_text())
        configuration['experiment']['test_freq']=1
        configuration['experiment']['save_ckpt_freq']=1
        (base/'inputs/stage2-training-config.yaml').write_text(yaml.safe_dump(configuration,sort_keys=False))
        (base/'wandb').mkdir(exist_ok=True)
        (base/'train.py').write_text('''import hashlib,os,sys
from pathlib import Path
base=Path(os.environ['LASER_RQ_BASE'])
os.environ['RQ_UPSTREAM_ROOT']=str(base/'upstream')
sys.path[:0]=[str(base/'source'),str(base/'upstream'),str(base),str(base/'runtime')]
from src.training import rq_imagenet_shared as shared
from src.training.rq_imagenet_runtime import write_json
native=shared.fresh_model
def fresh(device):
 result=native(device)
 if int(os.environ['RANK'])==0:
  h=hashlib.sha256()
  for name,tensor in result[0].state_dict().items():
   h.update(name.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
  write_json(base/'initial-weight-hash.json',dict(sha256=h.hexdigest(),event_order=str(shared.PROFILE.recipe.event_order)))
 return result
shared.fresh_model=fresh
shared.main('laser_native_support_order',base)
''')
        item['code_manifest']={str(p.relative_to(base)):digest(p) for folder in ('source','runtime','inputs')
            for p in (base/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.yaml','.json')}
        item['code_manifest']['train.py']=digest(base/'train.py')
        write(base/'code-manifest.json',item['code_manifest'])
    manifest={str(p.relative_to(BASE)):digest(p) for folder in ('support','source/runtime')
        for p in (BASE/folder).rglob('*.py')}
    manifest.update({name:digest(BASE/name) for name in ('entry.py','train.yaml','compatible_metrics.py')})
    write(OUT/'reproducibility/code-manifest.json',manifest)
    with zipfile.ZipFile(OUT/'reproducibility/launch-source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for relative in manifest:archive.write(BASE/relative,relative)
        for name in ('scripts/tools/launch_fid14_to100_then_support_order.py','scripts/tools/fid14_continuation_runtime.py'):
            archive.write(ROOT/name,name)


def environment(phase):
    return dict(os.environ,WANDB_API_KEY=Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip(),
        LASER_RUN_BASE=str(BASE),LASER_PERSISTENT_BASE=str(OUT),LASER_PHASE=phase,
        LASER_ACCUMULATION='8',LASER_COMPILE_BLOCKS='0',LASER_CHECKPOINT_IMMUTABLE_FILES='1',
        LASER_CHECKPOINT_STAGING_DIR=str(BASE/'checkpoint-staging'),
        LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(BASE/'checkpoint-upload-cache'),
        LASER_LAYOUT_MIGRATION_DIR=str(OUT/'verification'/phase/'layout'),
        LASER_REFERENCE_SHA256='3f9c92d15755e76ec312964a819e3e19b9cf3aadc618a7ae99f5e8aa96501260',
        PYTHONPATH=str(BASE)+':'+str(BASE/'source/runtime')+':'+str(BASE/'support'),
        CUDA_VISIBLE_DEVICES='0,1,2,3',TORCH_HOME=str(BASE/'torch-cache'),
        TORCHINDUCTOR_CACHE_DIR=str(BASE/'inductor-cache'),TORCHINDUCTOR_COMPILE_THREADS='4',
        WANDB_DIR=str(BASE/'wandb'),WANDB_CACHE_DIR=str(BASE/'wandb-cache'),WANDB_DATA_DIR=str(BASE/'wandb-data'),
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',NCCL_NVLS_ENABLE='0',PYTHONUNBUFFERED='1')


def run():
    import yaml
    lock=(BASE/'sequence.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads((OUT/'prepared.json').read_text())
    assert (OUT/'source-restoration.json').exists() and (OUT/'checkpoint-verification.json').exists()
    assert json.loads((OUT/'checkpoint-verification.json').read_text())['passed']
    for relative,expected in json.loads((OUT/'reproducibility/code-manifest.json').read_text()).items():
        assert digest(BASE/relative)==expected,relative
    child=None;cancelled=False
    def status(phase,**values):
        write(OUT/'launch-status.json',dict(phase=phase,pid=os.getpid(),time=time.time(),**values))
    def stop(*_):
        nonlocal cancelled
        cancelled=True
        if child is not None and child.poll() is None:child.send_signal(signal.SIGINT)
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    def launch(script,phase,env,arguments):
        nonlocal child
        with (BASE/(phase+'.log')).open('a') as log:
            child=subprocess.Popen([sys.executable,'-m','torch.distributed.run','--standalone',
                '--nproc-per-node=4',str(script),*arguments],cwd=BASE,env=env,
                stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            status(phase,child_pid=child.pid,log=str(BASE/(phase+'.log')),target_epoch=100)
            code=child.wait()
        if code or cancelled:
            status('cancelled' if cancelled else 'failed',failed_phase=phase,returncode=code)
            raise SystemExit(code or 130)
    # Disposable updates prove strict restoration, finite gradients and the
    # resized cursor before the online continuation touches its source state.
    config=yaml.safe_load((BASE/'train.yaml').read_text())
    config['options'].update(max_optimizer_steps=2,wandb_mode='disabled',fid_every=0,
        checkpoint_dir=str(OUT/'smoke/checkpoints'),output=str(BASE/'smoke/train'))
    (BASE/'smoke.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    env=environment('smoke');env['LASER_BENCHMARK']='1'
    launch(BASE/'entry.py','smoke',env,['--config',str(BASE/'smoke.yaml')])
    for rank in range(4):
        root=OUT/'verification/smoke'
        initial=json.loads((root/f'startup-rank{rank}.json').read_text())
        after=json.loads((root/f'step1-rank{rank}.json').read_text())
        assert initial['adam_step']==14317 and initial['optimizer_parameters']==782
        assert after['finite'] and after['adam_step']==14318
        assert json.loads((root/f'resume-rng-rank{rank}.json').read_text())['restored_exactly']
    write(OUT/'smoke-verification.json',dict(passed=True,world_size=4,updates_discarded=2,
        original_adam_age=14317,original_rng_restored=True))
    launch(BASE/'entry.py','training_to_epoch100',environment('production'),['--config',str(BASE/'train.yaml')])
    final=json.loads((OUT/'trial-checkpoint-ready.json').read_text())
    assert final['global_step']==62600 and final['epoch']==100 and not final['stopping'],final
    write(OUT/'completed-to100.json',dict(passed=True,global_step=62600,epoch=100,time=time.time()))
    for order in ARMS:
        item=plan['next_trials'][order];base=Path(item['base'])
        for relative,expected in item['code_manifest'].items():assert digest(base/relative)==expected,relative
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        env=dict(environment(order),LASER_RQ_BASE=str(base),TORCH_HOME=str(base/'torch-cache'),
            WANDB_DIR=str(base/'wandb'),LASER_CHECKPOINT_STAGING_DIR=str(base/'checkpoint-staging'),
            LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(base/'checkpoint-cache'))
        launch(base/'train.py','new_decoder_'+order,env,[])
        done=json.loads((ROOT/'outputs'/item['run']/'completed.json').read_text())
        assert done['epochs']==2 and done['global_step']==1252,done
    hashes=[json.loads((Path(plan['next_trials'][order]['base'])/'initial-weight-hash.json').read_text())['sha256'] for order in ARMS]
    assert len(set(hashes))==1,hashes
    status('completed',historical_epoch=100,new_trials_epochs_per_arm=2,matched_initial_sha256=hashes[0])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run'])
    args=parser.parse_args()
    prepare() if args.action=='prepare' else run()
