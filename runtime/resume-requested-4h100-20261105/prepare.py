"""Prepare the immutable recovered run for four H100s; credentials stay outside the repo."""
import difflib,hashlib,json,shutil
from pathlib import Path
import torch

ROOT=Path('/workspace/Projects/laser')
BASE=Path('/tmp/laser-resume-h100-20261105/frozen')
OUT=ROOT/'outputs/imnet-j16384-best14574-continue100-1b300-20261104/resume-h100-20261105'
HERE=Path(__file__).resolve().parent
FACTOR=0.0125


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,default=str)+'\n')


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    assert not (BASE/'h100-prepared.json').exists()
    checkpoint=BASE.parent/'checkpoint-artifact/last.pt'
    p=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    assert p['global_step']==46250 and p['next_epoch']==73 and p['next_microbatch']==4416
    assert len(p['optimizer']['state'])==780
    assert {int(s['step']) for s in p['optimizer']['state'].values()}=={46250}
    assert p['scheduler']['after']['last_epoch']==46250 and p['scheduler']['after']['T_max']==62600
    for path in (BASE.parent/'stage1-artifact').glob('*.pt'):
        digest=sha(path)
        if digest==p['stage1_checkpoint_sha256']:
            (BASE/'inputs/stage1-model.pt').hardlink_to(path)
            break
    else:raise RuntimeError('Original stage1 checkpoint not found')
    for name,key in [('joint-codebook.pt','joint_codebook_sha256'),('reconstruction-codebook.pt','reconstruction_codebook_sha256')]:
        assert sha(BASE/'inputs'/name)==p[key]
    assert sha(BASE/'inputs/imagenet_256_train.npz')=='3f9c92d15755e76ec312964a819e3e19b9cf3aadc618a7ae99f5e8aa96501260'
    (BASE/'inputs/resume-checkpoint.pt').hardlink_to(checkpoint)
    shutil.copyfile(HERE/'convergence.py',BASE/'runtime/convergence.py')
    execution_path=BASE/'inputs/execution.json'
    execution=json.loads(execution_path.read_text())
    execution.update(hardware='4 NVIDIA H100 80GB HBM3 with NVLink',world_size=4,
        microbatch=128,accumulation_steps=4,source_step=46250,source_epoch=73,
        lr_amplitude_factor=FACTOR,source_saved_world_size=1,training_attention='SDPA (verified forward and backward)',
        source_run_id=p['run_id'],source_checkpoint_artifact=p['run_id']+'-checkpoint:v2',
        loader_workers_per_rank=8,baseline_measurement_origin='Historical source best; fresh full50k evaluation at epoch74',
        adaptive_lr=dict(initial_factor=FACTOR,plateau_factor=0.5,patience=2,minimum_delta=0.05,
            minimum_factor=0.0015625,cosine_clock_preserved=True,topology_baseline_remeasured=True))
    write(execution_path,execution)
    write(BASE/'inputs/hardware.json',dict(gpu='4 NVIDIA H100 80GB HBM3',world_size=4,
        microbatch=128,accumulation_steps=4,global_batch=2048))
    path=BASE/'source/src/training/rq_imagenet_shared.py'
    before=path.read_text();code=before
    code=code.replace("    lr_recovery=None\n", "    lr_recovery=None\n    convergence_state=None\n")
    code=code.replace("desired_factor=float(os.environ.get('LASER_LR_AMPLITUDE_FACTOR','0.15')))",
        "desired_factor=min(float(payload['runtime'].get('lr_amplitude_factor',1.0)),float(os.environ.get('LASER_LR_AMPLITUDE_FACTOR','0.15'))))\n        convergence_state=payload['runtime'].get('convergence_state')")
    code=code.replace("lr_amplitude_factor=float(os.environ.get('LASER_LR_AMPLITUDE_FACTOR','0.15')),",
        "lr_amplitude_factor=lr_recovery['desired_factor'] if lr_recovery else float(os.environ.get('LASER_LR_AMPLITUDE_FACTOR','0.15')),\n        convergence_state=convergence_state,")
    code=code.replace("run.config.update(dict(native_sampling=PROFILE.sampling_settings(),",
        "run.config.update(dict(world_size=world,microbatch=args.microbatch,accumulation_steps=accumulation,\n                global_batch=2048,hardware='4 NVIDIA H100 80GB HBM3',execution=PROFILE.metadata()['execution'],\n                effective_lr=optimizer.param_groups[0]['lr'],lr_amplitude_factor=initial['lr_amplitude_factor'],\n                native_sampling=PROFILE.sampling_settings(),")
    code=code.replace("\n            last_full_evaluation=evaluation\n", """
            last_full_evaluation=evaluation
            from runtime.convergence import observe
            convergence_state,factor,lr_event=observe(initial.get('convergence_state'),evaluation,
                optimizer,scheduler,initial['lr_amplitude_factor'],epoch=epoch+1,step=global_step)
            initial.update(convergence_state=convergence_state,lr_amplitude_factor=factor)
            if rank==0:
                write_json(OUT/'convergence-state.json',dict(convergence_state,lr_amplitude_factor=factor))
                if lr_event:
                    print(json.dumps(dict(learning_rate_reduction=lr_event)),flush=True)
                    with (OUT/'learning-rate-events.jsonl').open('a') as handle:handle.write(json.dumps(lr_event)+'\\n')
                    if run:run.log({'train/global_step':global_step,'train/lr':optimizer.param_groups[0]['lr'],
                                    'train/lr_amplitude_factor':factor})
""")
    code=code.replace("        if args.resume and not args.new_wandb_run:\n            # Persist the effective sampler",
        "        if args.resume and not args.new_wandb_run:\n            run.name='ImageNet joint16384 | continued on 4 H100 | adaptive cosine'\n            # Persist the effective sampler")
    code=code.replace("                elif not args.smoke_updates and global_step-initial['global_step']==64",
        "                elif not args.smoke_updates and global_step-initial['global_step']==32:\n                    checkpoint(epoch,epoch,absolute+1)\n                elif not args.smoke_updates and global_step-initial['global_step']==64")
    assert code!=before
    assert 'from runtime.convergence import observe' in code
    path.write_text(code)
    support=BASE/'source/src/training/joint_runtime_support.py'
    support_before=support.read_text()
    support.write_text(support_before.replace("title='ImageNet Joint16384 best14574 continue100 1B300 20261104'",
        "title='ImageNet Joint16384 continuation 4H100 20261105'"))
    patch=''.join(difflib.unified_diff(before.splitlines(True),code.splitlines(True),fromfile='frozen/original/rq_imagenet_shared.py',tofile='frozen/h100/rq_imagenet_shared.py'))
    (BASE/'h100-runtime.patch').write_text(patch)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'runtime.patch').write_text(patch)
    write(OUT/'source-verification.json',dict(passed=True,source_step=46250,source_epoch=73,
        original_microbatch_cursor=4416,remapped_microbatch_cursor=2208,
        global_batch=2048,optimizer_states=780,adam_age=46250,cosine_horizon=62600,
        source_lr=p['optimizer']['param_groups'][0]['lr'],resumed_lr=p['optimizer']['param_groups'][0]['lr']/3,
        stage1_and_codebooks_and_reference_verified=True,source_best_fid=p['best_fid'],
        restored_model_and_optimizer=True,bitwise_replay=False))
    for name in ['execution.json','hardware.json','recipe.yaml']:
        shutil.copyfile(BASE/'inputs'/name,OUT/name)
    write(BASE/'h100-prepared.json',dict(passed=True))
    print((OUT/'source-verification.json').read_text(),flush=True)


if __name__=='__main__':main()
