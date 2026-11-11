"""Recover the verified 89.125-FID recipe; change execution and run identity only."""
from pathlib import Path
import difflib
import hashlib
import json
import shutil

BASE = Path('/tmp/laser-j32769-a8192-20261105/frozen')
OLD = Path('/tmp/laser-beam16-a100-20261105/frozen')
OUT = BASE.parent/'production-nvme'
PUBLIC = Path('/workspace/Projects/laser/outputs/imnet-j32769-a8192-native-e2to4-4a100-20261105')


def main():
    OUT.mkdir(exist_ok=True)
    PUBLIC.mkdir(parents=True, exist_ok=True)
    assert not (OUT/'launch.json').exists(), 'Inspect the existing continuation before preparing again'
    for name in ['training_acceleration.py', 'checkpoint_memory.py', 'loader_workers.py', 'media_policy.py']:
        shutil.copyfile(OLD/'runtime'/name, BASE/'runtime'/name)
    for name in ['stage1-model.pt','imagenet_256_train.npz']:
        p=BASE/'inputs'/name
        if not p.exists():p.hardlink_to(OLD/'inputs'/name)
    if not (BASE/'data').exists():(BASE/'data').symlink_to(OLD/'data',target_is_directory=True)
    p=BASE/'inputs/resume-checkpoint.pt'
    if not p.exists():p.hardlink_to(BASE.parent/'checkpoint-artifact/last.pt')
    for name in ['native-manifest.json','native-verification.json','training-input-verification.json','full-size-smoke-verification.json']:
        shutil.copyfile(BASE.parent/'source-artifact'/name, BASE/name)
    p=BASE/'train.py'
    code=p.read_text().replace("os.environ['LASER_STAGE2_OUTPUT']='/dev/shm/laser-joint32769-upstream-native-20261031/run'",
        "os.environ.setdefault('LASER_STAGE2_OUTPUT',str(root.parent/'production-nvme'))")
    code=code.replace('from src.training.rq_imagenet_shared import main',
        "from src import rq_upstream\nfrom runtime.training_acceleration import install\ninstall()\nfrom runtime.media_policy import install as media_policy\nmedia_policy()\nfrom src.training.rq_imagenet_shared import main")
    p.write_text(code)
    p=BASE/'source/src/training/rq_imagenet_shared.py'
    original=p.read_text();code=original
    def replace(old,new):
        nonlocal code
        assert code.count(old)==1,old
        code=code.replace(old,new)
    replace('import argparse\n','import argparse\nimport gc\nfrom functools import partial\n')
    replace('from runtime.amp_replay import ReplayWindow,recoverable_amp_step',
        'from runtime.amp_replay import ReplayWindow,recoverable_amp_step\nfrom runtime.checkpoint_memory import detach_cpu_optimizer_state,file_is_mapped\nfrom runtime.loader_workers import audit_loader_worker')
    replace('np.random.seed(seed);','np.random.seed(seed % (2**32));')
    replace("    args=p.parse_args()", """    p.add_argument('--run-id')
    args=p.parse_args()
    if args.run_id:
        RUN=args.run_id
        PROFILE.name='ImageNet joint32769 | original recipe | epoch2 to4 | 4 A100'""")
    replace("assert payload['run_id']==RUN and", "assert payload['run_id'] in (RUN,PROFILE.run_id) and")
    replace("        model.load_state_dict(payload['state_dict'],strict=True);optimizer.load_state_dict(payload['optimizer'])", """        model.load_state_dict(payload['state_dict'],strict=True);optimizer.load_state_dict(payload['optimizer'])
        from runtime.training_acceleration import configure_optimizer
        configure_optimizer(optimizer)
        detach_cpu_optimizer_state(optimizer)""")
    replace("        saved_rng=payload['rng_state_by_rank'][rank]\n        random.setstate(saved_rng['python']);np.random.set_state(saved_rng['numpy'])\n        torch.set_rng_state(saved_rng['cpu']);torch.cuda.set_rng_state(saved_rng['cuda'],device)\n        if hasattr(PROFILE,'coefficient_generator'):\n            PROFILE.coefficient_generator.set_state(saved_rng['coefficient'])\n        del payload", """        saved_world=len(payload['rng_state_by_rank'])
        saved_accumulation=int(payload['runtime']['accumulation_steps'])
        assert start_micro % saved_accumulation == 0
        assert global_step == completed_epoch*steps_per_epoch+start_micro//saved_accumulation
        start_micro=start_micro//saved_accumulation*accumulation
        assert all(int(s['step'])==global_step for s in optimizer.state.values())
        assert scheduler.after_scheduler.last_epoch==global_step
        assert scheduler.after_scheduler.T_max==62600
        if rank<saved_world:
            saved_rng=payload['rng_state_by_rank'][rank]
            random.setstate(saved_rng['python']);np.random.set_state(saved_rng['numpy'])
            torch.set_rng_state(saved_rng['cpu']);torch.cuda.set_rng_state(saved_rng['cuda'],device)
            if hasattr(PROFILE,'coefficient_generator'):
                PROFILE.coefficient_generator.set_state(saved_rng['coefficient'])
            del saved_rng
        else:
            seed_all(261001+1000003*global_step+1000000007*rank)
        del payload
        gc.collect()
        assert not file_is_mapped(args.resume.resolve()), 'Checkpoint mmap retained'""")
    replace("        pid=os.getpid(),**PROFILE.metadata())", """        pid=os.getpid(),restored_optimizer_age=global_step,
        optimizer_state_count_at_start=len(optimizer.state),
        scheduler_step_at_start=scheduler.after_scheduler.last_epoch,
        scheduler_horizon=scheduler.after_scheduler.T_max,
        gradient_clip_max_norm=float(config.optimizer.max_gn),
        training_autocast_dtype='float16',grad_scaler_enabled=scaler.is_enabled(),
        topology_change_is_not_bitwise_replay=True,**PROFILE.metadata())""")
    replace("os.environ['WANDB_API_KEY']=Path('/root/.config/laser/imagenet-dc-wandb.key').read_text().strip()",
        "assert os.environ.get('WANDB_API_KEY'), 'Supply credentials through launcher environment'")
    replace("                    teacher_temperature=PROFILE.target_temperature(),no_warmup_restart=True),", """                    teacher_temperature=PROFILE.target_temperature(),no_warmup_restart=True),
                current_execution=initial,""")
    replace("            run.summary['training/continued_from_step']=global_step", """            run.summary['training/continued_from_step']=global_step
            run.summary['continuation/current_global_step']=global_step
            run.config.update(dict(current_execution=initial),allow_val_change=True)""")
    replace("        persistent_workers=True,prefetch_factor=2)", """        persistent_workers=True,prefetch_factor=1,multiprocessing_context='spawn',
        worker_init_fn=partial(audit_loader_worker,audit_dir=str(folder/f'loader-workers-rank{rank}')))""")
    replace('                if not args.smoke_updates and (global_step==10 or global_step%250==0):', """                if not args.smoke_updates and global_step==initial['global_step']+10 and global_step%250:
                    checkpoint(epoch,epoch,absolute+1)
                    start=time.monotonic()
                if not args.smoke_updates and (global_step==10 or global_step%250==0):""")
    replace('                    if run and global_step>resume_logged_step:run.log(metrics)', """                    if run:
                        run.summary['continuation/current_global_step']=global_step
                        if global_step>resume_logged_step:run.log(metrics)""")
    compile(code,str(p),'exec');p.write_text(code)
    (PUBLIC/'runtime.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),code.splitlines(True),fromfile='archived/driver.py',tofile='a100/driver.py')))
    # Spawn workers must not inherit a parent image archive descriptor.
    shutil.copyfile(OLD/'source/src/training/rq_imagenet_runtime.py',BASE/'source/src/training/rq_imagenet_runtime.py')
    p=BASE/'source/src/training/joint_runtime_support.py';code=p.read_text()
    code=code.replace("'fresh':True", "'fresh':False,'source_epoch':2")
    code=code.replace('Plain upstream RQTransformer joint32769 fresh four epochs 20261031','Joint32769 native continuation epoch2 to4 4A100 20261105')
    p.write_text(code)
    execution=json.loads((BASE/'inputs/execution.json').read_text())
    execution.update(world_size=4,global_batch=2048,source_epoch=2,source_step=1252,
        hardware='4 NVIDIA A100-SXM4-80GB',checkpoint_storage='NVMe + verified W&B cloud commits',
        training_autocast_dtype='float16',gradient_clip_max_norm=1.0,
        training_targets='Archived stochastic soft targets with per-depth temperatures',
        topology_change_is_not_bitwise_replay=True,automatic_restart=True)
    (BASE/'inputs/execution.json').write_text(json.dumps(execution,indent=2)+'\n')
    for n in ['status.json','launch.json','training.log','checkpoint.json','supervisor.log']:
        p=PUBLIC/n
        if not p.is_symlink():p.symlink_to(OUT/n)
    p=PUBLIC/'live-nvme'
    if not p.is_symlink():p.symlink_to(OUT,target_is_directory=True)
    print('Original recipe preserved; four-A100 runtime prepared.',flush=True)


if __name__=='__main__':main()
