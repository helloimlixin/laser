"""Adapt the archived run's execution without changing its recipe or weights."""
from pathlib import Path
import difflib
import json
import shutil

BASE = Path('/tmp/laser-beam16-a100-20261105/frozen')
OUT = Path('/workspace/Projects/laser/outputs/imnet-beam16-resume-4a100-20261105')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    path = BASE / 'source/src/training/rq_imagenet_shared.py'
    original = path.read_text()
    code = original
    old = """        if saved_world != world and start_micro != 0:
            raise RuntimeError('World-size changes require an epoch-boundary checkpoint')"""
    new = """        # Cursors count microbatches; convert through completed GLOBAL updates.
        # ExactGlobalBatchSampler partitions the same deterministic global order
        # under either topology. Per-image augmentation is seeded by epoch/index.
        saved_accumulation = int(payload['runtime']['accumulation_steps'])
        assert payload['runtime']['global_batch'] == 2048
        assert start_micro % saved_accumulation == 0
        completed_updates = start_micro // saved_accumulation
        assert global_step == completed_epoch * steps_per_epoch + completed_updates
        source_micro = start_micro
        start_micro = completed_updates * accumulation
        assert all(int(s['step']) == global_step for s in optimizer.state.values())
        assert scheduler.after_scheduler.last_epoch == global_step
        assert scheduler.after_scheduler.T_max == 62600
        cursor_conversion = dict(source_world=saved_world, world_size=world,
            source_accumulation=saved_accumulation, accumulation_steps=accumulation,
            source_microbatch_cursor=source_micro, microbatch_cursor=start_micro,
            epoch=completed_epoch, completed_updates_in_epoch=completed_updates,
            global_step=global_step, global_batch=2048,
            global_sample_cursor_preserved=True, bitwise_replay=False)
        write_json(folder/f'cursor-conversion-rank{rank}.json', cursor_conversion)"""
    assert code.count(old) == 1
    code = code.replace(old, new)
    old = "                if not args.smoke_updates and (global_step==10 or global_step%250==0):"
    new = """                if not args.smoke_updates and global_step == initial['global_step'] + 10 and global_step % 250:
                    checkpoint(epoch,epoch,absolute+1)
                    start=time.monotonic()
                if not args.smoke_updates and (global_step==10 or global_step%250==0):"""
    assert code.count(old) == 1
    code = code.replace(old, new)
    old = """                    assert PROFILE.digest(aux)==before
                    checkpoint(epoch,epoch,absolute+1)"""
    new = """                    assert PROFILE.digest(aux)==before
                    fingerprints = [None] * world
                    dist.all_gather_object(fingerprints, parameter_fingerprint(model))
                    assert len(set(fingerprints)) == 1, 'DDP replicas diverged'
                    write_json(folder/f'replica-verification-rank{rank}.json',
                        dict(passed=True, world_size=world, fingerprints=fingerprints))
                    checkpoint(epoch,epoch,absolute+1)"""
    assert code.count(old) == 1
    code = code.replace(old, new)
    compile(code, str(path), 'exec')
    path.write_text(code)
    path = BASE/'inputs/execution.json'
    execution = json.loads(path.read_text())
    execution.update(initial_global_step=11000, initial_epoch=17,
        source_epoch=17, source_step=11000, world_size=4, resume_world_size=1,
        microbatch=64, accumulation_steps=8, hardware='4 NVIDIA A100-SXM4-80GB',
        training_attention='SDPA for training; native cached sampling')
    path.write_text(json.dumps(execution, indent=2)+'\n')
    (OUT/'runtime.patch').write_text(''.join(difflib.unified_diff(
        original.splitlines(True), code.splitlines(True),
        fromfile='archived/rq_imagenet_shared.py', tofile='a100/rq_imagenet_shared.py')))
    path = BASE/'train.py'
    code = path.read_text().replace(
        "os.environ['LASER_STAGE2_OUTPUT']=str(root/'run-output')",
        "os.environ.setdefault('LASER_STAGE2_OUTPUT',str(root/'run-output'))")
    path.write_text(code)
    path = BASE/'source/src/training/joint_runtime_support.py'
    path.write_text(path.read_text().replace('training_from_epoch=10', 'training_from_epoch=17'))
    for src, dst in [('checkpoint-artifact/last.pt', 'inputs/resume-checkpoint.pt'),
                     ('stage1-artifact/best_rfid_slot1_model.pt', 'inputs/stage1-model.pt')]:
        target = BASE/dst
        if not target.exists():
            target.hardlink_to(BASE.parent/src)
    for name in ['best-inception.json', 'workspace.json']:
        src = Path('/workspace/Projects/laser/outputs/imnet-j16384-a4096-beam16-rfid451-scratch100-2b300-20261104')/name
        if src.is_file() and src.stat().st_size:
            shutil.copyfile(src, OUT/name)
    for name in ['kernel-verification.json', 'cursor-verification.json', 'latest-artifact.json']:
        shutil.copyfile(BASE.parent/name, OUT/name)
    best = json.loads((BASE.parent/'best-inception-artifact.json').read_text())
    (OUT/'best-inception.json').write_text(json.dumps(dict(
        inception_score=best['metadata']['evaluation']['inception_score'],
        global_step=best['metadata']['global_step'],
        artifact=best['artifact'], evaluation=best['metadata']['evaluation']), indent=2)+'\n')
    print('Runtime prepared', flush=True)


if __name__ == '__main__':
    main()
