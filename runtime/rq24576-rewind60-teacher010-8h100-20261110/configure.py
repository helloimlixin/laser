"""Finish replay-specific trainer changes after installing the isolated runtime."""
import json
from pathlib import Path
import py_compile
import shutil
from install import install

HERE=Path(__file__).resolve().parent


def main():
    plan=json.loads((HERE/'plan.json').read_text());install(plan)
    base=Path(plan['base']);out=Path(plan['output'])
    path=base/'source/src/training/rq_imagenet_shared.py';s=path.read_text()
    old="""        if payload['run_id']==PROFILE.parent_run_id:
            restored_evaluation=dict(PROFILE.refinement['baseline'])
            restored_best_fid=restored_evaluation['fid']"""
    new="""        if payload['run_id']==PROFILE.parent_run_id:
            # The retained source has a different sampler; do not mix FID curves.
            restored_evaluation=None
            restored_best_fid=math.inf"""
    assert old in s;s=s.replace(old,new,1)
    s=s.replace("minimum_lr=PROFILE.refinement['lr']['min_lr'],warmup_epochs=1,",
        "minimum_lr=PROFILE.refinement['lr']['min_lr'],warmup_epochs=0,",1)
    old="if not args.smoke_updates and ((epoch+1-PROFILE.refinement['source_epoch'])%2==0 or epoch+1==args.epochs):"
    new="if not args.smoke_updates and epoch+1>PROFILE.refinement['requested_rewind_epoch'] and ((epoch+1-PROFILE.refinement['source_epoch'])%2==0 or epoch+1==args.epochs):"
    assert old in s;s=s.replace(old,new,1)
    old="""                if not verified_targets:
                    write_json(folder/f'targets-rank{rank}.json',PROFILE.verify_targets(aux,targets,codes))
                    verified_targets=True"""
    new="""                temperature=PROFILE.target_temperature()
                if not verified_targets or getattr(PROFILE,'audited_teacher_temperature',None)!=temperature:
                    proof=PROFILE.verify_targets(aux,targets,codes)
                    write_json(folder/f'targets-rank{rank}.json',proof)
                    write_json(folder/f'targets-temperature-{temperature}-rank{rank}.json',proof)
                    PROFILE.audited_teacher_temperature=temperature
                    verified_targets=True"""
    assert old in s;s=s.replace(old,new,1);path.write_text(s)
    comparison=Path('/workspace/Projects/laser/outputs/imagenet-rq24576-vs-original-latenterror-20261110/report.json')
    shutil.copyfile(comparison,base/'inputs/teacher010-matched-10k-comparison.json')
    for file in [path,base/'runtime/refinement.py',base/'runtime/native_recipe.py']:py_compile.compile(str(file),doraise=True)
    print(json.dumps(dict(configured=True,teacher_switch_epoch=60,first_fid_epoch=62,original_lr_schedule_preserved=True)),flush=True)


if __name__=='__main__':main()
