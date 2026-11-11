"""Apply a fixed-checkpoint FID50k winner to generation only, then resume."""
from pathlib import Path
import difflib
import hashlib
import json
import shutil
from calibrated_sampling import validate

ROOT=Path('/workspace/Projects/laser')
BASE=Path('/tmp/laser-beam451-stage2-4h100-20261107')
EXPERIMENT=ROOT/'outputs/imagenet-beam451-aligned-joint16k-scratch100-4h100-20261107/sampler-calibration-20261108'


def install(evidence):
    validate(evidence)
    assert evidence['training_recipe_sha256']==hashlib.sha256((BASE/'inputs/recipe.yaml').read_bytes()).hexdigest()
    recipe=BASE/'source/src/training/laser_joint_recipe.py';before=recipe.read_text()
    if 'calibrated_sampling' not in before:
        old='    def sampling_settings(self):\n        return dict('
        assert before.count(old)==1
        after=before.replace(old,"    def sampling_settings(self):\n        from runtime.calibrated_sampling import settings\n        calibrated=settings(self)\n        if calibrated is not None:return calibrated\n        return dict(")
        old="        assert payload['joint_codebook_sha256'] == self.codebook_sha256"
        assert after.count(old)==1
        after=after.replace(old,old+"\n        from runtime.calibrated_sampling import verify_resume\n        verify_resume(self,payload)")
    else:after=before
    driver=BASE/'source/src/training/rq_imagenet_shared.py';driver_before=driver.read_text()
    old='        run.config.update(rng_metadata,allow_val_change=True)';assert driver_before.count(old)==1
    driver_after=driver_before.replace(old,old+"\n        from runtime.calibrated_sampling import log_settings\n        log_settings(run,PROFILE)")
    old="    write_json(folder/f'startup-rank{rank}.json',initial)"
    assert driver_after.count(old)==1
    driver_after=driver_after.replace(old,old+'''
    if args.resume and (BASE/'inputs/generation-sampling-override.json').exists():
        from runtime.generation_interface import verify_generation
        saved_cpu_rng=torch.get_rng_state().clone()
        saved_cuda_rng=torch.cuda.get_rng_state(device).clone()
        sampler_probe=verify_generation(PROFILE,model,aux,device)
        assert torch.equal(saved_cpu_rng,torch.get_rng_state())
        assert torch.equal(saved_cuda_rng,torch.cuda.get_rng_state(device))
        write_json(folder/f'sampler-generation-interface-rank{rank}.json',dict(sampler_probe,
            sampling=PROFILE.sampling_settings(),generation_batch=PROFILE.generation_batch,
            restored_step=global_step,training_rng_unchanged=True))''')
    support=BASE/'source/src/training/joint_runtime_support.py';support_before=support.read_text()
    support_after=support_before.replace("sampler['top_k'][0]", "sampler['top_k']").replace("sampler['top_p'][0]", "sampler['top_p']")
    for path,content in [(recipe,after),(driver,driver_after),(support,support_after)]:compile(content,str(path),'exec')
    patch=''
    for path,old,new in [(recipe,before,after),(driver,driver_before,driver_after),(support,support_before,support_after)]:
        patch+=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='before/'+path.name,tofile='after/'+path.name))
    shutil.copyfile(Path(__file__).with_name('calibrated_sampling.py'),BASE/'runtime/calibrated_sampling.py')
    for path,content in [(recipe,after),(driver,driver_after),(support,support_after)]:path.write_text(content)
    path=BASE/'inputs/generation-sampling-override.json';tmp=path.with_suffix('.pending');tmp.write_text(json.dumps(evidence,indent=2)+'\n');tmp.replace(path)
    (EXPERIMENT/'generation-only.patch').write_text(patch)
    (EXPERIMENT/'applied.json').write_text(json.dumps(dict(sampling=evidence['sampling'],source_step=evidence['source_step'],
        fid=evidence['fid'],baseline_fid=evidence['baseline_fid'],training_recipe_unchanged=True),indent=2)+'\n')
