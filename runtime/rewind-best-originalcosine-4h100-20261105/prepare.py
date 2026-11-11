"""Restore the best weights and Adam state, then recover the unscaled cosine amplitude."""
import hashlib,json,shutil
from pathlib import Path
import torch
import yaml

ROOT=Path('/workspace/Projects/laser')
SOURCE=Path('/tmp/laser-best-continue100-4h100-20261105/frozen')
BASE=Path('/tmp/laser-best-originalcosine-4h100-20261105/frozen')
RUN='imnet-j16384-best14574-originalcosine-continue100-4h100-20261105'
OUT=ROOT/'outputs'/RUN
ARTIFACT='helloimlixin-rutgers/laser/imnet-j16384-a4096-fid1481-resume100-4h100-20261104-checkpoint:v26'


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,default=str)+'\n')


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    assert not BASE.exists(),'Inspect existing runtime before relaunch'
    BASE.mkdir(parents=True)
    for folder in ['source','upstream','runtime']:
        shutil.copytree(SOURCE/folder,BASE/folder,ignore=shutil.ignore_patterns('__pycache__'))
    (BASE/'inputs').mkdir()
    for path in (SOURCE/'inputs').iterdir():
        if not path.is_file():continue
        if path.stat().st_size>50*1024**2:(BASE/'inputs'/path.name).hardlink_to(path)
        else:shutil.copyfile(path,BASE/'inputs'/path.name)
    for name in ['data','torch-cache']:(BASE/name).symlink_to(SOURCE/name,target_is_directory=True)
    shutil.copyfile(SOURCE/'train.py',BASE/'train.py')
    shutil.copyfile(SOURCE/'continuation-runtime.patch',BASE/'continuation-runtime.patch')
    p=torch.load(BASE/'inputs/resume-checkpoint.pt',map_location='cpu',mmap=True,weights_only=False)
    assert p['global_step']==45698 and p['next_epoch']==73 and p['next_microbatch']==0
    assert len(p['optimizer']['state'])==780
    assert {int(s['step']) for s in p['optimizer']['state'].values()}=={45698}
    assert p['runtime']['lr_amplitude_factor']==.15
    assert p['scheduler']['after']['last_epoch']==45698 and p['scheduler']['after']['T_max']==62600
    for name,key in [('stage1-model.pt','stage1_checkpoint_sha256'),('joint-codebook.pt','joint_codebook_sha256'),('reconstruction-codebook.pt','reconstruction_codebook_sha256')]:
        assert sha(BASE/'inputs'/name)==p[key]
    baseline=json.loads((BASE/'inputs/verified-four-gpu-baseline.json').read_text())
    assert baseline['global_step']==45698 and baseline['world_size']==4 and baseline['generated_images']==50000
    assert baseline['sampler']==p['native_sampling']
    assert sha(BASE/'inputs/imagenet_256_train.npz')==baseline['reference_sha256']
    recipe=dict(p['recipe'])
    recipe.update(run_id=RUN,name='ImageNet joint16384 | FID14.5739 best rewind | ORIGINAL unscaled cosine | epoch100 | 4 H100',
        lineage=dict(source_run_id=p['run_id'],source_step=45698,source_epoch=73,source_artifact=ARTIFACT,
            reason='User confirmed rewind the FID14.5739 best weights and Adam state onto the original unscaled cosine schedule (1.0x), retaining the update clock through epoch100.'))
    (BASE/'inputs/recipe.yaml').write_text(yaml.safe_dump(recipe,sort_keys=False))
    source_lr=float(p['optimizer']['param_groups'][0]['lr']);ratio=1/.15;desired=source_lr*ratio
    execution=json.loads((BASE/'inputs/execution.json').read_text())
    execution.update(lr_amplitude_factor=1.0,source_lr_preserved=False,source_lr=source_lr,
        requested_lr_at_rewind=desired,lr_ratio_vs_saved=ratio,original_unscaled_cosine=True,
        original_cosine_base_lr=.0005,source_adam_moments_preserved=True,source_optimizer_age_preserved=True,
        cosine_clock_preserved=True,repair_trial=False,quality_early_stop=False,quality_pilot_enabled=False,
        quality_guard=dict(enabled=False,reason='User requested epoch100 continuation and confirmed original unscaled cosine'),
        adaptive_lr=None,source_checkpoint_artifact=ARTIFACT)
    write(BASE/'inputs/execution.json',execution)
    source=dict(path=str(BASE/'inputs/resume-checkpoint.pt'),global_step=45698,
        bytes=(BASE/'inputs/resume-checkpoint.pt').stat().st_size,state='COMMITTED',artifact=ARTIFACT,
        historical_one_gpu_fid=14.573915136162327,historical_one_gpu_is=94.00654602050781,
        matched_four_gpu_fid=baseline['fid'],matched_four_gpu_is=baseline['inception_score'])
    write(BASE/'run-output/original-best.json',source)
    write(BASE/'run-output/persistent-train/checkpoints/best.json',dict(source,best_fid=baseline['fid']))
    write(OUT/'original-best.json',source)
    support=BASE/'source/src/training/joint_runtime_support.py'
    support.write_text(support.read_text().replace("title='ImageNet FID14.5739 best rewind continue100 4H100 20261105'",
        "title='ImageNet FID14.5739 original unscaled cosine continue100 4H100 20261105'"))
    proof=dict(passed=True,source_artifact=ARTIFACT,source_step=45698,source_epoch=73,
        next_microbatch=0,adam_states=780,all_adam_steps=45698,cosine_step=45698,cosine_horizon=62600,
        saved_lr=source_lr,restored_unscaled_lr=desired,lr_amplitude_factor=1.0,
        lr_ratio_vs_saved=ratio,original_cosine_base_lr=.0005,adam_moments_preserved=True,
        target_epoch=100,target_step=62600,world_size=4,microbatch=128,accumulation_steps=4,
        global_batch=2048,quality_pilot_removed=True,quality_stopping_removed=True,
        cached_full50k_baseline_verified=True,tokenizer_codebooks_and_reference_verified=True)
    write(BASE/'continuation-verification.json',proof);write(OUT/'source-verification.json',proof)
    for name in ['recipe.yaml','execution.json','hardware.json','verified-four-gpu-baseline.json']:
        shutil.copyfile(BASE/'inputs'/name,OUT/name)
    shutil.copyfile(Path(__file__),BASE/'originalcosine-prepare.py')
    print(json.dumps(proof,indent=2),flush=True)


if __name__=='__main__':main()
