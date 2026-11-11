"""Restore the FID14.5739 checkpoint and honor the explicit epoch-100 continuation."""
import difflib,hashlib,json,shutil,zipfile
from pathlib import Path
import torch
import yaml

ROOT=Path('/workspace/Projects/laser')
BASE=Path('/tmp/laser-best-continue100-4h100-20261105/frozen')
OLD=Path('/tmp/laser-resume-h100-20261105/frozen')
BEST=Path('/tmp/laser-best-rewind-4h100-20261105/frozen')
RUN='imnet-j16384-best14574-rewind-continue100-4h100-20261105'
OUT=ROOT/'outputs'/RUN
ARTIFACT='helloimlixin-rutgers/laser/imnet-j16384-a4096-fid1481-resume100-4h100-20261104-checkpoint:v26'


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,default=str)+'\n')


def sha(path):
    with path.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def main():
    assert not BASE.exists(),'Inspect existing runtime before relaunch'
    with zipfile.ZipFile(OLD.parent/'source-artifact/source.zip') as archive:archive.extractall(BASE)
    for name in ['data','torch-cache']:(BASE/name).symlink_to(OLD/name,target_is_directory=True)
    (BASE/'inputs/stage1-model.pt').hardlink_to(OLD/'inputs/stage1-model.pt')
    (BASE/'inputs/resume-checkpoint.pt').hardlink_to(BEST/'inputs/resume-checkpoint.pt')
    (BASE/'inputs/matched-baseline-evaluation.json').rename(BASE/'inputs/historical-one-gpu-baseline.json')
    p=torch.load(BASE/'inputs/resume-checkpoint.pt',map_location='cpu',mmap=True,weights_only=False)
    assert p['global_step']==45698 and p['next_epoch']==73 and p['next_microbatch']==0
    assert len(p['optimizer']['state'])==780
    assert {int(s['step']) for s in p['optimizer']['state'].values()}=={45698}
    assert p['scheduler']['after']['last_epoch']==45698 and p['scheduler']['after']['T_max']==62600
    assert p['runtime']['world_size']==4 and p['runtime']['microbatch']==128 and p['runtime']['accumulation_steps']==4
    assert p['runtime']['lr_amplitude_factor']==.15
    for name,key in [('stage1-model.pt','stage1_checkpoint_sha256'),('joint-codebook.pt','joint_codebook_sha256'),('reconstruction-codebook.pt','reconstruction_codebook_sha256')]:
        assert sha(BASE/'inputs'/name)==p[key]
    assert sha(BASE/'inputs/imagenet_256_train.npz')==p['recipe']['evaluation']['reference_sha256']
    baseline=json.loads((BEST/'run-output/matched-baseline.json').read_text())
    reproduction=json.loads((BEST/'run-output/baseline-reproduction-gate.json').read_text())
    assert reproduction['accepted'] and reproduction['fid_delta']==reproduction['is_delta']==0
    assert baseline['global_step']==45698 and baseline['world_size']==4 and baseline['generated_images']==50000
    assert baseline['sampler']==p['native_sampling'] and baseline['seed']==261001
    assert baseline['reference_sha256']==sha(BASE/'inputs/imagenet_256_train.npz')
    recipe=dict(p['recipe'])
    recipe.update(run_id=RUN,name='ImageNet joint16384 | rewind FID14.5739 best | continue epoch100 | 4 H100',
        lineage=dict(source_run_id=p['run_id'],source_step=45698,source_epoch=73,source_artifact=ARTIFACT,
            reason='User explicitly requested rewind from the original FID14.5739 best and uninterrupted quality-monitored training through epoch100.'))
    (BASE/'inputs/recipe.yaml').write_text(yaml.safe_dump(recipe,sort_keys=False))
    write(BASE/'inputs/sampling-continuation.json',dict(source_run_id=p['run_id'],source_global_step=45698,
        source_checkpoint_artifact=ARTIFACT,previous_sampler=p['native_sampling'],sampler=p['native_sampling']))
    write(BASE/'inputs/verified-four-gpu-baseline.json',baseline)
    execution=json.loads((BASE/'inputs/execution.json').read_text())
    execution.update(hardware='4 NVIDIA H100 80GB HBM3',world_size=4,microbatch=128,accumulation_steps=4,
        global_batch=2048,source_run_id=p['run_id'],source_step=45698,source_epoch=73,
        source_checkpoint_artifact=ARTIFACT,source_saved_world_size=4,lr_amplitude_factor=.15,
        source_lr_preserved=True,training_attention='original native bmm',repair_trial=False,
        quality_guard=dict(enabled=False,reason='User explicitly requested continuation until epoch100'),
        quality_early_stop=False,quality_pilot_enabled=False,adaptive_lr=None,trial_stop_epoch=100,
        baseline_measurement_origin='Reused exact full50k four-H100 baseline verified on this machine before the prior eight-update trial',
        matched_full50k_initial_evaluation=True,eval_every_epochs=1,
        validated_initial_fid=baseline['fid'],validated_initial_is=baseline['inception_score'],
        is_preservation_floor=baseline['inception_score']-.5,
        historical_one_gpu_best_fid=14.573915136162327,historical_one_gpu_best_is=94.00654602050781,
        best_checkpoint_policy='Retain original best artifact; save improved matched FID with IS preserved; never interrupt epoch100 for a quality result',
        rng_policy='Restore all four source rank RNG streams, Adam, cosine clock, epoch73 sample cursor and original global batch')
    write(BASE/'inputs/execution.json',execution)
    write(BASE/'inputs/hardware.json',dict(hardware='4 H100 80GB',world_size=4,microbatch=128,
        accumulation_steps=4,global_batch=2048))
    source=dict(path=str(BASE/'inputs/resume-checkpoint.pt'),global_step=45698,
        bytes=(BASE/'inputs/resume-checkpoint.pt').stat().st_size,state='COMMITTED',artifact=ARTIFACT,
        historical_one_gpu_fid=14.573915136162327,historical_one_gpu_is=94.00654602050781,
        matched_four_gpu_fid=baseline['fid'],matched_four_gpu_is=baseline['inception_score'])
    write(BASE/'run-output/original-best.json',source)
    write(BASE/'run-output/persistent-train/checkpoints/best.json',dict(source,best_fid=baseline['fid']))
    write(OUT/'original-best.json',source)
    path=BASE/'source/src/training/rq_imagenet_shared.py';before=path.read_text();code=before
    marker='    ddp=DDP(model,device_ids=[local],broadcast_buffers=False,gradient_as_bucket_view=True,bucket_cap_mb=100)'
    assert code.count(marker)==1
    injection='''    if args.resume and global_step==45698:
        measured=json.loads((BASE/'inputs/verified-four-gpu-baseline.json').read_text())
        assert measured['global_step']==global_step and measured['world_size']==world==4
        assert measured['generated_images']==50000 and measured['seed']==261001
        assert measured['sampler']==PROFILE.sampling_settings() and PROFILE.generation_batch==64
        assert measured['reference_sha256']==sha256(BASE/'inputs/imagenet_256_train.npz')
        restored_evaluation={key:measured[key] for key in ('fid','inception_score','inception_score_std','sampler')}
        restored_best_fid=restored_evaluation['fid']
        inherited_evaluation_world_size=4
        verified_baseline_reused=True
'''
    code=code.replace(marker,injection+marker)
    code=code.replace("measurement_hardware='B200' if inherited_baseline else 'B300',","measurement_hardware='4 H100',")
    code=code.replace("continuation_hardware='B300',","continuation_hardware='4 H100',")
    # The inherited point is a verified earlier measurement, not a new evaluation.
    code=code.replace("remeasured_on_continuation_hardware=not inherited_baseline,", "remeasured_on_continuation_hardware=not inherited_baseline,\n                verified_existing_same_hardware_measurement=verified_baseline_reused,")
    # Remove the pilot branch, rather than relying on a cached repair-trial flag.
    start=code.index("                elif not args.smoke_updates and global_step-initial['global_step']==64")
    end=code.index('                if args.smoke_updates and global_step>=args.smoke_updates:',start)
    code=code[:start]+code[end:]
    # Keep evaluation and selection, with the quality-exit path removed for this explicit continuation.
    start=code.index('        if evaluation is not None:\n',code.index('        start_micro=0\n'))
    end=code.index('        start=time.monotonic()\n',start)
    code=code[:start]+code[end:]
    assert 'repair_trial_rejected' not in code and 'quality_bad_evaluations >= 2' not in code
    path.write_text(code)
    support=BASE/'source/src/training/joint_runtime_support.py'
    support.write_text(support.read_text().replace("title='ImageNet Joint16384 best14574 continue100 1B300 20261104'",
        "title='ImageNet FID14.5739 best rewind continue100 4H100 20261105'"))
    patch=''.join(difflib.unified_diff(before.splitlines(True),code.splitlines(True),fromfile='original/rq_imagenet_shared.py',tofile='continue100/rq_imagenet_shared.py'))
    (BASE/'continuation-runtime.patch').write_text(patch);(OUT/'runtime.patch').write_text(patch)
    proof=dict(passed=True,source_artifact=ARTIFACT,source_step=45698,source_epoch=73,next_microbatch=0,
        adam_states=780,all_adam_steps=45698,cosine_step=45698,cosine_horizon=62600,
        source_lr=p['optimizer']['param_groups'][0]['lr'],source_lr_preserved=True,
        target_epoch=100,target_step=62600,world_size=4,microbatch=128,accumulation_steps=4,
        global_batch=2048,all_source_rng_streams_restored=True,quality_pilot_removed=True,
        quality_stopping_removed=True,failed_plateau_schedule_absent=True,
        tokenizer_codebooks_and_reference_sha256_verified=True,cached_full50k_baseline_verified=True)
    write(BASE/'continuation-verification.json',proof);write(OUT/'source-verification.json',proof)
    for name in ['recipe.yaml','execution.json','hardware.json','verified-four-gpu-baseline.json']:
        shutil.copyfile(BASE/'inputs'/name,OUT/name)
    shutil.copyfile(Path(__file__),BASE/'continue100-prepare.py')
    print(json.dumps(proof,indent=2),flush=True)


if __name__=='__main__':main()
