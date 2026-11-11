"""Rewind the complete optimizer trajectory to the immutable epoch-73 best."""
import difflib,hashlib,json,shutil
from pathlib import Path
import torch
import yaml

ROOT=Path('/workspace/Projects/laser')
BASE=Path('/tmp/laser-best-rewind-4h100-20261105/frozen')
HERE=Path(__file__).resolve().parent
RUN='imnet-j16384-best14574-rewind-gated-4h100-20261105'
OUT=ROOT/'outputs'/RUN
ARTIFACT='helloimlixin-rutgers/laser/imnet-j16384-a4096-fid1481-resume100-4h100-20261104-checkpoint:v26'


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,default=str)+'\n')


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    assert not (BASE/'rewind-prepared.json').exists()
    p=torch.load(BASE/'inputs/resume-checkpoint.pt',map_location='cpu',mmap=True,weights_only=False)
    assert p['global_step']==45698 and p['next_epoch']==73 and p['next_microbatch']==0
    assert len(p['optimizer']['state'])==780
    assert {int(s['step']) for s in p['optimizer']['state'].values()}=={45698}
    assert p['scheduler']['after']['last_epoch']==45698 and p['scheduler']['after']['T_max']==62600
    assert p['runtime']['world_size']==4 and p['runtime']['microbatch']==128 and p['runtime']['accumulation_steps']==4
    assert p['runtime']['lr_amplitude_factor']==0.15
    for file,key in [('stage1-model.pt','stage1_checkpoint_sha256'),('joint-codebook.pt','joint_codebook_sha256'),('reconstruction-codebook.pt','reconstruction_codebook_sha256')]:
        assert sha(BASE/'inputs'/file)==p[key]
    assert sha(BASE/'inputs/imagenet_256_train.npz')==p['recipe']['evaluation']['reference_sha256']
    recipe=dict(p['recipe'])
    recipe.update(run_id=RUN,name='ImageNet joint16384 | rewind best epoch73 | matched quality gate | 4 H100',
        lineage=dict(source_run_id=p['run_id'],source_step=45698,source_epoch=73,source_artifact=ARTIFACT,
            reason='User requested rewind from preserved best after degraded continuation; restore source weights, Adam, RNG and LR without the failed plateau schedule.'))
    (BASE/'inputs/recipe.yaml').write_text(yaml.safe_dump(recipe,sort_keys=False))
    write(BASE/'inputs/sampling-continuation.json',dict(source_run_id=p['run_id'],source_global_step=45698,
        source_checkpoint_artifact=ARTIFACT,previous_sampler=p['native_sampling'],sampler=p['native_sampling']))
    old=json.loads((BASE/'inputs/execution.json').read_text())
    old.update(hardware='4 NVIDIA H100 80GB HBM3',world_size=4,microbatch=128,accumulation_steps=4,
        global_batch=2048,source_run_id=p['run_id'],source_step=45698,source_epoch=73,
        source_saved_world_size=4,source_checkpoint_artifact=ARTIFACT,lr_amplitude_factor=.15,
        training_attention='original native bmm; sampling unchanged',repair_trial=True,
        pilot_optimizer_updates=8,matched_full50k_initial_evaluation=True,
        baseline_measurement_origin='Fresh matched 4-H100 FID50k before the first gradient update',
        quality_guard=dict(enabled=True,maximum_fid_increase_vs_matched_baseline=.10,
            maximum_is_drop=.20,stop_after_failed_evaluations=1),
        rng_policy='Restore all four original rank RNG streams; same world, microbatch, global batch and sample cursor',
        best_checkpoint_policy='Quality gate failure stops training and restores the active checkpoint pointer to the best committed artifact',
        baseline_reproduction_gate=dict(maximum_absolute_fid_delta=.10,maximum_absolute_is_delta=.50),
        pilot_acceptance=dict(minimum_fid_improvement=.05,maximum_is_drop=.20),
        adaptive_lr=None)
    write(BASE/'inputs/execution.json',old)
    write(BASE/'inputs/hardware.json',dict(hardware='4 H100 80GB',world_size=4,microbatch=128,
        accumulation_steps=4,global_batch=2048))
    write(BASE/'inputs/best-four-gpu-reference.json',p['evaluation'])
    initial=dict(path=str(BASE/'inputs/resume-checkpoint.pt'),global_step=45698,
        bytes=(BASE/'inputs/resume-checkpoint.pt').stat().st_size,state='COMMITTED',artifact=ARTIFACT,
        fid=p['evaluation']['fid'],inception_score=p['evaluation']['inception_score'],
        historical_one_gpu_fid=14.573915136162327,historical_one_gpu_is=94.00654602050781)
    write(BASE/'run-output/persistent-train/checkpoints/best.json',initial)
    write(BASE/'run-output/rewind-source.json',initial)
    write(OUT/'rewind-source.json',initial)
    shutil.copyfile(HERE/'rewind_gate.py',BASE/'runtime/rewind_gate.py')
    path=BASE/'source/src/training/rq_imagenet_shared.py'
    before=path.read_text();s=before
    # The source scheduler amplitude is retained exactly.
    s=s.replace("measurement_hardware='B200' if inherited_baseline else 'B300',",
        "measurement_hardware='historical' if inherited_baseline else '4 H100',")
    s=s.replace("continuation_hardware='B300',", "continuation_hardware='4 H100',")
    oldfloor="        os.environ['LASER_PRESERVE_INCEPTION_MIN']=os.environ.get('LASER_PRESERVE_INCEPTION_MIN', str(comparison_baseline['inception_score']-.5))"
    assert s.count(oldfloor)==1
    s=s.replace(oldfloor,"        os.environ['LASER_PRESERVE_INCEPTION_MIN']=str(comparison_baseline['inception_score']-.20)")
    # Exact whole-line anchor: this gate runs once after the fresh baseline and before training.
    marker="    for epoch in range(completed_epoch,args.epochs):\n"
    assert s.count(marker)==1
    gate='''    if args.new_wandb_run and not args.smoke_updates:
        from runtime.rewind_gate import baseline_decision,record_rewind
        reference=json.loads((BASE/'inputs/best-four-gpu-reference.json').read_text())
        reproduction=baseline_decision(reference,comparison_baseline)
        if rank==0:
            write_json(OUT/'baseline-reproduction-gate.json',dict(reproduction,
                historical_four_gpu_reference=reference,fresh_matched_baseline=comparison_baseline,
                no_optimizer_updates_before_baseline=True))
            if run:
                run.summary.update({'rewind/baseline_passed':reproduction['accepted'],
                    'rewind/source_step':45698,'rewind/pilot_updates':8})
                run.save(str(OUT/'baseline-reproduction-gate.json'),base_path=str(OUT),policy='now')
        if not reproduction['accepted']:
            if rank==0:
                record_rewind(OUT,'Restored best did not reproduce the four-GPU reference; no optimizer updates allowed')
                if run:run.summary['training/status']='rewound_baseline_mismatch';run.finish()
            writer.shutdown();dist.barrier();dist.destroy_process_group();return

'''
    s=s.replace(marker,gate+marker)
    needle="global_step-initial['global_step']==64 and PROFILE.metadata()['execution'].get('repair_trial')"
    assert s.count(needle)==1
    s=s.replace(needle,"global_step-initial['global_step']==8 and PROFILE.metadata()['execution'].get('repair_trial')")
    accepted="""                    accepted=(evaluation['fid']<=comparison_baseline['fid']-.05 and
                              evaluation['inception_score']>=comparison_baseline['inception_score']-.5)"""
    assert s.count(accepted)==1
    s=s.replace(accepted,"                    from runtime.rewind_gate import pilot_decision\n                    accepted=pilot_decision(comparison_baseline,evaluation)['accepted']")
    s=s.replace("                    accepted=pilot_decision(comparison_baseline,evaluation)['accepted']\n",
        "                    accepted=pilot_decision(comparison_baseline,evaluation)['accepted']\n                    if accepted:\n                        quality_best_fid=evaluation['fid'];quality_best_is=evaluation['inception_score']\n                        os.environ['LASER_PRESERVE_INCEPTION_MIN']=str(max(float(os.environ['LASER_PRESERVE_INCEPTION_MIN']),quality_best_is-.20))\n")
    s=s.replace("criterion='FID lower by at least0.05 and IS drop at most0.5'", "criterion='FID lower by at least0.05 and IS drop at most0.20'")
    needle="                            if run:run.summary['training/status']='repair_trial_rejected';run.finish()"
    assert s.count(needle)==1
    s=s.replace(needle,"                            from runtime.rewind_gate import record_rewind\n                            record_rewind(OUT,'Eight-update pilot failed the matched FID/IS improvement gate')\n                            if run:run.summary['training/status']='rewound_to_best_after_rejected_trial';run.finish()")
    s=s.replace("evaluation['fid']>comparison_baseline['fid']+float(guard['maximum_fid_increase_vs_matched_baseline'])", "evaluation['fid']>quality_best_fid+float(guard['maximum_fid_increase_vs_matched_baseline'])")
    s=s.replace("quality_bad_evaluations >= 2:", "quality_bad_evaluations >= 1:")
    s=s.replace("reason='Two consecutive full50k evaluations exceed the matched-baseline FID tolerance or fail the Inception Score floor'", "reason='First full50k evaluation exceeding the best-FID tolerance or failing the matched IS floor'")
    needle="        if run:run.summary['training/status']='early_stopped_quality_regression' if quality_stopped else 'completed';run.finish()"
    assert s.count(needle)==1
    s=s.replace(needle,"        if quality_stopped:\n            from runtime.rewind_gate import record_rewind\n            record_rewind(OUT,'Full FID50k quality regression; restored best committed checkpoint as active')\n        if run:run.summary['training/status']='rewound_to_best_after_quality_regression' if quality_stopped else 'completed';run.finish()")
    path.write_text(s)
    support=BASE/'source/src/training/joint_runtime_support.py'
    support.write_text(support.read_text().replace("title='ImageNet Joint16384 best14574 continue100 1B300 20261104'",
        "title='ImageNet rewind best epoch73 gated 4H100 20261105'"))
    patch=''.join(difflib.unified_diff(before.splitlines(True),s.splitlines(True),fromfile='original/rq_imagenet_shared.py',tofile='rewind/rq_imagenet_shared.py'))
    (BASE/'rewind-runtime.patch').write_text(patch)
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'runtime.patch').write_text(patch)
    verification=dict(passed=True,restored_step=45698,epoch=73,next_microbatch=0,optimizer_states=780,
        all_adam_steps=45698,scheduler_step=45698,scheduler_horizon=62600,source_lr=p['optimizer']['param_groups'][0]['lr'],
        restored_lr=p['optimizer']['param_groups'][0]['lr'],amplitude_factor_unchanged=.15,
        all_four_rng_streams_available=True,source_layout_unchanged=True,global_batch=2048,
        stage1_and_codebooks_and_reference_sha256_verified=True,source_artifact=ARTIFACT,
        fresh_baseline_required=True,pilot_updates=8,pilot_acceptance=old['pilot_acceptance'],
        full_epoch_quality_stop_enabled=True,failed_schedule_removed=True)
    write(OUT/'source-verification.json',verification);write(BASE/'rewind-prepared.json',verification)
    for name in ['execution.json','recipe.yaml','hardware.json','best-four-gpu-reference.json']:
        shutil.copyfile(BASE/'inputs'/name,OUT/name)
    for name in ['prepare.py','rewind_gate.py']:
        shutil.copyfile(HERE/name,BASE/('rewind-'+name))
    print(json.dumps(verification,indent=2),flush=True)


if __name__=='__main__':main()
