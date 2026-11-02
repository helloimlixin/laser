#!/usr/bin/env python3
"""Resume a measured improvement; otherwise preserve the paused parent."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path('/workspace/Projects/laser')
TRIALS = ROOT / 'outputs/imagenet-laser-pair-ablation-20261015'
PARENT_BASE = Path('/tmp/laser-rq-causal-ar-imagenet-20261015')
PARENT_RUN = 'imagenet-laser-rq-causal-ar-rfid421-continue20-20261015'
NEW_RUN = 'imagenet-laser-rq-validated-ar-rfid421-continue20-20261015'
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts/tools'), str(PARENT_BASE / 'upstream')]
os.environ['RQ_UPSTREAM_ROOT'] = str(PARENT_BASE / 'upstream')

import torch
from omegaconf import OmegaConf
import launch_laser_hard_ar_imagenet as launcher
from src.training.rq_imagenet_recipes import LaserRecipe
from src.training.rq_imagenet_runtime import write_json
from rqvae.utils.config import load_config, augment_arch_defaults


def main():
    comparison = json.loads((TRIALS / 'comparison.json').read_text())
    assert comparison['passed'] and comparison['tested_step'] == 3884
    parent_record=json.loads((ROOT/'outputs'/PARENT_RUN/'checkpoint.json').read_text())
    source_best_fid=parent_record['best_fid']
    matched_gate=bool(comparison['qualifies_for_adoption'])
    adopt = matched_gate and comparison['fid'][comparison['best_branch']] <= source_best_fid
    comparison.update(matched_gain_gate_passed=matched_gate,source_best_fid=source_best_fid,
                      qualifies_for_adoption=adopt)
    write_json(TRIALS/'comparison.json',comparison)
    if not adopt:
        decision = dict(passed=True, improved=False, production_replaced=False,
            production_training_resumed=False, status='no_improvement_parent_preserved',
            source_best_fid=source_best_fid, matched_control_fid=comparison['fid']['scalar-hard'],
            best_trial=comparison['best_branch'], best_trial_fid=comparison['fid'][comparison['best_branch']],
            reason='No candidate both improves matched control by at least 1 FID and beats the preserved parent best.',
            preserved_checkpoint=parent_record['path'], preserved_global_step=parent_record['global_step'],
            timestamp=time.time())
        write_json(TRIALS/'adoption.json',decision)
        print(json.dumps(decision),flush=True)
        return
    branch = comparison['best_branch'] if adopt else 'scalar-hard'
    result = json.loads((TRIALS / branch / 'result.json').read_text())
    fid = json.loads((TRIALS / branch / 'fid50k.json').read_text())
    sampling = json.loads((TRIALS / branch / 'sampling-verification.json').read_text())
    assert sampling['passed'] and fid['generated_images'] == 50000
    assert result['global_step'] == fid['global_step'] == 3884
    assert not result['auxiliary_loss_terms'] and not result['sampled_omp_trajectories']
    recipe = OmegaConf.load(PARENT_BASE / 'inputs/recipe.yaml')
    if adopt:
        assert comparison['improvement_over_matched_control'] >= 1.
        recipe.run_id = NEW_RUN
        recipe.name = f'LASER validated {branch} | upstream RQ | continue20'
        recipe.microbatch = 32
        recipe.decoder = 'compound_pair' if branch.startswith('pair-') else 'scalar'
        recipe.coefficient_policy = 'soft' if branch.endswith('-soft') else 'hard'
        recipe.objective = ('fixed_support_soft_coefficient_cross_entropy'
                            if branch.endswith('-soft') else 'hard_next_token_cross_entropy')
        config_path = ROOT / 'configs/stage2/imagenet-laser-rq-validated-ar-20ep-20261015.yaml'
        OmegaConf.save(recipe, config_path)
        launcher.BASE = Path('/tmp/laser-rq-validated-ar-imagenet-20261015')
        launcher.SOURCE_BASE = PARENT_BASE
        launcher.RUN = NEW_RUN
        launcher.OUT = ROOT / 'outputs' / NEW_RUN
        launcher.RECIPE_PATH = config_path
        launcher.NEW_WANDB_RUN = True
        launcher.prepare()
        prepared = json.loads((launcher.OUT / 'prepared.json').read_text())
        prepared.update(fresh_stage2=False, fresh_optimizer=False,
                        initialization='validated matched decoder/coefficient continuation',
                        parent_global_step=3756, continued_at_step=3884)
        write_json(launcher.OUT / 'prepared.json', prepared)
    else:
        launcher.BASE, launcher.OUT, launcher.RUN = PARENT_BASE, ROOT/'outputs'/PARENT_RUN, PARENT_RUN
        launcher.NEW_WANDB_RUN = False
    base, out = launcher.BASE, launcher.OUT
    os.environ.update(json.loads((base / 'environment.json').read_text()))
    from src.training.k4_checkpoint_io import atomic_torch_save, _checkpoint_upload_source
    payload = torch.load(result['checkpoint'], map_location='cpu', mmap=True, weights_only=False)
    torch.set_num_threads(4)
    profile = LaserRecipe(base)
    config = load_config(base / 'inputs/stage2-training-config.yaml')
    arch = augment_arch_defaults(config.arch)
    arch.vocab_size = config.dataset.vocab_size
    with torch.device('meta'):
        model = profile.model(arch)
    model.load_state_dict(payload['state_dict'], strict=True, assign=True)
    names = list(dict(model.named_parameters()))
    states = payload['optimizer']['state']
    indices = [i for group in payload['optimizer']['param_groups'] for i in group['params']]
    assert len(names) == len(states) == len(indices)
    model_hash, optimizer_hash = hashlib.sha256(), hashlib.sha256()
    for name, value in payload['state_dict'].items():
        assert torch.isfinite(value).all(), name
        model_hash.update(name.encode())
        model_hash.update(value.contiguous().numpy().tobytes())
    age_groups = {}
    for name, index in zip(names, indices):
        state = states[index]
        expected_age = 128 if result['parameter_mapping'].get(name) is None else 3884
        assert int(state['step']) == expected_age, name
        age_groups[expected_age] = age_groups.get(expected_age, 0) + 1
        for field in ('exp_avg', 'exp_avg_sq'):
            value = state[field]
            assert torch.isfinite(value).all() and value.shape == payload['state_dict'][name].shape
            optimizer_hash.update(str(index).encode() + field.encode())
            optimizer_hash.update(value.contiguous().numpy().tobytes())
    if adopt:
        source_root=Path(result.get('frozen_source_base','/tmp/laser-pair-ablation-20261015/source'))
        adapter=('src/models/laser_compound_rq_adapter.py' if branch.startswith('pair-')
                 else 'src/models/laser_rq_adapter.py')
        for name in [adapter, 'src/training/laser_pair_ablation.py',
                     'src/training/laser_ar_factors.py', 'src/orthogonal_sparse_codec.py']:
            assert (base/'source'/name).read_bytes() == (source_root/name).read_bytes()
    retained_best_fid=fid['fid'] if adopt else min(source_best_fid,fid['fid'])
    payload.update(run_id=profile.run_id, recipe=OmegaConf.to_container(profile.recipe, resolve=True),
        best_fid=retained_best_fid, evaluation=fid,
        next_microbatch=128*16 if adopt else 128*8,
        config=OmegaConf.to_container(config, resolve=True),
        runtime=dict(fresh_stage2=False, fresh_optimizer=False, global_step=3884,
            parameter_count=sum(p.numel() for p in model.parameters()), parameter_tensors=len(names),
            world_size=4, microbatch=profile.microbatch, global_batch=2048,
            accumulation_steps=16 if adopt else 8, optimizer_updates_per_epoch=626,
            **profile.metadata()),
        continuation_origin=dict(experiment=str(TRIALS), branch=branch,
            model_state_sha256=model_hash.hexdigest(), adam_moments_sha256=optimizer_hash.hexdigest(),
            architecture_and_coefficient_policy_improved=adopt))
    if branch.endswith('-soft'):
        for rank, rng in enumerate(payload['rng_state_by_rank']):
            rng['coefficient'] = payload['coefficient_rng_by_rank'][rank]
    profile.verify_resume(payload)
    folder = out / 'persistent-train/checkpoints'
    folder.mkdir(parents=True, exist_ok=True)
    if not (out/'checkpoints').exists():
        (out/'checkpoints').symlink_to('persistent-train/checkpoints', target_is_directory=True)
    print(json.dumps(dict(phase='copying_verified_checkpoint', branch=branch,
                         improved=adopt, output=str(out))), flush=True)
    atomic_torch_save(payload, out/'checkpoints/last.pt')
    best = out/'checkpoints/best.pt'
    if adopt or fid['fid'] < source_best_fid:
        temporary = best.with_suffix('.linktmp')
        temporary.unlink(missing_ok=True)
        temporary.symlink_to(os.readlink(out/'checkpoints/last.pt'))
        temporary.replace(best)
    assert best.exists()
    persistent = (out/'checkpoints/last.pt').resolve()
    local = _checkpoint_upload_source(persistent)
    assert local.stat().st_size == persistent.stat().st_size > 10_000_000_000
    proof = dict(passed=True, branch=branch, continued_at_step=3884,
        matched_control_fid=comparison['fid']['scalar-hard'], continued_fid=fid['fid'],
        fid_improvement=comparison['improvement_over_matched_control'] if adopt else 0.,
        best_fid=retained_best_fid,protected_best_checkpoint=str(best.resolve()),
        fid_generated_images=50000, fid_reference_images=1281167,
        matched_performance_improvement_verified=adopt,
        architecture_changed=branch.startswith('pair-'),
        architecture_improvement_verified=adopt and branch.startswith('pair-'),
        coefficient_policy_improvement_verified=adopt and branch.endswith('-soft'),
        scalar_control_retained=not adopt,
        model_and_adam_unmodified=True, model_state_sha256=model_hash.hexdigest(),
        adam_moments_sha256=optimizer_hash.hexdigest(), optimizer_age_groups=age_groups,
        rng_streams_preserved=4, scheduler_unchanged=True, strict_model_reload=True,
        fp16_cache_verification=sampling, checkpoint=str(persistent), local_checkpoint=str(local),
        auxiliary_loss_terms=0, omp_trajectory_banks=0, timestamp=time.time())
    write_json(out/'pair-ablation-continuation.json', proof)
    write_json(out/'continuation-verification.json', proof)
    write_json(out/'checkpoint.json', dict(epoch=6,global_step=3884,path=str(out/'checkpoints/last.pt'),
                                          bytes=persistent.stat().st_size,best_fid=retained_best_fid))
    if adopt:
        for source in [TRIALS/'comparison.json', TRIALS/'prepared.json',
                       TRIALS/'full-model-transfer-preflight.json', TRIALS/'production-cpu-tests.xml',
                       TRIALS/'continuation-cpu-tests.xml',
                       TRIALS/branch/'fid50k.json', TRIALS/branch/'sampling-verification.json']:
            shutil.copyfile(source, out/'reproducibility'/source.name)
        if branch.startswith('pair-raw-'):
            shutil.copyfile(TRIALS/'prepared-raw-identity.json',out/'reproducibility/selected-trial-prepared.json')
            shutil.copyfile(TRIALS/'raw-identity-cpu-tests.xml',out/'reproducibility/selected-trial-cpu-tests.xml')
            shutil.copyfile(TRIALS/'source-raw-identity.zip',out/'reproducibility/selected-trial-source.zip')
        suite = ET.parse(TRIALS/'continuation-cpu-tests.xml').getroot().find('testsuite')
        assert int(suite.attrib['tests']) == 42 and int(suite.attrib['failures']) == int(suite.attrib['errors']) == 0
        write_json(out/'cpu-verification.json',dict(passed=True,tests=42,
            test_receipt=str(out/'reproducibility/continuation-cpu-tests.xml'),
            strict_model_reload=True,fp16_cache_verification=sampling))
    else:
        # Append the tested continuation point without changing the old curve.
        import wandb
        os.environ['WANDB_API_KEY']=Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip()
        run=wandb.init(entity='helloimlixin-rutgers',project='laser',id=PARENT_RUN,
                       resume='must',dir=str(PARENT_BASE/'wandb'))
        run.log({'train/global_step':3884,'train/epoch':3884/626,'eval/fid':fid['fid'],
                 'eval/inception_score':fid['inception_score']})
        run.summary['analysis/pair_ablation_improved']=False
        run.finish()
    launch_script = base/'resume_validated.py'
    recovery_status = ROOT/'outputs/imagenet-laser-historical-recovery-20261015/controller.json'
    launch_script.write_text('import sys,json,time\nfrom pathlib import Path\n'
        f'sys.path.insert(0,{str(ROOT/"scripts/tools")!r})\n'
        'import launch_laser_hard_ar_imagenet as launcher\n'
        f'launcher.BASE=Path({str(base)!r})\nlauncher.OUT=Path({str(out)!r})\n'
        f'launcher.RUN={profile.run_id!r}\nlauncher.NEW_WANDB_RUN={adopt!r}\n'
        f'recovery=Path({str(recovery_status)!r})\n'
        'while recovery.exists() and json.loads(recovery.read_text()).get("phase")!="complete":\n'
        ' time.sleep(4)\n'
        'sys.exit(launcher.run(resume=True))\n')
    with (out/'continuation-controller.log').open('a') as log:
        process = subprocess.Popen([sys.executable,str(launch_script)],cwd=ROOT,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(TRIALS/'adoption.json', dict(proof, run_id=profile.run_id,controller_pid=process.pid,
                                           run_url=f'https://wandb.ai/helloimlixin-rutgers/laser/runs/{profile.run_id}'))
    print(json.dumps(dict(run_id=profile.run_id,controller_pid=process.pid,branch=branch,improved=adopt)),flush=True)


if __name__ == '__main__':
    main()
