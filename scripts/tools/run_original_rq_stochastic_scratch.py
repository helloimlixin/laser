"""Freeze and supervise the original RQ ImageNet recipe with LASER pair targets."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import zipfile

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.tools.run_compound_energy_trial import environment, patch_trial_runtime, record, command


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(args):
    import yaml
    from src.training.stochastic_image_pairs import SOFT_POLICY_VERSION
    if (args.output/'preparation-complete.json').exists():
        raise RuntimeError('Already prepared; use run or resume-production')
    args.base.mkdir(parents=True, exist_ok=True)
    parent = Path('/tmp/laser-compound-energy-ab-20261007/control')
    archive = args.output/'recipe-sources/stage2-config.yaml'
    archive_hash = digest(archive)
    assert archive_hash == '6065c30c936ad534b329f09a652570ea689045c4f8019edc50087f2d81d46d87'
    archived = yaml.safe_load(archive.read_text())
    assert archived['sampling'] == dict(top_k=[256],top_p=[.95],temp=1.)
    official = REPO/'third_party/rq-vae-transformer/configs/imagenet256/stage2/in256-rqtransformer-8x8x4-1400M.yaml'
    recipe = yaml.safe_load(official.read_text())
    assert recipe['optimizer']['init_lr'] == .0005 and recipe['experiment']['epochs'] == 100
    reference = args.output/'recipe-sources/imagenet_256_train.npz'
    assert digest(reference) == '3f9c92d15755e76ec312964a819e3e19b9cf3aadc618a7ae99f5e8aa96501260'
    shutil.copyfile(official,args.output/'recipe-sources/repository-stage2-config.yaml')
    calibration_path = REPO/'outputs/stochastic-image-pairs-20261007/calibration.json'
    calibration = json.loads(calibration_path.read_text())
    assert calibration['passed'] and calibration['selected_temperature'] == .05
    shutil.copyfile(calibration_path,args.output/'stochastic-teacher-calibration.json')
    plan = dict(version='imagenet-original-rq-stochastic-scratch-v1',
        wandb_id='imagenet-rfid421-stochastic-soft-compound-rqpaper-scratch-8h100-20261007',
        source_stage2_checkpoint=None, fresh_model=True,fresh_optimizer=True,
        frozen_tokenizer_rfid=4.21091413, bundle_config_sha256=archive_hash,
        bundle_url='https://twg.kakaocdn.net/brainrepo/models/RQVAE/f5cf4e5f3f0b5088d52cbb5e85c1077f/imagenet_1.4B.tar.gz',
        paper_url='https://arxiv.org/html/2203.01941',
        original_architecture=dict(spatial_layers=42,depth_layers=6,width=1536,heads=24,
            grid=[8,8],compound_depth=4,residual_dropout=.1,attention_dropout=0.,embedding_dropout=0.),
        optimizer=dict(type='AdamW',lr=.0005,betas=[.9,.95],weight_decay=.0001,max_grad_norm=1.,
            global_batch=2048,microbatch_per_gpu=64,accumulation_steps=4,gpus=8),
        schedule=dict(type='cosine',epochs=100,updates_per_epoch=626,total_updates=62600,
            minimum_lr=0.,warmup_epochs=0,resume_own_preflight_without_reset=True),
        target_policy=dict(version=SOFT_POLICY_VERSION,atom_temperature=.05,
            coefficient_temperature=.01125,variants=16,site_chunk_size=128,
            coefficient_space='normalized',unclipped=True,
            teacher='fresh full-vocabulary stochastic OMP, joint least-squares refit, matching noisy coefficients',
            labels='exact interleaved conditional soft targets within fresh online trajectory mixture'),
        objective='equal atom/coefficient soft cross entropy, no auxiliary objectives',
        adaptations=['frozen LASER dictionary tokenizer replaces RQ-VAE',
            'four interleaved atom/coefficient pairs replace four residual-quantizer codes',
            'both heads receive every previous complete compound event through gated full-history attention',
            'physical atom temperature and normalized coefficient temperature calibrated in dictionary units',
            '16 newly sampled trajectories per image visit give consistent soft labels for both heads',
            'BF16 compute with FP32 master parameters, Adam moments, and stochastic OMP; microbatch64 accumulation4'],
        sampler=dict(atom_top_k=256,atom_top_p=.95,atom_temperature=1.,
            coefficient_top_k=0,coefficient_top_p=.95,coefficient_temperature=1.,
            source='actual released stage2-config.yaml; all coefficient bins retained before nucleus filtering'),
        metrics=dict(backend='original_rqtransformer',generated_images=50000,classes=1000,
            images_per_class=50,real_split='train',real_images=1281167,inception_splits=10,
            seed=261001,every_epochs=2,additional_fid_backends=False,
            reference_sha256=digest(reference)),
        checkpoints=dict(full_optimizer_rng=True,best_fid=True,best_inception=True,last=True,
            upload_online=True,verify_remote_digest=True,save_updates=250,save_epochs=2),
        training_images='/tmp/laser-imagenet-stage2/imagenet',training_images_downloaded=False,
        protected_previous_best_unchanged=True,preflight_updates=20,prepared_unix=time.time())
    runtime = args.base/'source/runtime'
    shutil.copytree(parent/'source/runtime',runtime,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(REPO/'src',runtime/'src',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    # Preserve the exercised persistence implementation, including ordered aliases.
    for name in ('background_checkpoint.py','k4_checkpoint_io.py'):
        shutil.copyfile(parent/'source/runtime/src/training'/name,runtime/'src/training'/name)
    patch_trial_runtime(runtime)
    trainer = runtime/'src/training/rqtransformer.py'
    text = trainer.read_text()
    old = '            epoch == 0 or epoch + 1 <= args.fid_early_epochs\n'
    assert text.count(old) == 1
    text = text.replace(old,'            epoch + 1 <= args.fid_early_epochs\n')
    # Removing only a best-name symlink leaks its immutable payload/cache.
    assert text.count('                        stale_path.unlink()') == 2
    text = text.replace('                        stale_path.unlink()',
                        '                        remove_checkpoint(stale_path)')
    trainer.write_text(text)
    shutil.copytree(parent/'support',args.base/'support',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(REPO/'scripts/tools/train_original_rq_stochastic_image_pairs.py',args.base/'entry.py')
    (args.base/'inputs').mkdir()
    (args.base/'inputs/resume-stage1-tokenizer.pt').symlink_to(parent/'inputs/resume-stage1-tokenizer.pt')
    (args.base/'inputs/imagenet_256_train.npz').symlink_to(reference)
    for name in ('torch-cache','inductor-cache'):
        (args.base/name).symlink_to(parent/name)
    output = args.output/'train'
    (output/'checkpoints').mkdir(parents=True,exist_ok=True)
    options = dict(checkpoint=str(args.base/'inputs/resume-stage1-tokenizer.pt'),
        data=plan['training_images'],token_cache=None,output=str(output),dataset='imagenet',
        model_preset='imagenet-1400m',distributed_backend='ddp',num_atoms=16384,sparsity_level=4,
        coeff_vocab_size=2048,coeff_max=3.,coeff_scale=6.4,
        coeff_scales=[8.203365325927734,4.265638828277588,3.0662174224853516,1.8273425102233887],
        coeff_target_space='normalized',coeff_target_mode='soft',coeff_target_temperature=.01125,
        stochastic_atom_temperature=.05,stochastic_atom_site_chunk=128,stochastic_atom_soft_target_variants=16,
        compound_tokens=False,physical_pair_context=True,epochs=100,batch_size=64,total_batch_size=2048,
        lr=.0005,lr_schedule='cosine',lr_schedule_epochs=100,min_lr=0.,warmup_epochs=0.,seed=261001,
        atom_temperature=1.,atom_top_k=256,atom_top_p=.95,coeff_temperature=1.,coeff_top_k=0,coeff_top_p=.95,
        metric_backend='original-rqvae',fid_reference_stats=str(args.base/'inputs/imagenet_256_train.npz'),
        fid_real_split='train',fid_num_samples=50000,fid_batch_size=64,fid_every=2,fid_early_epochs=0,
        fid_seed=261001,save_ckpt_freq=2,save_step_freq=250,keep_best_checkpoints=1,
        model_only_best_checkpoints=False,sample_grid_every=0,sample_grid_on_start=False,
        atom_loss_weight=1.,coeff_crps_weight=0.,geometry_loss_weight=0.,
        upload_checkpoints=True,checkpoint_upload_mode='files',upload_token_cache=False,
        wandb_entity='helloimlixin-rutgers',wandb_project='laser',wandb_id=plan['wandb_id'],
        wandb_name='ImageNet K4 | original RQ recipe | stochastic soft pairs | scratch100',wandb_mode='online',
        checkpoint_dir=str(output/'checkpoints'),max_optimizer_steps=0)
    for phase in ('preflight','production'):
        chosen = dict(options,resume=phase=='production')
        if phase == 'preflight':
            chosen.update(max_optimizer_steps=20,fid_every=0,save_step_freq=0,upload_checkpoints=False)
        else:
            chosen['resume_checkpoint'] = str(output/'checkpoints/last.pt')
        (args.base/f'{phase}.yaml').write_text(yaml.safe_dump(dict(options=chosen),sort_keys=False))
        shutil.copyfile(args.base/f'{phase}.yaml',args.output/f'{phase}.yaml')
    record(args.base/'plan.json',plan)
    record(args.output/'plan.json',plan)
    files = list(runtime.rglob('*.py'))+list((args.base/'support').rglob('*.py'))+[args.base/'entry.py']
    manifest = {str(p.relative_to(args.base)):digest(p) for p in files}
    record(args.output/'code-manifest.json',manifest)
    with zipfile.ZipFile(args.output/'original-rq-scratch-code.zip','w',zipfile.ZIP_DEFLATED) as bundle:
        for path in files:
            bundle.write(path,path.relative_to(args.base))
        for path in (args.base/'plan.json',args.base/'preflight.yaml',args.base/'production.yaml'):
            bundle.write(path,path.name)
        for name in ('scripts/tools/run_original_rq_stochastic_scratch.py','tests/test_stochastic_image_pairs.py',
                     'tests/test_omp_joint_targets.py','tests/test_physical_compound_prior.py'):
            bundle.write(REPO/name,name)
        for path in (args.output/'recipe-sources').glob('*.yaml'):
            bundle.write(path,'recipe-sources/'+path.name)
    record(args.output/'recovery-instructions.json',dict(
        entrypoint='entry.py',launcher='scripts/tools/run_original_rq_stochastic_scratch.py',
        resume_action='resume-production',runtime_archive='original-rq-scratch-code.zip',
        dependencies=['frozen-stage1-tokenizer.pt','imagenet_256_train.npz','pt_inception-2015-12-05-6726825d.pth'],
        recovery='Restore full last.pt or best-FID/IS checkpoint, all Adam moments, scheduler and eight rank RNG streams; retain target policy and data order',
        stage2_initialization='random weights and empty optimizer at step0; preflight20 updates retained exactly'))
    record(args.output/'preparation-complete.json',dict(passed=True,manifest_files=len(manifest),time=time.time()))
    print(json.dumps(dict(prepared=True,base=str(args.base),output=str(args.output),run=plan['wandb_id'])),flush=True)


def verify_preflight(args):
    import torch
    sys.path[:0] = [str(args.base/'source/runtime'),str(args.base/'support')]
    from src.training.k4_checkpoint_io import _checkpoint_upload_source
    from src.training.full_resume_upload import recovery_metadata
    os.environ['LASER_CHECKPOINT_UPLOAD_CACHE_DIR'] = str(args.base/'checkpoint-upload-cache')
    os.environ['LASER_CHECKPOINT_IMMUTABLE_FILES'] = '1'
    saved = json.loads((args.output/'last-local-save.json').read_text())
    payload = torch.load(_checkpoint_upload_source(Path(saved['target'])),map_location='cpu',mmap=True,weights_only=False)
    plan = json.loads((args.base/'plan.json').read_text())
    assert payload['original_rq_scratch_policy'] == plan['target_policy']
    metadata = recovery_metadata(payload)
    assert metadata['global_step'] == metadata['adam_step'] == 20 and metadata['rng_ranks'] == 8
    assert metadata['adam_parameters'] == 801 and metadata['next_microbatch'] == 80
    scheduler = payload['scheduler']
    assert scheduler['last_epoch'] == 20 and scheduler['T_max'] == 62600 and scheduler['eta_min'] == 0
    expected_lr = .0005*(1+math.cos(math.pi*20/62600))/2
    assert math.isclose(metadata['saved_learning_rates'][0],expected_lr,rel_tol=1e-10)
    proofs = []
    for rank in range(8):
        root = args.output/'verification/preflight'
        startup = json.loads((root/f'startup-rank{rank}.json').read_text())
        target = json.loads((root/f'stochastic-targets-rank{rank}.json').read_text())
        update = json.loads((root/f'step20-rank{rank}.json').read_text())
        assert startup['fresh_optimizer'] and startup['adam_state_count'] == 0 and startup['global_step'] == 0
        assert startup['lr'] == .0005 and not startup['old_stage2_checkpoint_loaded']
        assert update['finite'] and update['adam_step'] == 20
        assert target['both_inputs_stochastic'] and target['both_labels_soft'] and target['rng_replay_exact']
        proofs.append(dict(rank=rank,startup=startup,targets=target,step20=update))
    record(args.output/'preflight-verification.json',dict(passed=True,recovery=metadata,rank_proofs=proofs,
        next_action='exactly resume own20 scratch updates without changing optimizer or schedule',time=time.time()))


def publish(args):
    sys.path.insert(0,str(args.base/'support'))
    os.environ['WANDB_API_KEY'] = args.key_file.read_text().strip()
    from verified_wandb_checkpoint_upload import VerifiedCloudUpload
    plan = json.loads((args.base/'plan.json').read_text())
    upload = VerifiedCloudUpload('helloimlixin-rutgers/laser/'+plan['wandb_id'],args.output/'cloud-dependencies-receipt.json')
    stage = args.base/'recovery-dependencies'
    stage.mkdir(exist_ok=True)
    source_weights = args.base/'torch-cache/hub/checkpoints/pt_inception-2015-12-05-6726825d.pth'
    extras = [args.output/name for name in ('plan.json','production.yaml','preflight.yaml','code-manifest.json',
        'original-rq-scratch-code.zip','preflight-verification.json','recovery-instructions.json',
        'stochastic-teacher-calibration.json','test-verification.json','code-transition.json',
        'code-manifest-preflight.json','original-rq-scratch-preflight-code.zip',
        'retention-verification.json') if (args.output/name).exists()]
    inputs = [('frozen-stage1-tokenizer.pt',args.base/'inputs/resume-stage1-tokenizer.pt'),
        ('imagenet_256_train.npz',args.base/'inputs/imagenet_256_train.npz'),(source_weights.name,source_weights)]
    for name,path in inputs:
        target = stage/name
        if not target.exists():
            shutil.copyfile(path.resolve(strict=True),target)
        extras.append(target)
    upload(extras,0)


def run(args):
    with (args.base/'supervisor.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest = json.loads((args.output/'code-manifest.json').read_text())
        assert all(digest(args.base/name) == expected for name,expected in manifest.items()),'Frozen source changed'
        stopped = False
        phase = None
        def stop(*_):
            nonlocal stopped
            stopped = True
            if phase is not None:
                for proof in (args.output/'verification'/phase).glob('process-rank*.json'):
                    pid = json.loads(proof.read_text())['pid']
                    if str(args.base/'entry.py') in command(pid):
                        try:
                            os.kill(pid,signal.SIGTERM)
                        except ProcessLookupError:
                            pass
        signal.signal(signal.SIGTERM,stop)
        signal.signal(signal.SIGINT,stop)
        phases = ['preflight','production'] if args.action == 'run' else ['production']
        try:
            if args.action == 'resume-production':
                assert json.loads((args.output/'preflight-verification.json').read_text())['passed']
            for phase in phases:
                if stopped:
                    break
                env = environment(args.base,args.output,args.key_file,phase)
                launch = [sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=8',
                    str(args.base/'entry.py'),'--config',str(args.base/f'{phase}.yaml')]
                with (args.output/'training.log').open('a') as log:
                    child = subprocess.Popen(launch,cwd=args.base,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    while child.poll() is None:
                        record(args.output/'status.json',dict(state='stopping' if stopped else 'running',phase=phase,
                            supervisor_pid=os.getpid(),torchrun_pid=child.pid,from_scratch=True,target_epochs=100,time=time.time()))
                        time.sleep(5)
                if child.returncode:
                    raise RuntimeError(f'{phase} exited with code{child.returncode}; inspect training.log')
                if phase == 'preflight' and not stopped:
                    verify_preflight(args)
            record(args.output/'status.json',dict(state='stopped_resumable' if stopped else 'completed',
                phase=phase,from_scratch=True,time=time.time()))
        except BaseException as error:
            stop()
            record(args.output/'status.json',dict(state='failed',phase=phase,error=str(error),time=time.time()))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','run','resume-production','publish','verify-preflight'])
    parser.add_argument('--base',type=Path,default=Path('/tmp/laser-imagenet-original-rq-stochastic-scratch-20261007'))
    parser.add_argument('--output',type=Path,default=REPO/'outputs/imagenet-original-rq-stochastic-scratch-20261007')
    parser.add_argument('--key-file',type=Path,default=Path('/tmp/laser-resume-wandb-key'))
    args = parser.parse_args()
    {'prepare':prepare,'run':run,'resume-production':run,'publish':publish,'verify-preflight':verify_preflight}[args.action](args)
