#!/usr/bin/env python3
"""Replay the archived q995 compound recipe through two epochs with sign probes."""
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import time

ROOT = Path('/workspace/Projects/laser')
BASE = Path('/tmp/laser-compound-early-replay-20261016')
RUN = 'imagenet-compound-q995-early-sign-replay2-4h100-20261016'
OUT = ROOT/'outputs'/RUN
HISTORICAL = ROOT/'outputs/imagenet-rfid421-ffhq-q995-clipped-sigma42p65-scratch-4h200-20261009'
DATA = Path('/tmp/laser-rq-fine-joint512-imagenet-20261015')


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str)+'\n')
    temporary.replace(path)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare():
    import numpy as np
    import yaml
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'prepared.json').exists():
        raise RuntimeError('Already prepared')
    frozen = BASE/'archived/frozen'
    if not (BASE/'source').exists():
        (BASE/'source').symlink_to(frozen/'source', target_is_directory=True)
    # Preserve the model, objective, optimizer, target, and schedule code. The
    # short replay changes only the stopping epoch in this archived assertion.
    entry = (frozen/'entry-original.py').read_text()
    original = 'assert ARGS.epochs == ARGS.lr_schedule_epochs == 100 and ARGS.min_lr == 0'
    assert entry.count(original) == 1
    entry = entry.replace(original, 'assert ARGS.epochs == 2 and ARGS.lr_schedule_epochs == 100 and ARGS.min_lr == 0')
    (BASE/'historical-entry.py').write_text(entry)
    (BASE/'imagenet').mkdir(exist_ok=True)
    write(BASE/'imagenet/training-ready.json', dict(passed=True, training_images=1281167))
    (BASE/'inputs').mkdir(exist_ok=True)
    for old, new in [('stage1-model.pt','stage1-tokenizer.pt'),('imagenet_256_train.npz','imagenet_256_train.npz')]:
        if not (BASE/'inputs'/new).exists():
            os.link(DATA/'inputs'/old, BASE/'inputs'/new)
    if not (BASE/'torch-cache').exists():
        (BASE/'torch-cache').symlink_to(DATA/'torch-cache', target_is_directory=True)
    stage1_sha = sha(BASE/'inputs/stage1-tokenizer.pt')
    reference_sha = sha(BASE/'inputs/imagenet_256_train.npz')
    assert stage1_sha == 'dd28db9306d526bdc9fbf8016403e106c4f317fd8f5c04792f0953e53310fdab'
    assert reference_sha == '3f9c92d15755e76ec312964a819e3e19b9cf3aadc618a7ae99f5e8aa96501260'
    # Check every image and class against the original training sample order.
    index = np.load(DATA/'data/train-index.npz')
    labels, classes, filenames = (index[key] for key in ['labels','classes','filenames'])
    count = 0
    with gzip.open(BASE/'recovery/training-ordered-paths.tsv.gz','rt') as stream:
        for count, line in enumerate(stream, 1):
            name, label = line.rstrip('\n').split('\t')
            i = count-1
            actual_label = int(labels[i])
            actual_name = str(classes[actual_label])+'/'+str(filenames[i])
            assert name == actual_name and int(label) == actual_label, i
    assert count == 1281167
    repro = OUT/'reproducibility'
    repro.mkdir(exist_ok=True)
    for name in ['recovery-code.tar.gz','recovery-code-manifest.json','recovery-manifest.json','training-ordered-paths.tsv.gz']:
        shutil.copyfile(BASE/'recovery'/name, repro/name)
    shutil.copyfile(BASE/'historical-entry.py', repro/'historical-entry.py')
    shutil.copyfile(Path(__file__), repro/Path(__file__).name)
    shutil.copyfile(ROOT/'src/training/rq_imagenet_runtime.py', BASE/'jpeg_reader.py')
    late = HISTORICAL/'resume-best-fid-refine-4h200-20261012/chunked_checkpoint_io.py'
    shutil.copyfile(late, BASE/'chunked_checkpoint_io.py')
    config = yaml.safe_load((frozen/'production.yaml').read_text())
    options = config['options']
    options.update(checkpoint=str(BASE/'inputs/stage1-tokenizer.pt'), data=str(BASE/'imagenet'),
        output=str(OUT/'train'), checkpoint_dir=str(OUT/'train/checkpoints'),
        resume=False, resume_checkpoint=None, init_stage2_checkpoint=None,
        epochs=2, lr_schedule_epochs=100, batch_size=64, max_optimizer_steps=0,
        fid_reference_stats=str(BASE/'inputs/imagenet_256_train.npz'), fid_batch_size=512,
        fid_every=1, fid_early_epochs=2, save_ckpt_freq=1, save_step_freq=0,
        keep_best_checkpoints=2, sample_grid_every=0, upload_checkpoints=False,
        wandb_id=RUN, wandb_name='ImageNet | archived q995 early sign replay | scratch2 | 4H100')
    for folder in [OUT, BASE/'private-smoke']:
        for name in ['coefficient-support-policy.json','support-validation.json','original-fid-only-policy.json']:
            folder.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(HISTORICAL/name, folder/name)
    (BASE/'production.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    shutil.copyfile(BASE/'production.yaml', OUT/'train.yaml')
    smoke = dict(config, options=dict(options, output=str(BASE/'private-smoke/train'),
        checkpoint_dir=str(BASE/'private-smoke/train/checkpoints'), max_optimizer_steps=2,
        fid_every=0, fid_early_epochs=0, wandb_id=RUN+'-smoke', wandb_mode='offline'))
    (BASE/'smoke.yaml').write_text(yaml.safe_dump(smoke, sort_keys=False))
    plan = dict(prepared=True, run_id=RUN, source_run=HISTORICAL.name,
        source_artifact='helloimlixin-rutgers/laser/imagenet-sigma42p65-machine-migration-20261010:v0',
        archived_early_entry_and_source=True, fresh_model=True, fresh_optimizer=True,
        stage2_checkpoint_consumed=False, epochs=2, updates_per_epoch=626, stop_updates=1252,
        global_batch=2048, world_size=4, microbatch=64, accumulation=8,
        initial_lr=.0005, cosine_updates=62600, warmup=0,
        coefficient_bins=2048, normalized_coefficient_sigma=.125, atom_ce_weight=1.5,
        geometry_start_epoch=2, geometry_weight_during_first_two_epochs=0,
        held_out_sign_updates=[100,626,1252], held_out_images=128,
        generated_images_per_fid=50000, reference_images=1281167,
        stage1_sha256=stage1_sha, reference_sha256=reference_sha,
        every_training_path_and_label_matches_original=True, training_images=count,
        original_epoch98_weights_not_used=True,
        differences_from_historical_execution=['H100 instead of H200',
            'Microbatch64/accumulation8 instead of128/4; global batch retained',
            'JPEGs read from indexed original tar instead of extracted ImageFolder',
            'Stop after two epochs; retain original100-epoch cosine schedule',
            'Additional read-only sign diagnostics and epoch1 FID',
            'Generation batch512 with bounded16-image decoding'],
        bitwise_reproduction_claimed=False,
        replay_script_sha256=sha(Path(__file__)), archived_code_sha256=sha(repro/'recovery-code.tar.gz'))
    write(OUT/'prepared.json', plan)
    print(json.dumps(plan), flush=True)


def worker(args):
    os.environ['LASER_PHASE'] = 'preflight'  # Both launches intentionally start from scratch.
    os.environ['LASER_RUN_BASE'] = str(BASE)
    os.environ['LASER_PERSISTENT_BASE'] = str(args.output)
    # Import the archived entry without executing its __main__ training block.
    values = runpy.run_path(str(BASE/'historical-entry.py'), run_name='archived_early_entry')
    state = values['step'].__globals__
    training = values['training']
    torch, dist = values['torch'], values['dist']
    if args.output == BASE/'private-smoke':
        # These two disposable updates are verified and then discarded. Keep
        # the original persistent-path requirement for the production replay.
        training.persistent_checkpoint_dir = lambda output, requested: Path(requested)
    sys.path.insert(0, str(BASE))
    import chunked_checkpoint_io
    chunked_checkpoint_io.LOCAL = BASE/'chunk-cache'
    chunked_checkpoint_io.install(values['checkpoint_io'])
    spec = importlib.util.spec_from_file_location('early_replay_jpeg_reader', BASE/'jpeg_reader.py')
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    def image_folder(root, transform, augmentation_seed):
        return reader.TarImageNet(DATA/'data', 'train', transform, seed=augmentation_seed)
    state['EpochImageFolder'] = image_folder
    # Bound decoder activations without changing the tokens or pixels.
    decode = training.LaserAux.decode_compound
    def bounded_decode(aux, atoms, ids):
        return torch.cat([decode(aux,a,c) for a,c in zip(atoms.split(16),ids.split(16))])
    training.LaserAux.decode_compound = bounded_decode
    references, sign_sums = {}, None
    native_wrap = training.wrap_distributed_model
    def wrap(model, *a, **kw):
        assert sum(p.numel() for p in model.parameters()) == 1457980928
        references['model'] = model
        return native_wrap(model, *a, **kw)
    training.wrap_distributed_model = wrap
    native_objective = training.compound_objective
    def objective(atom_logits, coeff_logits, *a, **kw):
        nonlocal sign_sums
        assert kw['geometry_weight'] == 0, 'Early replay must stop before geometry activates'
        result = native_objective(atom_logits, coeff_logits, *a, **kw)
        if (state['CURRENT_STEP']+1) % 10 == 0:
            aux = state['COEFFICIENT_AUX']
            coefficients = state['CLEAN_COEFFICIENTS']
            with torch.no_grad():
                p = coeff_logits.detach().float().softmax(-1)
                correct = (p*((aux.coeff_bins>0)==(coefficients[...,None]>0))).sum(-1)
                flat = correct.reshape(-1,4)
                sums = torch.cat([flat.sum(0), (flat>.5).float().sum(0),
                                  flat.new_tensor([flat.shape[0]])]).double()
                sign_sums = sums if sign_sums is None else sign_sums+sums
        return result
    training.compound_objective = objective

    @torch.inference_mode()
    def held_out(step):
        model, aux = references['model'], state['COEFFICIENT_AUX']
        device = aux.dictionary.device
        probe = torch.load(ROOT/'outputs/imagenet-compound1622-sign-diagnosis-20261015/held-out-probe.pt',
                           map_location='cpu', weights_only=False)
        modes = {}
        was_training = model.training
        model.eval()
        try:
            # Diagnostics do not advance the training CPU/CUDA random streams.
            with torch.random.fork_rng(devices=[device.index]):
                for mode in ['clean_nearest_prefix','training_noise_prefix_sigma0125']:
                    sums = torch.zeros(4,dtype=torch.float64)
                    accuracy = torch.zeros_like(sums)
                    generator = torch.Generator(device=device).manual_seed(261015)
                    for start in range(0,128,4):
                        atoms = probe['atoms'][start:start+4].to(device)
                        coefficients = probe['coefficients'][start:start+4].to(device)
                        ids = probe['hard_ids'][start:start+4].to(device)
                        if mode != 'clean_nearest_prefix':
                            target = (-(coefficients[...,None]-aux.coeff_bins).square()/(2*.125**2)).softmax(-1)
                            ids = torch.multinomial(target.reshape(-1,2048),1,generator=generator).reshape_as(ids)
                        output = model(atoms*2048+ids,aux,probe['labels'][start:start+4].to(device),amp=True)
                        logits = output['coeff_logits'].float()
                        assert torch.isfinite(logits).all()
                        p = logits.softmax(-1)
                        correct = (p*((aux.coeff_bins>0)==(coefficients[...,None]>0))).sum(-1).reshape(-1,4)
                        sums += correct.double().sum(0).cpu()
                        accuracy += (correct>.5).double().sum(0).cpu()
                    modes[mode] = dict(correct_sign_probability=(sums/(128*64)).tolist(),
                        binary_sign_accuracy=(accuracy/(128*64)).tolist())
        finally:
            model.train(was_training)
        result = dict(global_step=step, images=128, predictions=32768,
            same_validation_indices_as_failed_head=True, original_tokenizer=True,
            prefix_modes=modes, diagnostic_only=True, training_rng_preserved=True)
        write(args.output/'sign'/f'held-out-step{step:07d}.json', result)
        if state['REAL_RUN'] is not None:
            payload = {'train/global_step':step}
            for mode, item in modes.items():
                for d,p in enumerate(item['correct_sign_probability'],1):
                    payload[f'diagnostics/{mode}_correct_sign_depth{d}'] = p
            state['REAL_RUN'].log(payload)
            state['REAL_RUN'].save(str(args.output/'sign'/f'held-out-step{step:07d}.json'),
                                   base_path=str(args.output),policy='now')

    native_step = torch.optim.AdamW.step
    def step(optimizer, *a, **kw):
        nonlocal sign_sums
        result = native_step(optimizer, *a, **kw)
        age = state['CURRENT_STEP']
        if sign_sums is not None:
            dist.all_reduce(sign_sums)
            mean = (sign_sums[:4]/sign_sums[-1]).tolist()
            for d,p in enumerate(mean,1):
                state['DIAGNOSTIC_METRICS'][f'train/correct_sign_probability_depth{d}'] = p
            if dist.get_rank()==0:
                write(args.output/'sign'/f'training-step{age:07d}.json',dict(global_step=age,
                    correct_sign_probability=mean, binary_sign_accuracy=(sign_sums[4:8]/sign_sums[-1]).tolist(),
                    globally_reduced=True, predictions=int(sign_sums[-1])*4))
            sign_sums = None
        if age in [100,626,1252]:
            dist.barrier()
            if dist.get_rank()==0:
                held_out(age)
            dist.barrier()
        return result
    torch.optim.AdamW.step = step
    # Give the replay its own run, with no writes to the historical run.
    native_init = state['native_wandb_init']
    def init(*a, **kw):
        run = native_init(*a, **kw)
        run.config.update(dict(early_sign_replay=True, historical_run=HISTORICAL.name,
            fresh_model=True, fresh_optimizer=True, historical_stage2_weights_consumed=False,
            training_order_verified=True, bitwise_historical_reproduction=False),allow_val_change=True)
        run.define_metric('train/correct_sign_probability_depth*',step_metric='train/global_step')
        run.define_metric('diagnostics/*',step_metric='train/global_step')
        return run
    state['native_wandb_init'] = init
    sys.argv = [str(BASE/'source/train.py'),'--config',str(args.config)]
    try:
        try:
            runpy.run_path(str(BASE/'source/train.py'),run_name='__main__')
        except SystemExit as ended:
            if ended.code not in (None,0):
                raise
    finally:
        values['WRITER'].close()
    if args.output == OUT and int(os.environ['RANK']) == 0:
        curves = {}
        for epoch in [1,2]:
            step = epoch*626
            evaluation = args.output/'train/evaluations'/f'original_generation_step_{step:07d}.json'
            sign = args.output/'sign'/f'held-out-step{step:07d}.json'
            if evaluation.exists() and sign.exists():
                curves[str(epoch)] = dict(evaluation=json.loads(evaluation.read_text()),
                    sign=json.loads(sign.read_text()),
                    historical_compound_fid={1:112.25180962767655,2:76.30317487032005}[epoch],
                    signed_atom_fid={1:127.17980566618633,2:77.62389932548473}[epoch])
        assert len(curves)==2, 'Missing requested early evaluation or sign diagnostic'
        write(args.output/'comparison.json',dict(completed_epochs=2,updates=1252,
            curves=curves,early_recipe_replayed=True,bitwise_historical_reproduction=False))


def run():
    import fcntl
    lock = (BASE/'controller.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env = dict(os.environ, LASER_ACCUMULATION='8', LASER_COMPILE_BLOCKS='1',
        LASER_FFHQ_DECODER_OBJECTIVE='1', LASER_COEFFICIENT_SIGMA_BINS=str(.125/(6/2047)),
        LASER_CHECKPOINT_STAGING_DIR=str(BASE/'checkpoint-staging'),
        LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(BASE/'checkpoint-upload-cache'),
        TORCH_HOME=str(BASE/'torch-cache'), OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
        PYTHONUNBUFFERED='1', TORCHINDUCTOR_CACHE_DIR=str(BASE/'inductor-cache'),
        WANDB_API_KEY=Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip(),
        WANDB_DIR=str(BASE/'wandb'), NCCL_NVLS_ENABLE='0')
    for name in ['checkpoint-staging','checkpoint-upload-cache','wandb']:
        (BASE/name).mkdir(exist_ok=True)
    for phase,output in [('smoke',BASE/'private-smoke'),('production',OUT)]:
        command = [sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=4',
                   str(Path(__file__).resolve()),'worker','--config',str(BASE/(phase+'.yaml')),
                   '--output',str(output)]
        with (OUT/(phase+'.log')).open('w') as stream:
            child = subprocess.Popen(command,cwd=BASE,env=env,stdout=stream,stderr=subprocess.STDOUT)
            write(OUT/'launch-status.json',dict(phase=phase,controller_pid=os.getpid(),
                child_pid=child.pid,run_id=RUN,started_at=time.time()))
            code = child.wait()
        if code:
            write(OUT/'launch-status.json',dict(phase=phase+'_failed',returncode=code,time=time.time()))
            raise RuntimeError('Replay failed; inspect '+phase+'.log')
        if phase=='smoke':
            import torch
            sys.path.insert(0,str(BASE))
            import chunked_checkpoint_io
            chunked_checkpoint_io.LOCAL=BASE/'chunk-cache'
            source=output/'train/checkpoints/last.pt'
            checkpoint=chunked_checkpoint_io.recover(source,lambda p:p)
            p=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
            assert p['global_step']==2
            assert len(p['state_dict'])==837
            assert {int(v['step']) for v in p['optimizer']['state'].values()}=={2}
            assert p['scheduler']['T_max']==62600
            write(OUT/'smoke-verification.json',dict(passed=True,global_step=2,
                model_tensors=837,optimizer_states=len(p['optimizer']['state']),
                adam_age=2,cosine_updates=62600,production_starts_from_scratch=True,
                smoke_weights_and_optimizer_not_consumed=True))
            del p
    write(OUT/'launch-status.json',dict(phase='completed',returncode=0,epochs=2,updates=1252,time=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','worker'])
    parser.add_argument('--config',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.action=='prepare': prepare()
    elif args.action=='worker': worker(args)
    else: run()
