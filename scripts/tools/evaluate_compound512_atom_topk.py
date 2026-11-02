#!/usr/bin/env python3
"""Isolate atom top-k with fixed epoch-one weights and a full 50k FID."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path('/workspace/Projects/laser')
BASE = Path('/tmp/laser-compound512-native-imagenet-20261016')
SOURCE_OUT = ROOT/'outputs/imagenet-laser-compound512-native-scratch2-4h100-20261016'
OUT = ROOT/'outputs/imagenet-compound512-epoch1-atomtopk256-fid50k-20261016'
AUDIT = ROOT/'outputs/imagenet-compound512-gap-20261016'
os.environ['RQ_UPSTREAM_ROOT'] = str(BASE/'upstream')
os.environ['TORCH_HOME'] = str(BASE/'torch-cache')
sys.path[:0] = [str(BASE/'source'), str(BASE/'runtime'), str(BASE/'upstream'), str(ROOT)]


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2)+'\n'); temporary.replace(path)


def queue():
    import fcntl
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT/'controller.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    proof = json.loads((AUDIT/'compound512-epoch001-cache.json').read_text())
    assert proof['passed'] and proof['global_step'] == 626
    checkpoint = Path(proof['checkpoint'])
    with checkpoint.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    reference = json.loads((SOURCE_OUT/'evaluations/epoch001.json').read_text())
    assert reference['generated_images'] == 50000 and reference['real_images'] == 1281167
    plan = dict(checkpoint=str(checkpoint), checkpoint_sha256=digest, checkpoint_bytes=checkpoint.stat().st_size,
        source_run=SOURCE_OUT.name, source_epoch=1, source_step=626, reference_fid=reference['eval/fid'],
        changed_setting='atom_top_k', baseline_value=0, candidate_value=256,
        source_sampling=dict(atom_temperature=.9, atom_top_k=0, atom_top_p=.9,
            coefficient_temperature=1., coefficient_top_k=0, coefficient_top_p=.85),
        generation_seed=261001, generation_batch=128, generated_images=50000,
        real_images=1281167, reference_split='train', training_updates_added=0,
        automatic_training_extension=False, queued_after_current_two_epoch_trial=True,
        evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write(OUT/'plan.json', plan)
    stopped = False; child = None
    def stop(*_):
        nonlocal stopped
        stopped = True
        if child is not None and child.poll() is None: child.send_signal(signal.SIGINT)
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    write(OUT/'status.json', dict(phase='queued', pid=os.getpid(), time=time.time()))
    while not stopped:
        previous = json.loads((SOURCE_OUT/'launch-status.json').read_text())
        if previous['phase'] == 'completed':
            active = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
            if not active.strip():break
        if previous['phase'] in ['cancelled','stopped_or_failed','failed']:
            write(OUT/'status.json', dict(phase='prerequisite_failed', previous=previous));return 1
        time.sleep(5)
    if stopped:write(OUT/'status.json', dict(phase='cancelled'));return 130
    environment = dict(os.environ, **json.loads((BASE/'environment.json').read_text()))
    command = [sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=4',str(Path(__file__).resolve()),'evaluate']
    with (OUT/'evaluate.log').open('a') as log:
        child = subprocess.Popen(command, cwd=BASE, env=environment, stdout=log, stderr=subprocess.STDOUT)
        write(OUT/'status.json', dict(phase='evaluating', pid=os.getpid(), child_pid=child.pid, time=time.time()))
        result = child.wait()
    write(OUT/'status.json', dict(phase='completed' if result == 0 else 'failed', returncode=result, time=time.time()))
    return result


def evaluate():
    import torch
    import torch.distributed as dist
    from omegaconf import OmegaConf
    from src.training.laser_compound_replay_recipe import CompoundReplayRecipe
    from src.training import rq_imagenet_shared as shared
    rank,world,local = (int(os.environ[k]) for k in ['RANK','WORLD_SIZE','LOCAL_RANK'])
    assert world == 4
    torch.set_num_threads(4); device = torch.device('cuda',local); torch.cuda.set_device(device)
    dist.init_process_group('nccl',device_id=device)
    plan = json.loads((OUT/'plan.json').read_text())
    profile = CompoundReplayRecipe(BASE)
    saved = torch.load(plan['checkpoint'], mmap=True, map_location='cpu', weights_only=False)
    profile.verify_resume(saved)
    epoch = int(plan['source_epoch'])
    assert saved['global_step'] == 626*epoch and saved['next_epoch'] == epoch and saved['next_microbatch'] == 0
    arch = OmegaConf.create(saved['config']['arch'])
    with torch.device('meta'): model = profile.model(arch)
    model.load_state_dict(saved['state_dict'], strict=True, assign=True)
    assert len(list(model.parameters())) == 837
    del saved
    model.eval().requires_grad_(False).to(device)
    aux,_ = profile.tokenizer(device)
    original = OmegaConf.to_container(profile.recipe.sampling, resolve=True)
    assert original == plan['source_sampling']
    profile.recipe.sampling.atom_top_k = 256
    candidate = OmegaConf.to_container(profile.recipe.sampling, resolve=True)
    assert [key for key in original if original[key] != candidate[key]] == ['atom_top_k']
    name = f'Compound512 epoch{epoch} | atom top-k256 only | fixed weights | FID50k'
    run = None
    if rank == 0:
        import wandb
        os.environ['WANDB_API_KEY'] = Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip()
        run = wandb.init(entity='helloimlixin-rutgers',project='laser',id=OUT.name,name=name,
            dir=str(BASE/'wandb'),config=dict(plan,wandb_id=OUT.name,sampling=candidate,
                fid_num_samples=50000,fid_reference_images=1281167,fid_real_split='train',weights_unmodified=True))
        run.define_metric('train/epoch')
        run.define_metric('eval/fid',step_metric='train/epoch',overwrite=True)
        write(OUT/'wandb.json',dict(url=run.url,id=OUT.name))
    shared.BASE,shared.OUT,shared.PROFILE,shared.RUN = BASE,OUT,profile,OUT.name
    shared.original_metrics.INCEPTION_PATH = BASE/'upstream/rqvae/metrics/inception.py'
    if rank == 0: shared.fid_workspace(run)
    dist.barrier()
    before = profile.digest(aux)
    started = time.time()
    result = shared.generation(model,aux,device,rank,world,epoch,run,batch=128,seed=261001)
    assert profile.digest(aux) == before
    if rank == 0:
        result.update(generated_images=50000,real_images=1281167,real_split='train',
            source_step=626*epoch,source_epoch=epoch,source_checkpoint_sha256=plan['checkpoint_sha256'],
            source_sampling=original,sampling=candidate,weights_unmodified=True,training_updates_added=0,
            rejection_sampling=False,generation_seed=261001,reference_fid=plan['reference_fid'],
            fid_delta=result['fid']-plan['reference_fid'],elapsed_seconds=time.time()-started)
        write(OUT/'fid50k.json',result)
        run.log({'train/epoch':epoch,'train/global_step':626*epoch,'eval/fid':result['fid'],'eval/inception_score':result['inception_score']})
        run.save(str(OUT/'fid50k.json'),base_path=str(OUT),policy='now');run.finish()
        print(json.dumps(result),flush=True)
    dist.barrier();dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['queue','evaluate'])
    parser.add_argument('--epoch',type=int,choices=[1,2],default=1)
    args = parser.parse_args()
    if args.epoch != 1:
        assert args.action == 'evaluate', 'Epoch-two validation uses an explicit saved-checkpoint plan'
        OUT = ROOT/f'outputs/imagenet-compound512-epoch{args.epoch}-atomtopk256-fid50k-20261016'
    if args.action == 'queue': raise SystemExit(queue())
    evaluate()
