"""Recover the original trial, changing execution topology and run identities only."""
from pathlib import Path
import hashlib,json,shutil,zipfile,difflib
import yaml
BASE=Path('/tmp/laser-beam16-duplicate-20261105')
OUT=Path('/workspace/Projects/laser/outputs/imnet-beam16-duplicate-20261105')
HERE=Path(__file__).resolve().parent

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
 original=(BASE/'original-runner.py').read_text()
 code=original
 def replace(a,b):
  nonlocal code
  assert a in code,a
  code=code.replace(a,b)
 replace('def read(path):', '''def rank():
    return int(os.environ.get('RANK', '0'))


def world():
    return int(os.environ.get('WORLD_SIZE', '1'))


def read(path):''')
 replace('def write(path, value):\n    path = Path(path)', 'def write(path, value):\n    if rank() != 0: return\n    path = Path(path)')
 replace("    torch.set_num_threads(2)", "    torch.set_num_threads(2)\n    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))\n    if world() > 1 and not torch.distributed.is_initialized():\n        from datetime import timedelta\n        torch.distributed.init_process_group('nccl', timeout=timedelta(hours=2))")
 replace('    install_bf16_training()','    install_bf16_training()\n    from runtime.training_acceleration import install\n    install()')
 replace("device = torch.device('cuda', 0)","device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', '0')))")
 replace("math.ceil(config['global_batch']/config['microbatch'])", "math.ceil(config['global_batch']/(config['microbatch']*world()))")
 replace("noise = torch.Generator(device=device).manual_seed(config['noise_seed'])", "noise = torch.Generator(device=device).manual_seed(config['noise_seed'] + rank())")
 replace("noise.set_state(payload['noise_rng'])", "noise.set_state(payload.get('noise_rng_by_rank', [payload['noise_rng']])[rank()])")
 replace("restore_rng(payload['rng_state_by_rank'][0], device)", "restore_rng(payload['rng_state_by_rank'][rank() if resume else 0], device)")
 replace('runtime.update(world_size=1,', "runtime.update(hardware='2 NVIDIA RTX PRO 6000 Blackwell 96GB per arm', training_attention='SDPA for training; native cached sampling', topology_change_is_not_bitwise_replay=True, world_size=world(),")
 replace('arm=arm, world_size=1)', 'arm=arm, world_size=world(), execution_topology_changed=True, bitwise_replay=False)')
 replace("ExactGlobalBatchSampler(dataset, config['global_batch'], 1, 0,", "ExactGlobalBatchSampler(dataset, config['global_batch'], world(), rank(),")
 replace("(folder/'training.lock').open('a')", "(folder/f'training-rank{rank()}.lock').open('a')")
 replace('    if not smoke_updates:\n        import wandb', '    if not smoke_updates and rank() == 0:\n        import wandb')
 replace("run_id = f'imnet-beam16-{arm}-s{config[\"source_step\"]}-2ep-1b200-20261105'", "run_id = f'imnet-beam16-{arm}-s{config[\"source_step\"]}-2ep-2rtx6k-dup-20261105'")
 replace("name=f'ImageNet beam16 | {arm} | matched +2ep | frozen tokenizer'", "name=f'ImageNet beam16 | {arm} | matched +2ep | 2 RTX6000 duplicate'")
 replace("job_type='beam-consistency-trial',resume='allow'", "job_type='beam-consistency-duplicate',group='imnet-beam16-matched-duplicate-20261105',resume='allow'")
 replace("source_artifact=read(local/'source-checkpoint-receipt.json')['artifact']))", "source_artifact=read(local/'source-checkpoint-receipt.json')['artifact'],\n                duplicate_of='helloimlixin-rutgers/laser/imnet-beam16-consistency-s10750-2ep-1b200-20261105',\n                execution=dict(gpus_per_arm=world(),hardware='RTX PRO 6000 Blackwell 96GB',attention='SDPA for training; native sampling',bitwise_replay=False)))")
 replace('''        nonlocal save_future
        if save_future is not None:''', '''        nonlocal save_future
        snapshots = [None] * world()
        current_rng = dict(rng=rng_state(state['device']), noise=state['noise'].get_state())
        if world() > 1:
            torch.distributed.all_gather_object(snapshots, current_rng)
        else:
            snapshots[0] = current_rng
        if rank() != 0: return
        if save_future is not None:''')
 replace("rng_state_by_rank=[rng_state(state['device'])],noise_rng=state['noise'].get_state(),", "rng_state_by_rank=[s['rng'] for s in snapshots],noise_rng=state['noise'].get_state(),\n            noise_rng_by_rank=[s['noise'] for s in snapshots],")
 replace("                norm,retries=recoverable_amp_step", "                synchronize_gradients(state['parameters'])\n                norm,retries=recoverable_amp_step")
 replace("                next_epoch,next_cursor=epoch,absolute+1", "                count, means = global_metrics(count, means, state['device'])\n                next_epoch,next_cursor=epoch,absolute+1")
 replace("                if updates<=3 or updates%10==0:", "                if rank() == 0 and (updates<=3 or updates%10==0):")
 replace("            run.summary['training/status']='completed'\n            run.save(str(folder/'completed.json'),base_path=str(folder),policy='now')\n            run.save(str(folder/'evaluation.json'),base_path=str(folder),policy='now')\n            run.finish();run=None", "            if run is not None:\n                run.summary['training/status']='completed'\n                run.save(str(folder/'completed.json'),base_path=str(folder),policy='now')\n                run.save(str(folder/'evaluation.json'),base_path=str(folder),policy='now')\n                run.finish();run=None")
 # Each rank generates 25k samples; metric implementation reduces exact total of 50k.
 replace("torch.manual_seed(settings['seed'])", "torch.manual_seed(settings['seed'] + rank())")
 replace("        for first in range(0,50000,settings['batch']):\n            count = min(settings['batch'],50000-first)\n            labels = torch.arange(first,first+count,device=device)%1000", "        for first in range(0,50000//world(),settings['batch']):\n            count = min(settings['batch'],50000//world()-first)\n            labels = (torch.arange(first,first+count,device=device)*world()+rank())%1000")
 replace("generated_images=first+count,target=50000", "generated_images=min(50000,(first+count)*world()),target=50000")
 replace("assert int(metric.fake_count)==50000", "assert int(metric.fake_count)==50000//world()")
 replace("        fid, inception, std = metric.compute(shuffle_for_inception=True)", "        fid, inception, std = metric.compute(shuffle_for_inception=True)\n        assert int(metric.fake_count)==50000")
 replace("seed=settings['seed'],inception_splits=10", "seed=settings['seed'],rank_seed_offset=True,world_size=world(),inception_splits=10")
 replace("if updates%config['save_every_updates']==0 and not smoke_updates:", "if (updates == 10 or updates%config['save_every_updates']==0) and not smoke_updates:")
 # Validate synchronization on a real optimizer update in preflight.
 replace("        if smoke_updates:\n            save_checkpoint();save_future.result()", "        if smoke_updates:\n            verify_replica_parameters(state)\n            save_checkpoint()\n            if save_future is not None:save_future.result()")
 replace("            save_checkpoint(evaluation);save_future.result()", "            save_checkpoint(evaluation)\n            if save_future is not None:save_future.result()")
 replace("        if not (local/'calibration.json').exists():", "        if not (local/'calibration.json').exists():")
 # Only --mode train is used by the separate parallel supervisor.
 extra='''
def synchronize_gradients(parameters):
    """SUM already global-batch-normalized gradients once per optimizer update."""
    import torch
    if world() == 1: return
    bucket = []
    size = 0
    def flush():
        if not bucket: return
        flat = torch._utils._flatten_dense_tensors([p.grad for p in bucket])
        torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.SUM)
        for p, reduced in zip(bucket, torch._utils._unflatten_dense_tensors(flat, [p.grad for p in bucket])):
            p.grad.copy_(reduced)
    for p in parameters:
        if p.grad is None: continue
        if size + p.numel() > 32 * 1024**2 and bucket:
            flush();bucket=[];size=0
        bucket.append(p);size+=p.numel()
    flush()


def global_metrics(count, means, device):
    import torch
    keys = list(means)
    values = torch.tensor([count] + [count * means[k] for k in keys], dtype=torch.float64, device=device)
    if world() > 1:torch.distributed.all_reduce(values)
    total = int(values[0])
    return total, {k:float(values[i+1]/total) for i,k in enumerate(keys)}


def verify_replica_parameters(state):
    import torch
    maximum = 0.
    for p in state['parameters']:
        reference = p.detach().clone()
        if world() > 1:torch.distributed.broadcast(reference, src=0)
        maximum = max(maximum, float((reference - p).abs().max()))
    assert maximum == 0., maximum
    write(state['local']/'replica-verification.json', dict(passed=True, maximum_parameter_error=maximum, world_size=world()))

'''
 replace('\ndef main():', extra+'\ndef main():')
 (BASE/'runner.py').write_text(code)
 (OUT/'runtime.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),code.splitlines(True),fromfile='original/runner.py',tofile='duplicate/runner.py')))
 config=yaml.safe_load((BASE/'config.yaml').read_text())
 config.update(local=str(BASE),output=str(OUT),source_base=str(BASE/'frozen'),microbatch=64,memory_gib=88,loader_workers=6)
 (BASE/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False));shutil.copyfile(BASE/'config.yaml',OUT/'config.yaml')
 plan=json.loads((BASE/'plan.json').read_text());plan['configuration']=config
 plan.update(execution_topology=dict(world_size_per_arm=2,concurrent_arms=2,microbatch=64,accumulation_steps=16,global_batch=2048,bitwise_replay=False))
 (BASE/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');shutil.copyfile(BASE/'plan.json',OUT/'plan.json')
 checkpoint=BASE/'checkpoint-artifact/last.pt'
 assert sha(checkpoint)==plan['source_checkpoint_sha256']
 stage1=BASE/'stage1-artifact/best_rfid_slot1_model.pt'
 assert sha(stage1)=='dd28db9306d526bdc9fbf8016403e106c4f317fd8f5c04792f0953e53310fdab'
 for dest,source in [(BASE/'source-checkpoint.pt',checkpoint),(BASE/'frozen/inputs/stage1-model.pt',stage1)]:
  if not dest.exists():dest.hardlink_to(source)
 receipt=dict(artifact=plan['source_artifact'],global_step=10750,sha256=plan['source_checkpoint_sha256'])
 (BASE/'source-checkpoint-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 shutil.copyfile(BASE/'calibration.json',OUT/'original-calibration.json')
 shutil.copyfile(BASE/'runner.py',OUT/'runner.py')
 shutil.copyfile(BASE/'original-runner.py',OUT/'original-runner.py')
 print('PREPARED',flush=True)

if __name__=='__main__':main()
