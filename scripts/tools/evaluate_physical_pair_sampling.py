"""Run or queue official50k fixed-checkpoint dictionary sampler comparisons."""
import argparse
from dataclasses import asdict
from datetime import timedelta
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import types

RUN_ID='imagenet-rfid421-epoch77-dictionary-sampling-official50k-8h100-20261006'


def record(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2,default=str)+'\n');temporary.replace(path)


def load_helper(path):
    spec=importlib.util.spec_from_file_location('frozen_pair_sampling',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def policies(helper):
    return {
        'native':helper.PairSamplingPolicy(mode='ancestral'),
        'depth-sharp':helper.PairSamplingPolicy(mode='ancestral',
            coefficient_temperatures=(.55,.7,.85,1.)),
        'joint-prior':helper.PairSamplingPolicy(mode='joint',atom_proposal='prior',candidate_atoms=32),
        'joint-prior-sharp':helper.PairSamplingPolicy(mode='joint',atom_proposal='prior',candidate_atoms=32,
            coefficient_temperatures=(.55,.7,.85,1.)),
    }


def supervise(args):
    import fcntl
    lock=(args.output/'official-comparison.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    child=None
    def stop(_sig,_frame):
        if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
        record(args.output/'official-comparison-status.json',dict(state='cancelled',time=time.time()))
        raise SystemExit(0)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    deadline=time.monotonic()+24*3600
    while args.action=='queue' and time.monotonic()<deadline:
        source=args.training_root/'global-cosine-trial-status.json'
        state=json.loads(source.read_text()) if source.exists() else {}
        ready=state.get('state')=='completed' and state.get('final_step')==62600
        free=False
        if ready:
            result=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],
                capture_output=True,text=True,check=True)
            free=not result.stdout.strip()
        record(args.output/'official-comparison-status.json',dict(state='waiting_for_epoch100_and_free_gpus',
            training_state=state.get('state'),training_target=100,gpus_free=free,source_epoch=77,
            official_images=50000,policies=['native','depth-sharp','joint-prior','joint-prior-sharp'],time=time.time()))
        if ready and free:break
        if state.get('state')=='failed':raise RuntimeError('Training failed; sampler queue did not take its GPUs')
        time.sleep(30)
    else:
        if args.action=='queue':raise RuntimeError('Sampler queue expired while waiting for epoch100')
    if args.action=='now':
        if args.memory_limit_gib<=0:raise ValueError('Immediate concurrent evaluation requires a memory cap')
        usage=subprocess.run(['nvidia-smi','--query-gpu=memory.total,memory.used',
            '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        available=[(float(row.split(',')[0])-float(row.split(',')[1]))/1024
            for row in usage.stdout.strip().splitlines()]
        if len(available)!=8 or min(available)<args.memory_limit_gib+1.5:
            raise RuntimeError(f'Insufficient GPU headroom for immediate evaluation: {available}')
        record(args.output/'immediate-memory-headroom.json',dict(available_gib=available,
            per_process_allocator_limit_gib=args.memory_limit_gib,training_unchanged=True,time=time.time()))
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=8',
        str(Path(__file__).resolve()),'evaluate','--base',str(args.base),'--output',str(args.output),
        '--training-root',str(args.training_root),'--key-file',str(args.key_file),
        '--helper',str(args.helper),'--generation-batch-size',str(args.generation_batch_size),
        '--decode-batch-size',str(args.decode_batch_size),'--memory-limit-gib',str(args.memory_limit_gib)]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',PYTHONUNBUFFERED='1',
        TORCH_HOME=str(args.base/'torch-cache'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='4',TORCH_NCCL_ASYNC_ERROR_HANDLING='1',NCCL_NVLS_ENABLE='0')
    with (args.output/'official-comparison.log').open('a') as stream:
        child=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        record(args.output/'official-comparison-status.json',dict(state='running',torchrun_pid=child.pid,
            source_epoch=77,official_images=50000,concurrent_training=args.action=='now',
            generation_batch_per_rank=args.generation_batch_size,decode_batch=args.decode_batch_size,
            memory_limit_gib=args.memory_limit_gib,time=time.time()))
        code=child.wait()
    record(args.output/'official-comparison-status.json',dict(state='completed' if code==0 else 'failed',
        exit_code=code,source_epoch=77,official_images=50000,time=time.time()))
    if code:raise RuntimeError('Official sampler comparison failed; inspect its log')


def evaluate(args):
    sys.path[:0]=[str(args.base/'source/runtime'),str(args.base/'support')]
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader,DistributedSampler
    from torchvision.datasets import ImageFolder
    from src.training.rqtransformer import LaserAux,build_model,val_image_transform,evaluate_generation_metrics,save_class_labeled_grid
    from src.data.imagenet_labels import class_names_for_dataset
    from official_metrics import install
    local_rank=int(os.environ['LOCAL_RANK']);rank=int(os.environ['RANK'])
    assert int(os.environ['WORLD_SIZE'])==8
    device=torch.device('cuda',local_rank);torch.cuda.set_device(device);torch.set_num_threads(4)
    if args.memory_limit_gib:
        total=torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(args.memory_limit_gib*2**30/total,device)
    dist.init_process_group('nccl',timeout=timedelta(hours=3))
    helper=load_helper(args.helper);settings=policies(helper)
    source=args.base/'inputs/source-epoch077-full.pt'
    payload=torch.load(source,map_location='cpu',mmap=True,weights_only=False)
    assert (payload['epoch'],payload['global_step'])==(77,48202)
    assert payload['original_rqtransformer_metrics']['fid']==15.24941539209243
    config=payload['config']
    aux=LaserAux(args.base/'inputs/resume-stage1-tokenizer.pt',config['num_atoms'],config['coeff_vocab_size'],
        config['coeff_max'],config['coeff_scale'],coeff_scales=config['coeff_scales'],
        soft_target_physical=False,clamp_coeffs=False,sparsity_level=4).to(device).eval()
    native_decode=aux.decode_tokens
    def bounded_decode(tokens):
        return torch.cat([native_decode(chunk) for chunk in tokens.split(args.decode_batch_size)],dim=0)
    aux.decode_tokens=bounded_decode
    with torch.device('meta'):
        model=build_model(config['num_atoms']+config['coeff_vocab_size'],config['num_atoms'],
            physical_pair_context=True,sparsity_level=4,coeff_vocab_size=config['coeff_vocab_size'],
            model_preset=config['model_preset'])
    model.load_state_dict(payload['state_dict'],strict=True,assign=True)
    model.requires_grad_(False).to(device).eval();del payload
    install(args.base/'source/runtime')
    dataset=ImageFolder(Path(config['data'])/'val',transform=val_image_transform())
    assert len(dataset)==50000 and len(dataset.classes)==1000
    loader=DataLoader(dataset,batch_size=64,sampler=DistributedSampler(dataset,num_replicas=8,rank=rank,
        shuffle=False,drop_last=False),num_workers=4,pin_memory=True)
    names=class_names_for_dataset('imagenet',dataset.classes)
    run=None
    if rank==0:
        os.environ['WANDB_API_KEY']=args.key_file.read_text().strip()
        import wandb
        wandb_dir=args.base/'sampling-evaluation-wandb'
        wandb_dir.mkdir(parents=True,exist_ok=True)
        run=wandb.init(entity='helloimlixin-rutgers',project='laser',id=RUN_ID,resume='must',mode='online',
            dir=str(wandb_dir),config={'sampler_policies':{k:asdict(v) for k,v in settings.items()},
                'generation_batch_per_rank':args.generation_batch_size,'decode_batch':args.decode_batch_size,
                'memory_limit_gib':args.memory_limit_gib,
                'execution_mode':'immediate_concurrent_training' if args.memory_limit_gib else 'dedicated_gpus',
                'wait_for_training_epoch':None if args.memory_limit_gib else 100},allow_val_change=True)
        run.summary['execution/state']='evaluating_official50k'
        run.summary['evaluation/official_results_pending']=True
    results={}
    for name,policy in settings.items():
        started=time.monotonic()
        if rank==0:
            current=json.loads((args.output/'official-comparison-status.json').read_text())
            current.update(state='running',active_policy=name,completed_policies=list(results),time=time.time())
            record(args.output/'official-comparison-status.json',current)
            print(json.dumps(dict(phase='official_sampler_begin',sampler=name,
                generated_images=50000,generation_batch_per_rank=args.generation_batch_size)),flush=True)
        def sample(self,batch_size,model_aux,cond=None,**kwargs):
            return helper.sample_physical_pairs(self,batch_size,model_aux,cond,policy=policy,amp=kwargs.get('amp',True))
        model.sample_sparse=types.MethodType(sample,model)
        with torch.random.fork_rng(devices=[local_rank]):
            torch.manual_seed(261001+rank);torch.cuda.manual_seed(261001+rank)
            fid,score,std=evaluate_generation_metrics(model,aux,loader,num_samples=50000,batch_size=args.generation_batch_size,
                num_condition_classes=1000,atom_temperature=.9,atom_top_p=.9,
                coeff_temperature=1.,coeff_top_p=.85,metric_backend='original-rqvae',compute_inception_score=True)
        result=dict(fid=fid,inception_score=score,inception_score_std=std,source_epoch=77,
            source_global_step=48202,metric_backend='original_rqtransformer',generated_images=50000,
            real_images=50000,real_split='val',inception_splits=10,seed=261001,
            world_size=8,generation_batch_per_rank=args.generation_batch_size,decode_batch=args.decode_batch_size,
            memory_limit_gib=args.memory_limit_gib,elapsed_seconds=time.monotonic()-started,policy=asdict(policy))
        record(args.output/f'memory-{name}-rank{rank}.json',dict(
            max_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
            max_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30,
            allocator_limit_gib=args.memory_limit_gib,time=time.time()))
        results[name]=result
        if rank==0:
            record(args.output/f'official-{name}.json',result)
            run.log({f'eval/{name}/fid_original_rqtransformer':fid,
                f'eval/{name}/inception_score_original_rqtransformer':score,
                f'eval/{name}/inception_score_std_original_rqtransformer':std,
                'sampler/policy':name})
            run.save(str(args.output/f'official-{name}.json'),base_path=str(args.output),policy='now')
            run.summary[f'eval/{name}/fid_original_rqtransformer']=fid
            run.summary[f'eval/{name}/inception_score_original_rqtransformer']=score
            run.summary[f'eval/{name}/inception_score_std_original_rqtransformer']=std
            print(json.dumps(dict(sampler=name,**result)),flush=True)
        # A fixed class grid accompanies each official evaluation.
        classes=torch.tensor([269,612,265,628,689,301,932,798],device=device)
        with torch.inference_mode(),torch.random.fork_rng(devices=[local_rank]):
            torch.manual_seed(261006+rank);torch.cuda.manual_seed(261006+rank)
            tokens=model.sample_sparse(8,aux,cond=classes[rank:rank+1].repeat(8),amp=True)
            images=aux.decode_tokens(tokens).float().add(1).mul(.5).clamp(0,1)
            gathered=[torch.empty_like(images) for _ in range(8)] if rank==0 else None
            dist.gather(images,gather_list=gathered,dst=0)
            if rank==0:
                path=args.output/f'official-{name}-samples.png'
                save_class_labeled_grid(torch.cat(gathered).cpu(),classes.cpu(),names,path,samples_per_class=8)
                import wandb
                run.log({f'samples/{name}':wandb.Image(str(path))})
        dist.barrier()
    if rank==0:
        record(args.output/'official-comparison-results.json',results)
        run.save(str(args.output/'official-comparison-results.json'),base_path=str(args.output),policy='now')
        run.summary['execution/state']='completed'
        run.summary['evaluation/official_results_pending']=False
        run.summary['evaluation/official_evaluation_completed']=True
        run.finish(exit_code=0)
    dist.destroy_process_group()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['queue','now','evaluate'])
    for name in ['base','output','training-root','key-file','helper']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--generation-batch-size',type=int,default=512)
    parser.add_argument('--decode-batch-size',type=int,default=64)
    parser.add_argument('--memory-limit-gib',type=float,default=0.)
    args=parser.parse_args()
    if args.generation_batch_size<1 or args.decode_batch_size<1 or args.memory_limit_gib<0:
        parser.error('Batch sizes must be positive and memory limit nonnegative')
    if args.action in ('queue','now'):supervise(args)
    else:evaluate(args)


if __name__=='__main__':main()
