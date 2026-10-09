"""Evaluate one immutable full-history checkpoint using official RQ metrics."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def record(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str)+'\n')
    temporary.replace(path)


def load_model(args, device):
    import torch
    from src.training.rqtransformer import LaserAux, build_model
    from src.models.physical_compound_prior import PhysicalCompoundRQTransformer
    payload = torch.load(args.source, map_location='cpu', mmap=True, weights_only=False)
    if payload['global_step'] != args.source_step or payload['epoch'] != 79:
        raise RuntimeError('Evaluation checkpoint identity changed')
    trial = payload['compound_energy_trial']
    if trial['branch'] != 'geometry' or trial['source_step'] != 49454:
        raise RuntimeError('Expected the geometry fork of the requested FID15.08 source')
    config = payload['config']
    aux = LaserAux(args.base/'inputs/resume-stage1-tokenizer.pt', config['num_atoms'],
        config['coeff_vocab_size'], config['coeff_max'], config['coeff_scale'],
        coeff_scales=config['coeff_scales'], soft_target_physical=False,
        clamp_coeffs=False, sparsity_level=4).to(device).eval()
    native_decode = aux.decode_tokens
    def bounded_decode(tokens):
        return torch.cat([native_decode(chunk) for chunk in tokens.split(args.decode_batch)], dim=0)
    aux.decode_tokens = bounded_decode
    with torch.device('meta'):
        model = PhysicalCompoundRQTransformer.from_scalar(build_model(
            config['num_atoms']+config['coeff_vocab_size'],config['num_atoms'],
            physical_pair_context=True,sparsity_level=4,coeff_vocab_size=config['coeff_vocab_size'],
            model_preset=config['model_preset']))
    model.load_state_dict(payload['state_dict'], strict=True, assign=True)
    model.requires_grad_(False).to(device).eval()
    return model, aux, config


def worker(args):
    runtime = args.base/'source/runtime'
    sys.path[:0] = [str(runtime),str(args.base/'support')]
    import torch
    import torch.distributed as dist
    from src.training.compound_geometry_trial_hook import seeded_official_evaluation
    from src.training.rqtransformer import evaluate_generation_metrics, val_image_transform
    from official_metrics import install
    local_rank = int(os.environ.get('LOCAL_RANK','0'))
    rank = int(os.environ.get('RANK','0'))
    device = torch.device('cuda',local_rank)
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(args.memory_limit_gib*2**30/total,device)
    if args.action == 'evaluate':
        if int(os.environ['WORLD_SIZE']) != 8:
            raise RuntimeError('Official protocol requires eight evaluation ranks')
        dist.init_process_group('nccl',timeout=timedelta(hours=3))
    model,aux,config = load_model(args,device)
    if args.action == 'probe':
        with torch.inference_mode():
            tokens=model.sample_sparse(args.generation_batch,aux,
                cond=torch.arange(args.generation_batch,device=device),
                atom_temperature=.9,atom_top_p=.9,coeff_temperature=1.,coeff_top_p=.85,amp=True)
            images=aux.decode_tokens(tokens)
            if not torch.isfinite(images).all():
                raise RuntimeError('Nonfinite checkpoint probe images')
        record(args.output/'memory-probe.json',dict(passed=True,quality_metric_computed=False,
            source_step=args.source_step,generation_batch=args.generation_batch,decode_batch=args.decode_batch,
            allocator_limit_gib=args.memory_limit_gib,
            maximum_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
            maximum_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30,time=time.time()))
        return
    install(runtime)
    from torch.utils.data import DataLoader,DistributedSampler
    from torchvision.datasets import ImageFolder
    dataset=ImageFolder(Path(config['data'])/'val',transform=val_image_transform())
    if len(dataset)!=50000 or len(dataset.classes)!=1000:
        raise RuntimeError('Expected the existing local ImageNet validation50k data')
    loader=DataLoader(dataset,batch_size=64,
        sampler=DistributedSampler(dataset,num_replicas=8,rank=rank,shuffle=False,drop_last=False),
        num_workers=2,pin_memory=True)
    run=None
    if rank==0:
        os.environ['WANDB_API_KEY']=args.key_file.read_text().strip()
        import wandb
        directory=args.output/'wandb'
        directory.mkdir(exist_ok=True)
        run=wandb.init(entity='helloimlixin-rutgers',project='laser',id=args.run_id,
            name=f'ImageNet | geometry step{args.source_step} | official50k',
            resume='allow',mode='online',dir=str(directory),
            config=json.loads((args.output/'evaluation-plan.json').read_text()))
        run.summary['evaluation/state']='official50k_running'
    started=time.monotonic()
    result=seeded_official_evaluation(evaluate_generation_metrics,model,aux,loader,50000,args.generation_batch,
        seed=args.seed,process_rank=rank,num_condition_classes=1000,
        atom_temperature=.9,atom_top_k=0,atom_top_p=.9,
        coeff_temperature=1.,coeff_top_k=0,coeff_top_p=.85,
        compute_inception_score=True,metric_backend='original-rqvae',fid_reference_stats=None)
    report=dict(fid=result[0],inception_score=result[1],inception_score_std=result[2],
        checkpoint=str(args.source),source_epoch=79,source_global_step=args.source_step,
        trained_updates_since_source=args.source_step-49454,seed=args.seed,
        metric_backend='original_rqtransformer',real_images=50000,generated_images=50000,
        real_split='val',inception_splits=10,world_size=8,generation_batch_per_rank=args.generation_batch,
        decode_batch=args.decode_batch,sampler='native ancestral',
        sampling=dict(atom_temperature=.9,atom_top_p=.9,coefficient_temperature=1.,coefficient_top_p=.85),
        frozen_weights=True,elapsed_seconds=time.monotonic()-started,time=time.time())
    record(args.output/f'memory-rank{rank}.json',dict(
        maximum_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
        maximum_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30,allocator_limit_gib=args.memory_limit_gib))
    if rank==0:
        record(args.output/'official-result.json',report)
        run.log({'train/global_step':args.source_step,
            'eval/fid_original_rqtransformer':result[0],
            'eval/inception_score_original_rqtransformer':result[1],
            'eval/inception_score_std_original_rqtransformer':result[2]})
        run.summary['evaluation/official_result']=report
        run.summary['evaluation/state']='completed'
        run.finish()
        from verified_wandb_checkpoint_upload import VerifiedCloudUpload
        VerifiedCloudUpload('helloimlixin-rutgers/laser/'+args.run_id,args.output/'online-result-receipt.json')(
            [args.output/'official-result.json',args.output/'evaluation-plan.json',
             args.output/'memory-probe.json',Path(__file__)],79)
        print(json.dumps(report),flush=True)
    dist.barrier()
    dist.destroy_process_group()


def supervise(args):
    import fcntl
    lock=(args.output/'evaluation.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    kwargs=['--base',str(args.base),'--output',str(args.output),'--source',str(args.source),
        '--source-step',str(args.source_step),'--key-file',str(args.key_file),'--run-id',args.run_id,
        '--generation-batch',str(args.generation_batch),'--decode-batch',str(args.decode_batch),
        '--memory-limit-gib',str(args.memory_limit_gib),'--seed',str(args.seed)]
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=8',
        str(Path(__file__).resolve()),'evaluate',*kwargs]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',TORCH_HOME=str(args.base/'torch-cache'),
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1',
        NCCL_NVLS_ENABLE='0',TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
    with (args.output/'evaluation.log').open('a') as stream:
        child=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        record(args.output/'status.json',dict(state='running',supervisor_pid=os.getpid(),
            torchrun_pid=child.pid,source_step=args.source_step,official_images=50000,time=time.time()))
        code=child.wait()
    complete=code==0 and (args.output/'official-result.json').exists()
    record(args.output/'status.json',dict(state='completed' if complete else 'failed',
        exit_code=code,source_step=args.source_step,time=time.time()))
    if not complete:
        raise RuntimeError('Official checkpoint evaluation failed; inspect evaluation.log')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['probe','evaluate','supervise'])
    for name in ('base','output','source','key-file'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--source-step',type=int,required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--generation-batch',type=int,default=64)
    parser.add_argument('--decode-batch',type=int,default=4)
    parser.add_argument('--memory-limit-gib',type=float,default=9.)
    parser.add_argument('--seed',type=int,default=261001)
    args=parser.parse_args()
    if args.action=='supervise':supervise(args)
    else:worker(args)


if __name__=='__main__':
    main()
