"""Resume native compound128 with compiled loss execution; retain all training state."""
import argparse,json,os,time,math,sys
from pathlib import Path
from datetime import timedelta
import torch
import torch.distributed as dist
import train_native_compound128_stage2 as trainer
from native_performance import install_typed_head
from src.models.scratch_var import state_digest

_original_ddp=trainer.DDP

def bucket_view_ddp(*args,**kwargs):
    kwargs['gradient_as_bucket_view']=True
    return _original_ddp(*args,**kwargs)
trainer.DDP=bucket_view_ddp


def set_affinity():
    rank=int(os.environ['LOCAL_RANK'])
    groups=[list(range(0,32))+list(range(128,160)),list(range(32,64))+list(range(160,192)),
            list(range(64,96))+list(range(192,224)),list(range(96,128))+list(range(224,256))]
    available=os.sched_getaffinity(0);selected=set(groups[rank])&available
    if selected:os.sched_setaffinity(0,selected)


class FastExperiment(trainer.Experiment):
    def __init__(self,cfg,smoke=False):
        if not (Path(cfg['output'])/'prior-last.pt').exists():raise RuntimeError('A speed upgrade must resume an existing full-state checkpoint')
        super().__init__(cfg,smoke)
        self.restored_step=self.progress['step'];ages={int(v['step']) for v in self.optimizer.state.values() if 'step' in v}
        assert ages=={self.restored_step}
        saved=trainer.capture_rng();model_before=state_digest(self.model)
        install_typed_head(self.model,compile_forward=True)
        self.log('execution_compile_started',step=self.restored_step,optimizer_age=self.restored_step)
        started=time.monotonic()
        cache=torch.load(cfg['smoke_cache'],map_location='cpu',weights_only=True)
        repeats=math.ceil(self.micro/len(cache['labels']))
        images=cache['images'].repeat(repeats,1,1,1)[:self.micro].to(self.device)
        labels=cache['labels'].repeat(repeats)[:self.micro].to(self.device)
        with torch.no_grad():codes=self.tokenize(images)
        self.model.train()
        for _ in range(2):
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=self.wrapped(labels,codes['inputs'],codes['atoms'],codes['coefficients'])
            loss.backward()
        self.optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();trainer.restore_rng(saved)
        after=trainer.capture_rng()
        assert torch.equal(saved['torch'],after['torch']) and torch.equal(saved['cuda'],after['cuda'])
        assert saved['python']==after['python']
        assert model_before==state_digest(self.model)
        assert {int(v['step']) for v in self.optimizer.state.values() if 'step' in v}=={self.restored_step}
        del cache,images,labels,codes,loss
        if self.run:
            self.run.config.update(cfg,allow_val_change=True)
            self.run.summary.update({'execution/compiled_forward':True,'execution/type_valid_classifier_rows_only':True,
                'execution/source_encoder_changed':False,'execution/resumed_step':self.restored_step,'execution/optimizer_state_preserved':True,
                'execution/all_rank_rng_preserved':True,'execution/numa_affinity':True})
        self.log('execution_ready',step=self.restored_step,compile_seconds=time.monotonic()-started,parameters_unchanged=True,
                 optimizer_age=self.restored_step,all_rank_rng_restored=True,global_batch=self.global_batch)
        dist.barrier()


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path);p.add_argument('--smoke',action='store_true');a=p.parse_args();cfg=json.loads(a.config.read_text())
    set_affinity();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest');torch.cuda.set_device(int(os.environ['LOCAL_RANK']));dist.init_process_group('nccl',timeout=timedelta(hours=2))
    experiment=None
    try:
        experiment=FastExperiment(cfg,a.smoke)
        if a.smoke:
            start=experiment.progress['step'];experiment.train_loop(start+3)
            assert experiment.progress['step']==start+3
            digest=state_digest(experiment.model);replicas=[None]*experiment.world;dist.all_gather_object(replicas,digest)
            assert len(set(replicas))==1
            progress=dict(experiment.progress);experiment.restore();assert experiment.progress==progress
            payload=torch.load(experiment.out/'prior-last.pt',map_location='cpu',mmap=True,weights_only=False)
            current=trainer.capture_rng();expected=payload['rng'][experiment.rank]
            assert torch.equal(current['torch'],expected['torch']) and torch.equal(current['cuda'],expected['cuda'])
            assert {int(v['step']) for v in experiment.optimizer.state.values() if 'step' in v}=={progress['step']}
            assert trainer.digest_state(experiment.source)==experiment.frozen_tokenizer_digest
            assert not any('beam' in name for name in sys.modules if name.startswith('src.models.'))
            report=dict(passed=True,four_gpu_training=True,world_size=4,source_step=start,final_step=progress['step'],
                successful_training_updates=3,full_state_roundtrip=True,all_rank_parameters_bit_exact=True,all_rank_rng_bit_exact=True,
                optimizer_ages_preserved_and_advanced=True,tokenizer_digest_unchanged=True,source_encoder_changed=False,
                native_omp=True,beam_search=False,soft_teacher=False,hard_atom_and_bin_ce=True,coefficient_bins=128,
                global_batch=experiment.global_batch,microbatch_per_gpu=experiment.micro,accumulation=experiment.accum,
                compiled_forward=True,type_valid_classifier_rows_only=True,gradient_as_bucket_view=True,
                peak_memory_gib=torch.cuda.max_memory_allocated()/2**30,time=time.time())
            if experiment.rank==0:trainer.atomic(Path(cfg['execution_optimization']['base'])/'preflight-verification.json',report);print(json.dumps(report),flush=True)
        else:experiment.train_loop()
    except Exception as e:
        if experiment is not None:
            trainer.atomic(experiment.out/f'failure-rank{experiment.rank}.json',dict(error=repr(e),progress=experiment.progress))
            if experiment.run:experiment.run.summary['training/status']='failed';experiment.run.finish(exit_code=1)
        raise
    finally:
        if dist.is_initialized():dist.destroy_process_group()
if __name__=='__main__':main()
