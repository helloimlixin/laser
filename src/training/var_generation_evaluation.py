"""Paired FID and Inception Score from one class-balanced VAR sample set."""
from contextlib import contextmanager
import random
import time
import numpy as np
import torch
import torch.distributed as dist
from src.rqvae_metrics import OriginalRQVAEInception, DistributedOriginalRQVAEMetrics, load_reference_statistics, frechet_distance


from src.original_rq_training import FeatureMoments
from src.training.var_generation_cpu_metrics import numpy_inception_score


class PairedGenerationMetrics:
    feature_dim=2048
    num_logits=1008
    def __init__(self,device,*,compute_inception_score,statistics_directory,reference_stats_path,inception):
        from pathlib import Path
        self.device=device;self.inception=inception;self.reference_stats_path=Path(reference_stats_path)
        self.statistics_directory=Path(statistics_directory);self.moments=FeatureMoments(device)
        if self.feature_dim!=2048:
            self.moments.total=torch.zeros(self.feature_dim,dtype=torch.float64,device=device)
            self.moments.cross=torch.zeros(self.feature_dim,self.feature_dim,dtype=torch.float64,device=device)
        self.fake_probabilities=[];self.global_count=None
        assert compute_inception_score
    @property
    def fake_count(self):return self.moments.count if self.global_count is None else self.global_count
    @property
    def fake_sum(self):return self.moments.total
    @property
    def fake_cross(self):return self.moments.cross
    @torch.no_grad()
    def update(self,images,*,real):
        assert not real
        features,logits=self.inception(images.float(),return_logits=True)
        self.moments.update(features);self.fake_probabilities.append(logits.softmax(1).cpu())
    @torch.no_grad()
    def compute(self,*,shuffle_for_inception=True,shuffle_seed=0):
        import json,os,subprocess,sys
        from pathlib import Path
        rank=dist.get_rank() if dist.is_initialized() else 0;world=dist.get_world_size() if dist.is_initialized() else 1
        folder=self.statistics_directory;folder.mkdir(parents=True,exist_ok=True);result_path=folder/'scores.json'
        if rank==0:result_path.unlink(missing_ok=True)
        np.save(folder/f'probabilities-rank{rank}.npy',np.concatenate([p.numpy() for p in self.fake_probabilities],0))
        if dist.is_initialized():count,mean,covariance=self.moments.finish()
        else:
            count=int(self.moments.count);mean=self.moments.total/count
            covariance=(self.moments.cross-count*torch.outer(mean,mean))/(count-1)
            mean,covariance=mean.cpu().numpy(),covariance.cpu().numpy()
        self.global_count=count
        if rank==0:
            np.savez(folder/'fid-moments.npz',mu=mean,sigma=covariance)
            cpu=Path(__file__).with_name('var_generation_cpu_metrics.py')
            command=[sys.executable,str(cpu),'--reference',str(self.reference_stats_path),'--moments',str(folder/'fid-moments.npz'),'--directory',str(folder),'--world',str(world),'--count',str(count),'--seed',str(shuffle_seed)]
            subprocess.run(command,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'),check=True,timeout=300)
        else:
            deadline=time.monotonic()+600
            while not result_path.exists():
                if time.monotonic()>deadline:raise TimeoutError('CPU metric aggregation did not finish')
                time.sleep(.1)
        row=json.loads(result_path.read_text());return row['fid'],row['inception_score'],row['inception_score_std']


@contextmanager
def evaluation_rng(device):
    python_state,numpy_state=random.getstate(),np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[device.index] if device.type=='cuda' else []):yield
    finally:
        random.setstate(python_state);np.random.set_state(numpy_state)


def configure_generation_logging(run):
    run.define_metric('generation/epoch')
    run.define_metric('generation/*',step_metric='generation/epoch')
    run.summary['evaluation/metrics']='FID50k and Inception Score mean/std; 10 shuffled splits of the same 50000 images'


@torch.no_grad()
def evaluate_generation(experiment):
    self=experiment
    total=int(self.cfg.get('fid_samples',50000));classes=int(self.model.num_classes)
    batch=int(self.cfg.get('generation_batch_size',32))
    if total % classes or total % self.world:raise ValueError('Evaluation must be complete and class balanced')
    was_training=self.model.training
    with evaluation_rng(self.device):
        self.model.eval()
        try:
            if self.inception is None:
                self.inception=OriginalRQVAEInception().eval().requires_grad_(False).to(self.device)
            metric=PairedGenerationMetrics(self.device,compute_inception_score=True,
                statistics_directory=self.out/'evaluation-statistics'/f"epoch{self.progress['epoch']:03d}-step{self.progress['step']:09d}",
                reference_stats_path=self.cfg['fid_reference'],inception=self.inception)
            counts=torch.zeros(classes,device=self.device,dtype=torch.int64)
            positions=list(range(self.rank,total,self.world));started=time.monotonic()
            for first in range(0,len(positions),batch):
                selected=positions[first:first+batch]
                labels=torch.tensor([i%classes for i in selected],device=self.device)
                # Preserve the existing generation seed, batching, and decoder arithmetic.
                with torch.autocast(device_type=self.device.type,dtype=torch.bfloat16,enabled=self.device.type=='cuda'):
                    latent=self.model.sample(labels,cfg=1.5,top_k=900,top_p=.96,
                        seed=1000003*self.progress['epoch']+selected[0])
                    images=torch.cat([self.source.decoder(self.source.post_quant_conv(part)).float().clamp(-1,1)
                                      for part in latent.split(8)])
                metric.update((images+1)*.5,real=False)
                counts+=torch.bincount(labels,minlength=classes)
            if int(metric.fake_count)!=total//self.world:raise ValueError('Incomplete local generated sample set')
            if dist.is_initialized():dist.all_reduce(counts)
            if not (counts==total//classes).all():raise ValueError('Generation labels are not class balanced')
            shuffle_seed=int(self.cfg.get('evaluation_seed',26110601))+self.progress['epoch']
            if self.device.type=='cuda':torch.cuda.manual_seed(shuffle_seed)
            else:torch.random.default_generator.manual_seed(shuffle_seed)
            fid,mean,std=metric.compute(shuffle_for_inception=True,shuffle_seed=shuffle_seed)
            if int(metric.fake_count)!=total:raise ValueError('Incomplete global generated sample set')
            record=dict(step=self.progress['step'],epoch=self.progress['epoch'],validation_fid50k=fid,
                inception_score=mean,inception_score_std=std,inception_score_splits=10,
                inception_score_shuffle_seed=shuffle_seed,inception_score_shuffle_algorithm='numpy_PCG64',count=total,reference_count=50000,
                reference_split='validation',cfg=1.5,top_k=900,top_p=.96,
                elapsed_seconds=time.monotonic()-started)
            if self.rank==0:
                import json
                path=self.out/f"fid-epoch{self.progress['epoch']:03d}.json"
                temporary=path.with_suffix('.pending');temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(path)
                self.log('generation',**record)
            if dist.is_initialized():dist.barrier()
            return record
        finally:self.model.train(was_training)
