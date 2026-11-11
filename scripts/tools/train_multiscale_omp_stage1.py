"""ImageNet stage-1 autoencoder: independent 4096-atom OMP dictionaries per scale, continuous coefficients."""
import argparse,contextlib,fcntl,hashlib,json,os,random,signal,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch import distributed as td
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,DistributedSampler,Subset
from torchvision.utils import save_image
from src.models.multiscale_laser_var import PATCH_NUMS
from src.models.scratch_var import ScratchVQVAE,state_digest
from src.models.multiscale_omp import MultiScaleOMPAutoencoder
from src.models.discriminator import NLayerDiscriminator
from src.models.rqvae.lpips import LPIPS
from src.rqvae_metrics import OriginalRQVAEInception,frechet_distance
from src.original_rq_training import FeatureMoments
from train_joint4096_var import ArchiveImages,SmokeImages,OffsetSampler,evaluation_rng,capture_rng,restore_rng,should_stop,atomic,sha
import dist as var_dist


def shared_backbone_digest(model,arm):
    h=hashlib.sha256()
    modules=[(name,getattr(model,name)) for name in ['encoder','decoder','quant_conv','post_quant_conv']]
    modules.append(('residual_convolutions',model.quantize.quant_resi))
    for prefix,module in modules:
        for name,tensor in sorted(module.state_dict().items()):
            h.update((prefix+'.'+name).encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class Experiment:
    def __init__(self,cfg,smoke=False):
        self.cfg=cfg;self.arm=cfg['arm'];self.smoke=smoke;self.rank=td.get_rank();self.world=td.get_world_size()
        self.device=torch.device('cuda',int(os.environ['LOCAL_RANK']));self.out=Path(cfg['output']);self.out.mkdir(parents=True,exist_ok=True)
        self.stop=False;self.run=None;self.inception=None
        signal.signal(signal.SIGTERM,lambda *_:setattr(self,'stop',True));signal.signal(signal.SIGINT,lambda *_:setattr(self,'stop',True))
        self.contract={k:v for k,v in cfg.items() if k not in ['output','wandb','workers','checkpoint_every','preview_every','source_bundle']}
        self.contract_sha=hashlib.sha256(json.dumps(self.contract,sort_keys=True).encode()).hexdigest()
        for name,digest in cfg.get('runtime_sha256',{}).items():
            if sha(Path(__file__).parent/name)!=digest:raise ValueError('Runtime changed: '+name)
        if self.rank==0:
            self.lock=(self.out/'training.lock').open('a');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if (self.out/'contract.json').exists() and json.loads((self.out/'contract.json').read_text())!=self.contract:raise ValueError('Training contract changed')
            atomic(self.out/'contract.json',self.contract);atomic(self.out/'config.json',cfg)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cfg['seed'])
            self.model=MultiScaleOMPAutoencoder(channels=cfg['channels'],atoms=cfg['atoms_per_scale'],sparsity=cfg['sparsity'],ch=cfg['vae_width'],patch_nums=tuple(cfg['patch_nums']))
        if list(self.model.quantize.v_patch_nums)!=cfg['patch_nums'] or sum(p*p for p in cfg['patch_nums'])*cfg['sparsity']!=680:
            raise ValueError('Configured scales do not match the actual 680-pair model')
        if self.rank==0 and not (self.out/'initialization.json').exists():
            atomic(self.out/'initialization.json',dict(initialization='fresh_random_no_checkpoint',seed=cfg['seed'],model_sha256=state_digest(self.model),shared_backbone_sha256=shared_backbone_digest(self.model,self.arm),parameters=sum(p.numel() for p in self.model.parameters()),sparse_pairs_per_image=680,atoms_per_scale=4096,coefficient_representation='continuous_float32',discrete_tokenization_stage=2,scales=cfg['patch_nums'],sparsity=cfg['sparsity']))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cfg['seed']+2001);self.discriminator=NLayerDiscriminator(norm='none',spectral=True,antialias=True)
        self.model.to(self.device);self.discriminator.to(self.device)
        self.perceptual=LPIPS().eval().requires_grad_(False).to(self.device)
        if cfg.get('execution',{}).get('lpips_channels_last',False):
            self.perceptual.to(memory_format=torch.channels_last)
        self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=cfg['lr'],betas=(.5,.9),weight_decay=0.,fused=True)
        self.d_optimizer=torch.optim.Adam(self.discriminator.parameters(),lr=1.e-4,betas=(.5,.9))
        self.wrapped=DDP(self.model,device_ids=[self.device.index],broadcast_buffers=False)
        self.d_wrapped=DDP(self.discriminator,device_ids=[self.device.index],broadcast_buffers=False)
        self.progress=dict(epoch=0,batch=0,step=0)
        torch.manual_seed(cfg['seed']+self.rank);np.random.seed(cfg['seed']+self.rank);random.seed(cfg['seed']+self.rank)
        if smoke:self.train=SmokeImages(cfg['smoke_cache']);self.val=None
        else:
            receipt=json.loads(Path(cfg['training_archive_receipt']).read_text())
            if not receipt['verified'] or receipt['md5']!='1d675b47d978889d74fa0da5fadfb00e':raise ValueError('ImageNet archive is not verified')
            self.train=ArchiveImages(cfg['train_archive'],cfg['train_index'],True,cfg['seed'])
            self.val=ArchiveImages(cfg['val_archive'],cfg['val_index'],False,cfg['seed'],labels=cfg['val_labels'])
        self.micro=cfg['microbatch'];self.accum=cfg['accumulation'];self.global_batch=self.micro*self.accum*self.world
        if self.global_batch!=cfg['global_batch']:raise ValueError('Global image batch mismatch')
        self.updates=((len(self.train)+self.world-1)//self.world//self.micro)//self.accum
        if self.updates<1:raise ValueError('Not enough images for a complete optimizer update')
        if (self.out/'stage1-last.pt').exists():self.restore()
        if self.rank==0 and not smoke:
            import wandb
            self.run=wandb.init(entity='helloimlixin-rutgers',project='laser',id=cfg['wandb']['id'],name=cfg['wandb']['name'],group='imagenet-multiscale-omp-stage1-20261103',job_type='stage1-continuous-omp',resume='allow',dir=str(self.out),config=cfg)
            self.run.config.update(cfg,allow_val_change=True)
            self.run.name=cfg['wandb']['name']
            self.run.define_metric('train/step');self.run.define_metric('train/*',step_metric='train/step');self.run.define_metric('reconstruction_full/rfid',summary='min')
            self.run.summary.update({'training/status':'running','bottleneck/sparse_pairs_per_image':680,'bottleneck/atoms_per_scale':4096,'bottleneck/independent_dictionaries':4,'bottleneck/continuous_coefficients':True,'bottleneck/discrete_tokenization_stage':2,'bottleneck/spatial_sites':sum(p*p for p in cfg['patch_nums']),'bottleneck/sparsity':cfg['sparsity'],'initialization/trainable_weights':'fresh random','training/target_epochs':cfg['epochs'],'training/global_batch':self.global_batch})
            self.run.summary.update({'training/world_size':self.world,'training/microbatch_per_gpu':self.micro,'training/accumulation':self.accum})
            if cfg.get('migration'):self.run.summary['training/topology_migration']=cfg['migration']
            a=wandb.Artifact(cfg['wandb']['id']+'-source',type='source',metadata={'contract_sha256':self.contract_sha,'sparse_pairs_per_image':680})
            a.add_file(cfg['source_bundle'],name='source.tar.gz');a.add_file(str(self.out/'config.json'));a.add_file(str(self.out/'initialization.json'))
            for key in ['train_index','val_index','val_labels','fid_reference','training_archive_receipt']:a.add_file(cfg[key],name='data/'+Path(cfg[key]).name)
            if (self.out/'migration-receipt.json').exists():a.add_file(str(self.out/'migration-receipt.json'))
            self.run.log_artifact(a).wait();atomic(self.out/'wandb.json',dict(url=self.run.url,id=self.run.id))
        self.log('setup',parameters=sum(p.numel() for p in self.model.parameters()),global_batch=self.global_batch,updates_per_epoch=self.updates,sparse_pairs_per_image=680)

    def log(self,phase,**values):
        if self.rank==0:
            record=dict(phase=phase,time=time.time(),**values);atomic(self.out/'status.json',record);print(json.dumps(record),flush=True)
            if self.run:self.run.log({phase+'/'+k:v for k,v in values.items()})

    def save(self,publish=False):
        rng=[None]*self.world;td.all_gather_object(rng,capture_rng())
        if self.rank==0:
            path=self.out/'stage1-last.pt';tmp=path.with_suffix('.tmp')
            torch.save(dict(model=self.model.state_dict(),optimizer=self.optimizer.state_dict(),discriminator=self.discriminator.state_dict(),discriminator_optimizer=self.d_optimizer.state_dict(),progress=self.progress,rng=rng,contract_sha256=self.contract_sha,world_size=self.world,record_hit=getattr(self.model.quantize,'record_hit',None)),tmp);tmp.replace(path)
            receipt=dict(progress=self.progress,checkpoint_sha256=sha(path),bytes=path.stat().st_size,full_state=True,contract_sha256=self.contract_sha,world_size=self.world)
            if self.cfg.get('migration'):receipt['migration']=self.cfg['migration']
            if self.arm=='laser':torch.save(self.model.quantize.export_dictionaries(),self.out/'dictionaries.pt')
            atomic(self.out/'checkpoint.json',receipt)
            if publish and self.run:
                import wandb
                a=wandb.Artifact(self.cfg['wandb']['id']+'-stage1',type='model',metadata={**receipt,'sparse_pairs_per_image':680,'atoms_per_scale':4096,'continuous_coefficients':True,'independent_dictionaries':4,'sparsity':self.cfg['sparsity'],'patch_nums':self.cfg['patch_nums']})
                a.add_file(str(path),name='checkpoint.pt',policy='immutable');a.add_file(str(self.out/'config.json'))
                if self.arm=='laser':a.add_file(str(self.out/'dictionaries.pt'))
                aliases=['latest',f"step-{self.progress['step']:09d}"]
                if self.progress['batch']==0:aliases.append(f"epoch-{self.progress['epoch']:03d}")
                logged=self.run.log_artifact(a,aliases=aliases);logged.wait();receipt['artifact']=logged.qualified_name;atomic(self.out/'checkpoint.json',receipt);self.run.summary['checkpoint/latest_step']=self.progress['step']
        td.barrier()

    def restore(self):
        p=torch.load(self.out/'stage1-last.pt',map_location='cpu',weights_only=False,mmap=True)
        if p['contract_sha256']!=self.contract_sha or p['world_size']!=self.world:raise ValueError('Incompatible stage1 checkpoint')
        self.model.load_state_dict(p['model']);self.optimizer.load_state_dict(p['optimizer']);self.discriminator.load_state_dict(p['discriminator']);self.d_optimizer.load_state_dict(p['discriminator_optimizer']);self.progress=dict(p['progress'])
        if p.get('record_hit') is not None:self.model.quantize.record_hit=p['record_hit']
        restore_rng(p['rng'][self.rank]);self.log('resume',step=self.progress['step'],epoch=self.progress['epoch'],all_rank_rng_restored=True)

    @torch.no_grad()
    def preview(self):
        with evaluation_rng(self.device):
            self.model.eval()
            if self.rank==0:
                dataset=SmokeImages(self.cfg['smoke_cache'])
                images=dataset.images[:8].to(self.device)
                with torch.autocast('cuda',dtype=torch.bfloat16):recon=self.model(images)[0]
                if not recon.isfinite().all():raise FloatingPointError('Nonfinite reconstruction')
                path=self.out/f"recon-step{self.progress['step']:09d}.png";save_image(torch.stack(((images+1)*.5,(recon.float().clamp(-1,1)+1)*.5),1).flatten(0,1),path,nrow=8)
                if self.run:
                    import wandb
                    self.run.log({'reconstruction/preview':wandb.Image(str(path),caption=f"{self.arm}; alternating original/reconstruction; step {self.progress['step']}")})
                atomic(self.out/'last-preview.json',dict(step=self.progress['step'],path=str(path),finite=True))
            td.barrier();self.model.train()

    @torch.no_grad()
    def evaluate(self):
        with evaluation_rng(self.device):
            self.model.eval()
            if self.inception is None:self.inception=OriginalRQVAEInception().eval().requires_grad_(False).to(self.device)
            moments=FeatureMoments(self.device);totals=torch.zeros(3,device=self.device,dtype=torch.float64)
            loader=DataLoader(Subset(self.val,list(range(self.rank,50000,self.world))),batch_size=self.cfg['eval_batch'],num_workers=self.cfg['workers'],pin_memory=True,multiprocessing_context='spawn',generator=torch.Generator().manual_seed(self.rank))
            for images,_ in loader:
                images=images.to(self.device,non_blocking=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):recon=self.model(images)[0]
                decoded=(recon.float().clamp(-1,1)+1)*.5;real=(images+1)*.5
                moments.update(self.inception(decoded)[0]);mse=(decoded-real).square().mean((1,2,3));totals+=torch.stack((mse.sum(),(-10*mse.clamp_min(1e-12).log10()).sum(),mse.new_tensor(len(images)))).double()
            n,mu,cov=moments.finish();td.all_reduce(totals)
            if n!=50000 or int(totals[2])!=50000:raise ValueError('Incomplete 50k reconstruction evaluation')
            if self.rank==0:
                ref=np.load(self.cfg['fid_reference']);score=frechet_distance(ref['mu'],ref['sigma'],mu,cov)
                result=dict(epoch=self.progress['epoch'],step=self.progress['step'],rfid=score,psnr=float(totals[1]/n),mse=float(totals[0]/n),count=n,reference_count=50000,sparse_pairs_per_image=680,atoms_per_scale=4096)
                atomic(self.out/f"reconstruction-epoch{self.progress['epoch']:03d}.json",result);self.log('reconstruction_full',**result)
                best=self.out/'best-reconstruction.json'
                if not best.exists() or score<json.loads(best.read_text())['rfid']:
                    atomic(best,{**result,'checkpoint':json.loads((self.out/'checkpoint.json').read_text())})
                    if self.run:self.run.summary.update({'reconstruction/best_rfid':score,'reconstruction/best_epoch':self.progress['epoch']})
            td.barrier();self.model.train()

    def train_loop(self,max_steps=None):
        # An epoch checkpoint is committed before evaluation. Finish a pending
        # full validation after a restart, including after the final epoch.
        if not self.smoke and self.progress['epoch']>0 and self.progress['batch']==0 and not (self.out/f"reconstruction-epoch{self.progress['epoch']:03d}.json").exists():self.evaluate()
        if max_steps is not None and self.progress['step']>=max_steps:return
        cfg=self.cfg
        for epoch in range(self.progress['epoch'],cfg['epochs']):
            self.train.epoch=epoch;sampler=DistributedSampler(self.train,self.world,self.rank,shuffle=True,seed=cfg['seed']);sampler.set_epoch(epoch)
            begin=self.progress['batch'];loader=DataLoader(self.train,batch_size=self.micro,sampler=OffsetSampler(sampler,begin*self.micro),num_workers=cfg['workers'] if not self.smoke else 0,pin_memory=True,drop_last=True,multiprocessing_context='spawn' if cfg['workers'] and not self.smoke else None,generator=torch.Generator().manual_seed(cfg['seed']+epoch+self.rank))
            self.model.train();self.optimizer.zero_grad(set_to_none=True);self.d_optimizer.zero_grad(set_to_none=True)
            metrics=torch.zeros(8,device=self.device);counts=torch.zeros(4,4096,device=self.device,dtype=torch.int64);tick=time.monotonic()
            for batch,(images,labels) in enumerate(loader,start=begin):
                if batch>=self.updates*self.accum:break
                if self.rank==0 and self.progress['step']==0 and batch==0:
                    atomic(self.out/'first-batch.json',dict(pixels_sha256=hashlib.sha256(images.numpy().tobytes()).hexdigest(),labels_sha256=hashlib.sha256(labels.numpy().tobytes()).hexdigest()))
                images=images.to(self.device,non_blocking=True);sync=(batch+1)%self.accum==0
                ratio=min(1.,(self.progress['step']+1)/cfg['warmup_updates'])
                for group in self.optimizer.param_groups:group['lr']=cfg['lr']*ratio
                adversarial=self.progress['step']>=cfg['adversarial_start']
                start=cfg['adversarial_start']+cfg['critic_warmup_steps']
                ramp=min(1.,(self.progress['step']-start+1)/max(1,cfg['gan_ramp_steps'])) if self.progress['step']>=start else 0.
                for p in self.discriminator.parameters():p.requires_grad_(False)
                with (contextlib.nullcontext() if sync else self.wrapped.no_sync()),torch.autocast('cuda',dtype=torch.bfloat16):
                    recon,_,book=self.wrapped(images)
                    with torch.autocast('cuda',enabled=False):
                        pixels=recon.float();l1=F.l1_loss(pixels,images);lpips=self.perceptual(pixels,images);reconstruction=l1+lpips
                        gan=pixels.new_zeros(());weight=pixels.new_zeros(())
                        if adversarial and ramp>0:
                            gan=-self.discriminator(pixels).mean();last=self.model.decoder.conv_out.weight
                            rg=torch.autograd.grad(reconstruction,last,retain_graph=True)[0];gg=torch.autograd.grad(gan,last,retain_graph=True)[0]
                            weight=((rg.norm()/(gg.norm()+1.e-4))*.5).clamp(max=.1).detach()*ramp
                    loss=reconstruction+book+weight*gan;(loss/self.accum).backward()
                ids=self.model.quantize.last_atom_ids if self.arm=='laser' else self.model.last_atom_ids
                if ids.numel()!=len(images)*680:raise ValueError('Tokenizer violated 680-pair budget')
                offset=0
                for scale,pn in enumerate(cfg['patch_nums']):
                    counts[scale]+=torch.bincount(ids[:,offset:offset+pn*pn].flatten(),minlength=4096)
                    offset+=pn*pn
                d_loss=loss.new_zeros(())
                if adversarial:
                    for p in self.discriminator.parameters():p.requires_grad_(True)
                    with (contextlib.nullcontext() if sync else self.d_wrapped.no_sync()),torch.autocast('cuda',enabled=False):
                        logits=self.d_wrapped(torch.cat((images,pixels.detach())));real_logits,fake_logits=logits.chunk(2)
                        d_loss=(F.relu(1-real_logits).mean()+F.relu(1+fake_logits).mean())*.5;(d_loss/self.accum).backward()
                    if sync:
                        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(),1.,error_if_nonfinite=True);self.d_optimizer.step();self.d_optimizer.zero_grad(set_to_none=True)
                metrics+=torch.stack((loss.detach(),l1.detach(),lpips.detach(),book.detach(),d_loss.detach(),gan.detach(),weight.detach(),(weight*gan).detach()))/self.accum
                if not sync:continue
                norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.,error_if_nonfinite=True);self.optimizer.step();self.optimizer.zero_grad(set_to_none=True)
                self.progress=dict(epoch=epoch,batch=batch+1,step=self.progress['step']+1)
                td.all_reduce(metrics);metrics/=self.world;td.all_reduce(counts)
                if int(counts.sum())!=self.global_batch*680:raise ValueError('Global sparse-support counts differ from sparse-pair budget')
                if self.progress['step']<=10 or self.progress['step']%10==0:
                    probability=counts.double()/counts.sum(1,keepdim=True);entropy=-(probability*probability.clamp_min(1.e-30).log()).sum(1)
                    self.log('train',step=self.progress['step'],epoch=epoch,epoch_progress=epoch+(batch+1)/(self.updates*self.accum),loss=float(metrics[0]),l1=float(metrics[1]),lpips=float(metrics[2]),bottleneck_loss=float(metrics[3]),discriminator_loss=float(metrics[4]),generator_adversarial=float(metrics[5]),gan_weight=float(metrics[6]),weighted_gan_loss=float(metrics[7]),gradient_norm=float(norm),lr=self.optimizer.param_groups[0]['lr'],mean_scale_atom_perplexity=float(entropy.exp().mean()),used_atoms_across_scales=int((counts>0).sum()),sparse_pairs_per_image=680,images_seen=self.progress['step']*self.global_batch,images_per_second=self.global_batch/(time.monotonic()-tick),peak_memory_gib=torch.cuda.max_memory_allocated()/2**30)
                metrics.zero_();counts.zero_();tick=time.monotonic()
                flag=torch.tensor(int(should_stop(self.stop,(self.out/'STOP').exists(),max_steps,self.progress['step'])),device=self.device);td.all_reduce(flag,op=td.ReduceOp.MAX)
                if self.progress['step'] in [1,10] or self.progress['step']%cfg['checkpoint_every']==0 or flag.item():self.save(publish=not self.smoke and (self.progress['step']==10 or self.progress['step']%cfg['checkpoint_every']==0 or bool(flag.item())))
                if self.progress['step']==1 or self.progress['step']%cfg['preview_every']==0:self.preview()
                if flag.item():return
            self.progress.update(epoch=epoch+1,batch=0);self.save(publish=not self.smoke)
            if not self.smoke:self.evaluate()
        if self.run:self.run.summary['training/status']='completed';self.run.finish()


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--max-steps',type=int);a=p.parse_args();cfg=json.loads(a.config.read_text())
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=cfg.get('execution',{}).get('cudnn_benchmark',False);var_dist.initialize(backend='nccl',timeout=120);experiment=None
    try:
        experiment=Experiment(cfg,a.smoke)
        if not a.smoke and cfg.get('migration'):experiment.save(publish=True)
        experiment.train_loop(a.max_steps)
        if a.smoke:
            previous=dict(experiment.progress);experiment.restore();saved=torch.load(experiment.out/'stage1-last.pt',map_location='cpu',weights_only=False,mmap=True)
            current=capture_rng();expected=saved['rng'][experiment.rank]
            if not torch.equal(current['torch'],expected['torch']) or not torch.equal(current['cuda'],expected['cuda']):raise ValueError('RNG restore differs')
            if not torch.equal(experiment.model.decoder.conv_out.weight.detach().cpu(),saved['model']['decoder.conv_out.weight']):raise ValueError('Model restore differs')
            ages={int(v['step']) for v in experiment.optimizer.state.values() if 'step' in v}
            if ages!={previous['step']}:raise ValueError('Optimizer age differs')
            if experiment.rank==0:atomic(experiment.out/'smoke-complete.json',dict(completed=True,progress=previous,full_state_roundtrip=True,model_bit_exact=True,rng_bit_exact=True,optimizer_age=list(ages),sparse_pairs_per_image=680,atoms_per_scale=4096,gan_exercised=cfg['adversarial_start']==0))
        elif experiment.run and experiment.progress['epoch']<cfg['epochs']:experiment.run.summary['training/status']='paused_after_checkpoint';experiment.run.finish()
    except Exception as error:
        if experiment is not None:
            atomic(experiment.out/f'failure-rank{experiment.rank}.json',dict(type=type(error).__name__,error=str(error),progress=experiment.progress))
            if experiment.run:experiment.run.summary['training/status']='failed';experiment.run.finish(exit_code=1)
        raise
    finally:
        if td.is_initialized():td.destroy_process_group()

if __name__=='__main__':main()
