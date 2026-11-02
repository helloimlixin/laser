#!/usr/bin/env python3
"""Hold old epoch-one/two weights fixed; isolate K2048 and target-noise effects."""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/workspace/Projects/laser')
BASE = Path('/tmp/laser-rq-joint20k-depth5-imagenet-20261017')
OLD = ROOT/'outputs/imagenet-laser-joint20k-depth5-scratch100-4h100-20261017'
NEW = ROOT/'outputs/imagenet-laser-joint20k-depth5-rqnoise-k2048-scratch100-4h100-20261017'
OUT = ROOT/'outputs/imagenet-laser-joint20k-depth5-epochs1-2-k2048-control-20261017'
os.environ['RQ_UPSTREAM_ROOT'] = str(BASE/'upstream')
os.environ['TORCH_HOME'] = str(BASE/'torch-cache')
sys.path[:0] = [str(OUT/'source'),str(BASE/'source'),str(BASE/'upstream'),str(BASE),str(BASE/'runtime')]

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from PIL import Image
from joint_integer_sampling_audit import retained_probability
from src.training.laser_joint_recipe import JointLaserRecipe
from src.training.rq_imagenet_runtime import write_json
from src.training.rq_sample_panels import designated_preview,MEDIA_KEYS
from src.training import rq_imagenet_shared as shared


def main():
    from datetime import timedelta
    rank,world,local = [int(os.environ[k]) for k in ('RANK','WORLD_SIZE','LOCAL_RANK')]
    assert world == 4
    torch.set_num_threads(4)
    device = torch.device('cuda',local)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(16*2**30/torch.cuda.get_device_properties(device).total_memory,device)
    dist.init_process_group('nccl',device_id=device,timeout=timedelta(minutes=60))
    plan = json.loads((OUT/'plan.json').read_text())
    run = None
    if rank == 0:
        import wandb
        os.environ['WANDB_API_KEY'] = Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip()
        run = wandb.init(entity='helloimlixin-rutgers',project='laser',id=OUT.name,
            name='LASER depth5 epochs1–2 | isolate K2048 and target noise',resume='allow',
            dir=os.environ['WANDB_DIR'],config=plan,job_type='fixed-checkpoint-causal-comparison')
        run.define_metric('comparison/epoch')
        run.define_metric('comparison/*',step_metric='comparison/epoch')
        run.define_metric('samples/global_step')
        run.define_metric('samples/*',step_metric='samples/global_step')
        write_json(OUT/'wandb.json',dict(id=run.id,url=run.url))
    dist.barrier()
    tokenizer_profile = JointLaserRecipe(BASE)
    aux,_ = tokenizer_profile.tokenizer(device)
    frozen = tokenizer_profile.digest(aux)
    shared.BASE = BASE
    shared.original_metrics.INCEPTION_PATH = BASE/'upstream/rqvae/metrics/inception.py'
    results = []
    for item in plan['checkpoints']:
        epoch,step = item['epoch'],item['global_step']
        destination = OUT/f'epoch{epoch:03d}'
        if (destination/'fid50k.json').exists():
            cached = json.loads((destination/'fid50k.json').read_text())
            assert cached['epoch'] == epoch and cached['global_step'] == step
            assert cached['checkpoint_sha256'] == item['receipt']['sha256']
            assert cached['sampling'] == plan['sampling']
            assert cached['generated_images'] == 50000 and cached['real_images'] == 1281167
            assert cached['seed'] == 261001
            assert json.loads((destination/'checkpoint-verification.json').read_text())['baseline_preview_pixels_exact']
            results.append(cached)
            dist.barrier()
            continue
        profile = JointLaserRecipe(BASE)
        saved = torch.load(item['local_checkpoint'],map_location='cpu',mmap=True,weights_only=False)
        profile.verify_resume(saved)
        assert saved['global_step'] == epoch*626 == step and saved['next_epoch'] == epoch
        assert saved['next_microbatch'] == 0 and len(saved['rng_state_by_rank']) == 4
        assert all(int(state['step']) == step for state in saved['optimizer']['state'].values())
        with torch.device('meta'):
            model = profile.model(OmegaConf.create(saved['config']['arch']))
        model.load_state_dict(saved['state_dict'],strict=True,assign=True)
        del saved
        model.requires_grad_(False).eval().to(device)
        profile.adapt(model,aux)
        assert tuple(model.block_size) == (8,8,5) and model.vocab_size == [20481]*5
        versions = [(name,p._version) for name,p in model.named_parameters()]
        if rank == 0:
            write_json(OUT/'status.json',dict(phase='baseline_pixel_parity',epoch=epoch,time=time.time()))
        with torch.inference_mode():
            designated_preview(profile,model,aux,device,rank,world,step,None,destination/'baseline-parity')
        if rank == 0:
            for key in MEDIA_KEYS:
                name = key.split('/')[-1]
                expected = np.asarray(Image.open(OLD/f'samples/step{step:06d}-{name}.png'))
                actual = np.asarray(Image.open(destination/f'baseline-parity/smoke/samples/step{step:06d}-{name}.png'))
                assert np.array_equal(expected,actual),key
            write_json(destination/'checkpoint-verification.json',dict(passed=True,epoch=epoch,step=step,
                strict_model_reload=True,optimizer_age=step,baseline_preview_pixels_exact=True,
                panels_verified=9,preview_images=512,checkpoint_sha256=item['receipt']['sha256']))
        dist.barrier()
        # Only inference settings change. Checkpoint targets/optimizer stay intact.
        profile.recipe.sampling = OmegaConf.create(plan['sampling'])
        shared.PROFILE,shared.RUN,shared.OUT = profile,profile.run_id,destination/'k2048'
        retained = torch.zeros(5,8,dtype=torch.float64,device=device)
        original_cached = model.cached_forward

        def audited_cached(*args,**kwargs):
            logits = original_cached(*args,**kwargs)
            depth = kwargs['sample_loc'][2]
            # First 1,024 FID images, covering every class. Subsequent batches
            # use the exact original cached_forward without audit overhead.
            p = logits.float().softmax(-1)
            retained[depth,0] += (-(p*p.clamp_min(1.e-30).log()).sum(-1)).sum()
            for index,top_k in enumerate((1024,2048,None)):
                mass,support = retained_probability(logits,top_k=top_k,top_p=.95)
                retained[depth,1+2*index] += mass.sum()
                retained[depth,2+2*index] += support.sum()
            retained[depth,7] += len(logits)
            if depth == 4 and int(retained[depth,7]) >= 16384:
                model.cached_forward = original_cached
            return logits

        model.cached_forward = audited_cached
        if rank == 0:
            write_json(OUT/'status.json',dict(phase='fid50k',epoch=epoch,time=time.time()))
        started = time.monotonic()
        fid = shared.generation(model,aux,device,rank,world,epoch,None,batch=128,seed=261001)
        model.cached_forward = original_cached
        dist.all_reduce(retained)
        if rank == 0:
            rows = []
            for depth in range(5):
                row = retained[depth].cpu().tolist()
                assert int(row[7]) == 65536
                rows.append(dict(depth=depth+1,logit_rows=int(row[7]),predictive_entropy=row[0]/row[7],
                    filters={str(k):dict(mean_original_mass=row[1+2*i]/row[7],mean_support_size=row[2+2*i]/row[7])
                             for i,k in enumerate((1024,2048,'unrestricted'))}))
            write_json(destination/'retained-probability-audit.json',dict(passed=True,generated_images=1024,
                prefix_sampling=plan['sampling'],rows=rows))
        with torch.inference_mode():
            designated_preview(profile,model,aux,device,rank,world,step,None,destination/'k2048')
        assert versions == [(name,p._version) for name,p in model.named_parameters()]
        assert profile.digest(aux) == frozen
        old_fid = item['old_evaluation']['eval/fid']
        new_fid = item['new_evaluation']['eval/fid']
        row = dict(epoch=epoch,global_step=step,old_noise_k1024_fid=old_fid,
            old_noise_k2048_fid=fid['fid'],new_noise_k2048_fid=new_fid,
            sampler_gain_at_old_weights=old_fid-fid['fid'],
            noise_recipe_gain_at_fixed_k2048=fid['fid']-new_fid,total_gain=old_fid-new_fid,
            original_rq_fid=item['original_rq_evaluation']['eval/fid'],
            old_noise_k2048_inception_score=fid['inception_score'],
            generated_images=50000,real_images=1281167,real_split='train',seed=261001,
            sampling=plan['sampling'],checkpoint_sha256=item['receipt']['sha256'],
            weights_unmodified=True,training_updates_added=0,rejection_sampling=False,
            elapsed_seconds=time.monotonic()-started)
        assert abs(row['total_gain']-row['sampler_gain_at_old_weights']-row['noise_recipe_gain_at_fixed_k2048']) < 1.e-9
        results.append(row)
        if rank == 0:
            import wandb
            write_json(destination/'fid50k.json',row)
            run.log({'comparison/epoch':epoch,'comparison/fid_old_noise_k1024':old_fid,
                'comparison/fid_old_noise_k2048':fid['fid'],'comparison/fid_new_noise_k2048':new_fid,
                'comparison/sampler_gain':row['sampler_gain_at_old_weights'],
                'comparison/noise_recipe_gain':row['noise_recipe_gain_at_fixed_k2048'],
                'comparison/fid_original_rq':row['original_rq_fid']})
            old_paths = {key:destination/f'k2048/smoke/samples/step{step:06d}-{key.split("/")[-1]}.png'
                         for key in MEDIA_KEYS}
            new_overview = NEW/f'samples/step{step:06d}-fixed.png'
            run.log({'samples/global_step':step,
                **{key:wandb.Image(str(path),caption=f'Old training noise; K2048; epoch {epoch}') for key,path in old_paths.items()},
                'comparison/new_noise_fixed':wandb.Image(str(new_overview),caption=f'New training noise; K2048; epoch {epoch}')})
            write_json(OUT/'partial-results.json',dict(results=results))
            print(json.dumps(dict(event='epoch_complete',**row)),flush=True)
            if not (OUT/'workspace.json').exists():
                import wandb_workspaces.workspaces as ws
                import wandb_workspaces.reports.v2 as wr
                workspace = ws.Workspace(entity='helloimlixin-rutgers',project='laser',
                    name='LASER depth5 early sampler and noise comparison',
                    runset_settings=ws.RunsetSettings(filters=[ws.Config('wandb_id') == OUT.name]),
                    sections=[ws.Section(name='Matched checkpoints and sampler',is_open=True,panels=[
                        wr.MarkdownPanel(markdown='All FIDs use 50,000 generated images against all 1,281,167 training images. '
                            'Old checkpoints are immutable; the active training run is unchanged. Positive gain means lower FID.'),
                        wr.LinePlot(title='FID50k at matched optimizer updates',x='comparison/epoch',
                            y=['comparison/fid_old_noise_k1024','comparison/fid_old_noise_k2048',
                               'comparison/fid_new_noise_k2048','comparison/fid_original_rq']),
                        wr.MediaBrowser(title='Old noise, K2048 — fixed classes',media_keys=['samples/fixed'],num_columns=1),
                        wr.MediaBrowser(title='New noise, K2048 — fixed classes',media_keys=['comparison/new_noise_fixed'],num_columns=1),
                    ])])
                workspace.save()
                write_json(OUT/'workspace.json',dict(url=workspace.url))
        dist.barrier()
        # An instance-level bound cached_forward retains its model. Restore
        # class method lookup, then release audit closures before the next load.
        del model.cached_forward
        del model,original_cached,audited_cached
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    if rank == 0:
        gain_shrink = dict(total=results[0]['total_gain']-results[1]['total_gain'],
            sampler_component=results[0]['sampler_gain_at_old_weights']-results[1]['sampler_gain_at_old_weights'],
            noise_recipe_component=results[0]['noise_recipe_gain_at_fixed_k2048']-results[1]['noise_recipe_gain_at_fixed_k2048'])
        assert abs(gain_shrink['total']-gain_shrink['sampler_component']-gain_shrink['noise_recipe_component']) < 1.e-9
        report = dict(completed=True,plan=plan,results=results,gain_shrink=gain_shrink,
            tokenizer_unmodified=True,active_training_settings_changed=False,
            limits='One paired training seed and one evaluation seed; this isolates the observed noise-recipe difference at K2048, not a multi-seed expected improvement. Full two-by-two interaction requires also sampling new weights at K1024.')
        write_json(OUT/'report.json',report)
        run.summary.update({'comparison/completed':True,'comparison/gain_shrink':gain_shrink,
            'comparison/noise_gain_epoch1':results[0]['noise_recipe_gain_at_fixed_k2048'],
            'comparison/noise_gain_epoch2':results[1]['noise_recipe_gain_at_fixed_k2048']})
        run.log({'comparison/results':wandb.Table(columns=list(results[0]),data=[[r[k] for k in results[0]] for r in results])})
        run.save(str(OUT/'report.json'),base_path=str(OUT),policy='now')
        run.finish()
        write_json(OUT/'status.json',dict(phase='completed',results=results,time=time.time()))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':main()
