"""Prepare an isolated full-state compound-history continuation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    args=p.parse_args()
    args.base=args.base.resolve();args.output=args.output.resolve()
    repo=Path(__file__).resolve().parents[2]
    sys.path[:0]=[str(repo),str(args.parent/'support')]
    import torch
    import yaml
    from src.training.rqtransformer import build_model
    from src.models.physical_compound_prior import PhysicalCompoundRQTransformer
    from src.training.physical_compound_resume import migrate_checkpoint,validate_optimizer_ages
    from src.training.full_resume_upload import recovery_metadata
    from src.training.k4_checkpoint_io import atomic_torch_save
    torch.set_num_threads(8)
    args.base.mkdir(parents=True,exist_ok=True);args.output.mkdir(parents=True,exist_ok=True)
    runtime=args.base/'source/runtime'
    shutil.copytree(args.parent/'source/runtime',runtime,dirs_exist_ok=True)
    shutil.copytree(args.parent/'support',args.base/'support',dirs_exist_ok=True)
    for relative in ('src/models/physical_compound_prior.py','src/training/physical_compound_resume.py',
                     'src/training/full_resume_upload.py'):
        shutil.copyfile(repo/relative,runtime/relative)
    for name in ('inputs','torch-cache'):
        (args.base/name).symlink_to(args.parent/name,target_is_directory=True)
    for name in ('wandb','wandb-cache','wandb-data','wandb-config','production'):
        (args.base/name).mkdir(exist_ok=True)
    payload=torch.load(args.source,map_location='cpu',mmap=True,weights_only=False)
    assert (payload['epoch'],payload['global_step'])==(77,48202)
    assert payload['optimizer']['param_groups'][0]['lr']==1e-6
    c=payload['config']
    with torch.device('meta'):
        model=build_model(c['num_atoms']+c['coeff_vocab_size'],c['num_atoms'],
            physical_pair_context=True,sparsity_level=4,coeff_vocab_size=c['coeff_vocab_size'],
            model_preset=c['model_preset'])
    model.load_state_dict(payload['state_dict'],strict=True,assign=True)
    names=list(dict(model.named_parameters()))
    PhysicalCompoundRQTransformer.from_scalar(model)
    migrated=migrate_checkpoint(payload,model,names,accumulation=4)
    assert migrated['scheduler'] is payload['scheduler']
    assert migrated['rng_state_by_rank'] is payload['rng_state_by_rank']
    metadata=recovery_metadata(migrated)
    os.environ.update(LASER_CHECKPOINT_STAGING_DIR=str(args.base/'checkpoint-staging'),
        LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(args.base/'checkpoint-upload-cache'),
        LASER_CHECKPOINT_IMMUTABLE_FILES='1')
    checkpoints=args.output/'train/checkpoints';checkpoints.mkdir(parents=True,exist_ok=True)
    anchor=checkpoints/'resume-epoch077-compound-anchor.pt'
    atomic_torch_save(migrated,anchor)
    (checkpoints/'last.pt').symlink_to(anchor.name)
    (args.base/'compound-transfer.json').write_text(json.dumps(migrated['compound_transfer'],indent=2)+'\n')
    (args.output/'prepared-checkpoint.json').write_text(json.dumps(metadata,indent=2,default=str)+'\n')
    run_id='imagenet-rfid421-epoch77-full-compound-history-lr1e6-8h100-20261006'
    recipe=yaml.safe_load((args.parent/'recipe.yaml').read_text())
    options=recipe['options']
    options.update(output=str(args.output/'train'),checkpoint_dir=str(checkpoints),batch_size=64,
        checkpoint=str(args.base/'inputs/resume-stage1-tokenizer.pt'),max_optimizer_steps=20,
        wandb_id=run_id,wandb_name='ImageNet K4 | full compound history | epoch77 | LR1e-6 to0 | 8H100',
        fid_batch_size=64,sample_grid_batch_size=8)
    options.pop('resume_checkpoint',None)
    (args.base/'recipe.yaml').write_text(yaml.safe_dump(recipe,sort_keys=False))
    entry=(args.parent/'entry.py').read_text()
    def replace(old,new):
        nonlocal entry
        assert entry.count(old)==1,(old,entry.count(old))
        entry=entry.replace(old,new)
    replace(" return result\ntraining.LaserAux.__init__=aux_init",
        " global PROBE_AUX\n"
        " PROBE_AUX=self\n"
        " return result\ntraining.LaserAux.__init__=aux_init")
    replace("BEST_FID=[]\nBEST_IS=[]",
        "COMPOUND_TRANSFER=json.loads((BASE/'compound-transfer.json').read_text())\n"
        "from src.training.physical_compound_resume import verify_live_optimizer, ARCHITECTURE\n"
        "COMPOUND_MODEL=None\n"
        "original_compound_build=training.build_model\n"
        "def compound_build(*args,**kwargs):\n"
        " global COMPOUND_MODEL\n"
        " from src.models.physical_compound_prior import PhysicalCompoundRQTransformer\n"
        " COMPOUND_MODEL=PhysicalCompoundRQTransformer.from_scalar(original_compound_build(*args,**kwargs))\n"
        " from types import SimpleNamespace\n"
        " COMPOUND_MODEL._compound_probe_aux=SimpleNamespace(dictionary=PROBE_AUX.dictionary,\n"
        "  coeff_bins=PROBE_AUX.coeff_bins,coeff_scales=PROBE_AUX.coeff_scales,\n"
        "  coeff_vocab_size=PROBE_AUX.coeff_vocab_size,num_atoms=PROBE_AUX.num_atoms)\n"
        " return COMPOUND_MODEL\n"
        "training.build_model=compound_build\n"
        "BEST_FID=[]\nBEST_IS=[]")
    replace(" step=int(payload['global_step']);epoch=int(payload['epoch'])",
        " step=int(payload['global_step']);epoch=int(payload['epoch'])\n"
        " payload=dict(payload,compound_transfer=COMPOUND_TRANSFER,config=dict(payload['config'],\n"
        "  architecture=ARCHITECTURE,compound_event_history=True,compound_event_order='raster_then_depth_atom_coefficient_pair'))")
    replace("  counters={int(s['step']) for s in optimizer.state.values()};assert len(counters)==1\n  INITIAL_ADAM_STEP=counters.pop()",
        "  INITIAL_ADAM_STEP=verify_live_optimizer(optimizer,COMPOUND_MODEL,COMPOUND_TRANSFER,INITIAL_RECOVERY['global_step'])")
    replace("  assert {int(s['step']) for s in optimizer.state.values()}=={INITIAL_ADAM_STEP+UPDATES}",
        "  assert verify_live_optimizer(optimizer,COMPOUND_MODEL,COMPOUND_TRANSFER,global_step)==INITIAL_ADAM_STEP+UPDATES")
    replace(" result=original_step(optimizer,*args,**kwargs);UPDATES+=1",
        " if UPDATES==0:\n"
        "  history_names=set(COMPOUND_TRANSFER['new_parameter_names'])\n"
        "  gradients=[p.grad.float().norm() for name,p in COMPOUND_MODEL.named_parameters() if name in history_names]\n"
        "  assert all(bool(torch.isfinite(g)) for g in gradients) and sum(float(g) for g in gradients)>0\n"
        "  record(VERIFY/('history-gradient-rank'+os.environ['RANK']+'.json'),dict(finite=True,nonzero=True,\n"
        "   total_gradient_norm=sum(float(g) for g in gradients),new_parameters=len(gradients)))\n"
        " result=original_step(optimizer,*args,**kwargs);UPDATES+=1")
    replace(" if UPDATES in (1,20):\n  tensors=PARAMETERS+",
        " if UPDATES in (1,20):\n"
        "  if os.environ['RANK']=='0':\n"
        "   from compound_fixed_probe import measure\n"
        "   probe=measure(COMPOUND_MODEL,BASE/'fixed-teacher-batch.pt',BASE/'inputs/resume-stage1-tokenizer.pt',ARGS,PARAMETERS[0].device)\n"
        "   record(VERIFY/('compound-probe-step'+str(UPDATES)+'.json'),probe)\n"
        "  tensors=PARAMETERS+")
    entry=entry.replace("architecture='physical-pair-scalar-rqtransformer-imagenet-1400m'",
        "architecture='gated_full_history_physical_compound_v2',compound_event_history=True,architecture_changed=True")
    entry=entry.replace("across8 GPUs and3 accumulation microbatches","across8 GPUs and4 accumulation microbatches")
    compile(entry,str(args.base/'entry.py'),'exec')
    (args.base/'entry.py').write_text(entry)
    shutil.copyfile(repo/'scripts/tools/compound_fixed_probe.py',args.base/'support/compound_fixed_probe.py')
    fixed=repo/'outputs/full-history-compound-training-20261006/gated-v2/fixed-teacher-batch.pt'
    shutil.copyfile(fixed,args.base/'fixed-teacher-batch.pt')
    for relative in ('entry.py','recipe.yaml','compound-transfer.json'):
        shutil.copyfile(args.base/relative,args.output/relative)
    evidence=dict(prepared=True,run_id=run_id,source=str(args.source),source_epoch=77,source_step=48202,
        common_weights_preserved=True,common_adam_preserved=True,common_adam_step=13772,
        new_adam_step=0,new_parameter_tensors=19,new_parameter_values=28337664,
        initial_lr=1e-6,lr_zero_epoch=100,scheduler_global_step=48202,
        world_size=8,global_batch=2048,microbatch=64,accumulation=4,
        rng_ranks_preserved=8,pilot_updates=20,automatic_continue_target_epoch=100,
        official_fid_images=50000,official_is_splits=10,
        source_metrics_carried_as_new_model_metrics=False,checkpoint=str(anchor),
        optimizer_ages=validate_optimizer_ages(migrated))
    (args.output/'preparation-complete.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(evidence),flush=True)


if __name__=='__main__':
    main()
