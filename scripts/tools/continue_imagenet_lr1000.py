"""Resume current ImageNet CRPS training with exactly 1000x saved LR amplitude."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import continue_imagenet_fid16379 as recovery

RUN_ID = 'imagenet-rfid421-crps005-lr1000-floor0-8h100-20261006'
FACTOR = 1000.
record = recovery.record


def install_loss_tracking(base):
    repository = Path(__file__).resolve().parents[2]
    shutil.copyfile(repository / 'src/training/training_loss_tracker.py',
                    base / 'source/runtime/src/training/training_loss_tracker.py')
    p = base / 'entry.py'
    code = p.read_text()
    def replace(old, new):
        nonlocal code
        assert code.count(old) == 1, old
        code = code.replace(old, new)
    replace('UPDATES=0\n', 'UPDATES=0\nTRAIN_OBJECTIVE_SUM=None\nTRAIN_LOSS_TRACKER=None\nLATEST_TRAIN_LOSS={}\n')
    replace('def physical_pair_objective(atom_logits,coeff_logits,atoms,probabilities,bins,weight,accumulation):\n',
            'def physical_pair_objective(atom_logits,coeff_logits,atoms,probabilities,bins,weight,accumulation):\n'
            ' global TRAIN_OBJECTIVE_SUM\n')
    replace(" audit=VERIFY/('crps-step20-rank'+os.environ['RANK']+'.json')\n",
            " count=atoms.shape[0]\n"
            " contribution=torch.stack((total.detach()*accumulation,classification.detach(),crps.detach(),weight.detach()))*count\n"
            " contribution=torch.cat((contribution,contribution.new_tensor([count])))\n"
            " if TRAIN_OBJECTIVE_SUM is None:TRAIN_OBJECTIVE_SUM=contribution\n"
            " else:TRAIN_OBJECTIVE_SUM.add_(contribution)\n"
            " audit=VERIFY/('crps-step20-rank'+os.environ['RANK']+'.json')\n")
    replace(' global UPDATES,INITIAL_ADAM_STEP,LAST_STEP_END,PROFILER\n',
            ' global UPDATES,INITIAL_ADAM_STEP,LAST_STEP_END,PROFILER,TRAIN_OBJECTIVE_SUM,TRAIN_LOSS_TRACKER,LATEST_TRAIN_LOSS\n')
    replace(' result=original_step(optimizer,*args,**kwargs);UPDATES+=1\n',
            ' result=original_step(optimizer,*args,**kwargs);UPDATES+=1\n'
            ' assert TRAIN_OBJECTIVE_SUM is not None\n'
            ' dist.all_reduce(TRAIN_OBJECTIVE_SUM,op=dist.ReduceOp.SUM)\n'
            ' samples=int(TRAIN_OBJECTIVE_SUM[-1]);values=(TRAIN_OBJECTIVE_SUM[:-1]/samples).tolist()\n'
            ' LATEST_TRAIN_LOSS=TRAIN_LOSS_TRACKER.update(*values,samples,INITIAL_RECOVERY["global_step"]+UPDATES)\n'
            ' TRAIN_OBJECTIVE_SUM=None\n'
            ' if UPDATES in (1,20):\n'
            '  record(VERIFY/("loss-step"+str(UPDATES)+"-rank"+os.environ["RANK"]+".json"),dict(metrics=LATEST_TRAIN_LOSS,state=TRAIN_LOSS_TRACKER.state_dict()))\n')
    replace("  INITIAL_RECOVERY=recovery_metadata(payload)\n",
            "  from src.training.training_loss_tracker import TrainingLossTracker\n"
            "  global TRAIN_LOSS_TRACKER\n"
            "  TRAIN_LOSS_TRACKER=TrainingLossTracker(payload['global_step'],payload.get('train_loss_tracking'))\n"
            "  INITIAL_RECOVERY=recovery_metadata(payload)\n")
    replace(" step=int(payload['global_step']);epoch=int(payload['epoch'])\n",
            " step=int(payload['global_step']);epoch=int(payload['epoch'])\n"
            " if TRAIN_LOSS_TRACKER is not None:\n"
            "  assert TRAIN_LOSS_TRACKER.state_dict()['last_step']==step\n"
            "  payload=dict(payload,train_loss_tracking=TRAIN_LOSS_TRACKER.state_dict())\n")
    replace(' config.update(resume_source_epoch=',
            " config.update(training_loss_reporting='sample-weighted global batch across8 GPUs and3 accumulation microbatches',\n"
            "  training_loss_components='cross entropy and CRPS separately; resumable EMA decay0.95',manual_lr_increase_factor=1000.)\n"
            ' config.update(resume_source_epoch=')
    replace(' WB=original_init(*args,**kwargs)\n',
            ' WB=original_init(*args,**kwargs)\n'
            ' original_log=WB.log\n'
            ' def log(data,*args,**kwargs):\n'
            '  if "train/loss" in data and LATEST_TRAIN_LOSS:\n'
            '   data["train/loss_last_microbatch"]=data["train/loss"]\n'
            '   data.update(LATEST_TRAIN_LOSS)\n'
            '  return original_log(data,*args,**kwargs)\n'
            ' WB.log=log\n')
    replace(' if baseline.exists():\n',
            " if baseline.exists() and INITIAL_RECOVERY['original_rqtransformer_metrics'] is not None:\n")
    # The source block is replaced by automatic rewinds; keep metadata outside it.
    p.write_text(code)


def compare_tensors(source, restored):
    import torch
    compared = 0
    for key, tensor in source['state_dict'].items():
        assert torch.equal(tensor, restored['state_dict'][key]), key
        compared += tensor.numel()
    for key, fields in source['optimizer']['state'].items():
        for field, tensor in fields.items():
            if torch.is_tensor(tensor):
                assert torch.equal(tensor, restored['optimizer']['state'][key][field]), (key, field)
                compared += tensor.numel()
    for rank, streams in enumerate(source['rng_state_by_rank']):
        for name, tensor in streams.items():
            assert torch.equal(tensor, restored['rng_state_by_rank'][rank][name])
    assert restored['objective_revision'] == source['objective_revision']
    return compared


def prepare(base, evidence, parent_base, parent_evidence):
    import torch
    import yaml
    torch.set_num_threads(4)
    audit = evidence / 'continuation-20261005'
    checkpoints = audit / 'train/checkpoints'
    checkpoints.mkdir(parents=True, exist_ok=True)
    latest = checkpoints / 'last.pt'
    assert not latest.exists(), 'Refuse to prepare over an existing continuation'
    for directory in ('source/runtime', 'support'):
        shutil.copytree(parent_base / directory, base / directory, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('imagenet_scale_lr.py','imagenet_fid_rewind_guard.py','imagenet_fid_rewind_supervisor.py','continue_imagenet_lr1000.py'):
        shutil.copyfile(Path(__file__).with_name(name), base / 'support' / name)
    sys.path[:0] = [str(base / 'source/runtime'), str(base / 'support')]
    from imagenet_scale_lr import scale_saved_learning_rate
    from imagenet_zero_floor_lr import create_scheduler
    from src.training import k4_checkpoint_io as checkpoint_io
    from src.training.full_resume_upload import recovery_metadata
    inputs = base / 'inputs'; inputs.mkdir(exist_ok=True)
    parent_recipe = yaml.safe_load((parent_base / 'recipe.yaml').read_text())
    parent_id = parent_recipe['options']['wandb_id']
    pins = parent_base / 'final-cpu-upload'
    raw = torch.load(pins / 'last.pt', map_location='cpu', weights_only=False, mmap=True)
    source = inputs / 'source-current-full.pt';os.link(pins / 'last.pt', source)
    original = recovery_metadata(raw)
    assert original['adam_step'] == original['global_step']-34430
    assert original['rng_ranks'] == original['world_size'] == 8 and original['adam_parameters'] == 782
    assert original['next_microbatch'] % 3 == 0
    assert raw['scheduler']['last_epoch'] == original['global_step']-35056
    optimizer, state = scale_saved_learning_rate(raw['optimizer'], raw['scheduler'], FACTOR)
    fake = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=optimizer['param_groups'][0]['lr'])
    assert create_scheduler(fake, initial_lr=state['policy']['initial_lr'], min_lr=0.,
        total_steps=state['policy']['total_steps'], completed_steps=state['last_epoch'],state_dict=state).state_dict()==state
    recipe=parent_recipe
    recipe['options'].update(checkpoint=str(inputs/'resume-stage1-tokenizer.pt'),output=str(base/'production/train'),
        checkpoint_dir=str(checkpoints),wandb_id=RUN_ID,lr_schedule_restart_id=RUN_ID,
        wandb_name=f"ImageNet rFID4.21 K4 | CE+CRPS0.05 | current-state LR x1000={optimizer['param_groups'][0]['lr']:.3e} | floor0 | 8 H100")
    recipe['options'].pop('resume_checkpoint',None)
    for name in ('resume-stage1-tokenizer.pt','resume-weights-inception-2015-12-05-6726825d.pth'):
        os.link(parent_base/'inputs'/name,inputs/name)
    for name in ('torch-cache','inductor-cache'):
        (base/name).symlink_to((parent_base/name).resolve(),target_is_directory=True)
    shutil.copyfile(parent_base/'official-baseline.json',base/'official-baseline.json')
    shutil.copyfile(parent_base/'entry.py',base/'entry.py')
    install_loss_tracking(base)
    entry=(base/'entry.py').read_text();start=entry.index(' config.update(resume_source_epoch=');end=entry.index(' WB=original_init(*args,**kwargs)',start)
    entry=entry[:start]+(
        f" config.update(resume_source_epoch={original['epoch']},resume_source_global_step={original['global_step']},\n"
        f"  source_run={'helloimlixin-rutgers/laser/'+parent_id!r},source_checkpoint_step={original['global_step']},\n"
        f"  source_checkpoint_epoch={original['epoch']},source_checkpoint_fid={original['fid']!r},\n"
        "  source_optimizer_restored=True,trained_source_optimizer_available=True,\n"
        f"  optimizer_initialization='preserved current trained AdamW moments and all782 counters at{original['adam_step']}',\n"
        "  rng_initialization='preserved all8 current checkpoint RNG streams and microbatch cursor',\n"
        "  evaluation_protocol='Official RQ-Transformer FID/10-split IS only; validation50k/generated50k',\n"
        "  learning_rate_schedule='saved zero-floor cosine through epoch100; amplitude multiplied1000; strict FID rewind',\n"
        "  lr_continuation='trained state and schedule clock/history preserved; LR amplitude x1000',\n"
        f"  restart_initial_lr={optimizer['param_groups'][0]['lr']!r},source_scheduler_step={state['last_epoch']},\n"
        f"  source_learning_rate={original['saved_learning_rates'][0]!r},manual_lr_increase_factor=1000.,\n"
        "  lr_floor_before=0.,lr_floor_after=0.,automatic_fid_rewind=True,cosine_decay_end_epoch=100)\n"
    )+entry[end:]
    (base/'entry.py').write_text(entry);(base/'recipe.yaml').write_text(yaml.safe_dump(recipe,sort_keys=False))
    from imagenet_fid_rewind_guard import install_guard
    install_guard(base)
    winner_payloads={}
    for kind,name in (('fid','best-fid-resume.pt'),('is','best-is-resume.pt')):
        winner=torch.load(pins/name,map_location='cpu',weights_only=False,mmap=True)
        metadata=recovery_metadata(winner); metrics=metadata['original_rqtransformer_metrics'];assert metrics is not None
        score=metrics['fid'] if kind=='fid' else metrics['inception_score']
        alias=checkpoints/f"best_{'fid' if kind=='fid' else 'is'}_{score:.4f}_epoch_{metadata['epoch']:03d}.pt"
        winner_payloads[kind]=(winner,metadata,alias,score)
    fid_alias=winner_payloads['fid'][2];is_alias=winner_payloads['is'][2]
    rankings=dict(best_fid=[(winner_payloads['fid'][3],str(fid_alias))],best_inception=[(winner_payloads['is'][3],str(is_alias))])
    def payload(source, opt, sched):
        return dict(source,optimizer=opt,scheduler=sched,config=dict(source['config'],**recipe['options']),**rankings,
            lr_revision=dict(source_run=parent_id,factor=FACTOR,old_learning_rate=source['optimizer']['param_groups'][0]['lr'],
                new_learning_rate=opt['param_groups'][0]['lr'],schedule_clock_preserved=True,optimizer_moments_reset=False))
    os.environ.update(LASER_CHECKPOINT_STAGING_DIR=str(base/'checkpoint-staging'),
        LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(base/'checkpoint-upload-cache'),LASER_CHECKPOINT_IMMUTABLE_FILES='1')
    # Revise winning full checkpoints too, so strict rewind keeps the new scale.
    winner_proofs={}
    for kind,(winner,metadata,alias,score) in winner_payloads.items():
        wopt,wsched=scale_saved_learning_rate(winner['optimizer'],winner['scheduler'],FACTOR)
        checkpoint_io.atomic_torch_save(payload(winner,wopt,wsched),alias)
        restored=torch.load(checkpoint_io._checkpoint_upload_source(alias),map_location='cpu',weights_only=False,mmap=True)
        compared=compare_tensors(winner,restored);assert restored['scheduler']==wsched
        winner_proofs[kind]=dict(epoch=metadata['epoch'],global_step=metadata['global_step'],score=score,
            source_lr=metadata['saved_learning_rates'][0],revised_lr=wopt['param_groups'][0]['lr'],
            model_and_adam_tensors_identical=True,rank_rng_identical=True,tensor_values_compared=compared)
        del restored
    checkpoint_io.atomic_torch_save(payload(raw,optimizer,state),latest)
    (checkpoints/'resume-anchor.pt').symlink_to(latest.resolve().relative_to(checkpoints))
    restored=torch.load(checkpoint_io._checkpoint_upload_source(latest),map_location='cpu',weights_only=False,mmap=True)
    compared=compare_tensors(raw,restored)
    assert restored['scheduler']==state
    assert {k:v for k,v in state.items() if k!='multiplier'}=={k:v for k,v in raw['scheduler'].items() if k!='multiplier'}
    metadata=recovery_metadata(restored)
    with source.open('rb') as stream:digest=hashlib.file_digest(stream,'md5').hexdigest()
    record(audit/'source-parent.json',dict(base=str(parent_base),run_id=parent_id))
    record(audit/'source-state-verification.json',original)
    record(audit/'restart-state-verification.json',dict(metadata=metadata,source_md5=digest,
        revision=restored['lr_revision'],tensor_values_compared=compared,model_and_adam_tensors_identical=True,
        rank_rng_identical=True,schedule_clock_preserved=True,objective_revision=restored['objective_revision'],winners=winner_proofs))
    record(audit/'source-checkpoint-selection.json',dict(file=str(source),epoch=original['epoch'],
        global_step=original['global_step'],adam_step=original['adam_step'],fid=original['fid'],md5=digest,
        bytes=source.stat().st_size,full_optimizer_state=True))
    for name in ('entry.py','recipe.yaml','official-baseline.json'):shutil.copyfile(base/name,audit/name)
    (evidence/'status.json').symlink_to('continuation-20261005/status.json')
    print(json.dumps(dict(prepared=True,run=RUN_ID,source_step=original['global_step'],
        old_lr=original['saved_learning_rates'][0],new_lr=metadata['saved_learning_rates'][0],factor=FACTOR,
        tensor_values_compared=compared,winners=winner_proofs)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','upload-recovery','supervise','train-once'])
    for name in ('base','evidence','key-file'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--parent-base',type=Path);p.add_argument('--parent-evidence',type=Path)
    args=p.parse_args()
    if args.action=='prepare':
        assert args.parent_base is not None and args.parent_evidence is not None
        prepare(args.base,args.evidence,args.parent_base,args.parent_evidence);return
    import yaml
    run_id=yaml.safe_load((args.base/'recipe.yaml').read_text())['options']['wandb_id']
    recovery.RUN_ID=recovery.runner.RUN_ID=run_id
    recovery.RUN_PATH=recovery.runner.RUN_PATH='helloimlixin-rutgers/laser/'+run_id
    parent=json.loads((args.evidence/'continuation-20261005/source-parent.json').read_text())
    recovery.PARENT,recovery.SOURCE_RUN_ID=Path(parent['base']),parent['run_id']
    if args.action=='upload-recovery':recovery.upload_recovery(args.base,args.evidence,args.key_file)
    elif args.action=='train-once':recovery.runner.supervise(args.base,args.evidence,args.key_file)
    else:
        from imagenet_fid_rewind_supervisor import supervise_with_rewinds
        supervise_with_rewinds(args.base,args.evidence,args.key_file,driver=Path(__file__).resolve())


if __name__=='__main__':main()
