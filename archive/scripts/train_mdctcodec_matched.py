#!/usr/bin/env python3
"""Prepare or train one arm of the paired, fresh 6 kbps MDCTCodec experiment."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
os.environ.setdefault('VISQOL_BINARY',str(Path('outputs/visqol/bin/visqol').resolve()))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf
import soundfile as sf
import torch
import wandb

from src.mdctcodec_matched import MatchedData, MatchedModel, PairedAudioDataset, PairedAudit, TrainingCoefficientRange, tensor_hash
from src.stage1_setup import laser_model_kwargs
from src.training.common import _make_selected_checkpoint_artifact_callback

DEFAULT_ROOT=Path('outputs/mdctcodec_matched_6kbps')
SPEAKERS=['p360','p361','p362','p363','p364','p374','p376','s5']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_range_fix(root, source):
    if root.resolve()==source.resolve():raise ValueError('Preserve the original experiment in its own directory')
    if (root/'prepared.json').exists():
        print('Already prepared',root);return
    root.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((source/'protocol.json').read_text())
    assert sha(source/'manifest.json')==protocol['manifest_sha256']
    for name in ['manifest.json','calibration.json']:
        shutil.copyfile(source/name,root/name)
    for arm in ['laser','rvq']:
        item=protocol['initializations'][arm]
        assert sha(item['path'])==item['sha256']
        target=root/f'{arm}_initial.pt'
        shutil.copyfile(item['path'],target)
        assert sha(target)==item['sha256']
        item['path']=str(target.resolve())
    protocol['name']='mdctcodec-matched-scratch-6kbps-rangefix'
    protocol['group']='mdctcodec-matched-scratch-6kbps-rangefix-20260912'
    protocol['range_policy']={'name':'training_p999_fast_attack_slow_release',
        'margin':1.1,'release':0.001,'window':100,'guard_start':1000,'maximum_mean_clipping':0.05,
        'update_timing':'After each generator update; training coefficients only; no updates during validation/inference',
        'storage':'One FP32 bound per checkpoint in hyper_parameters; observer history in callback state',
        'rate':'Unchanged K2 x (13+7) bits x 150Hz; no adaptive per-frame or per-file side information'}
    protocol['stop_peer_on_failure']=True
    protocol['supersedes']={'root':str(source.resolve()),'protocol_sha256':sha(source/'protocol.json'),
        'reason':'Initialization-only range failed as the encoder coefficient scale evolved; 100% clipping observed.'}
    calibration=json.loads((root/'calibration.json').read_text())
    calibration['bounds_policy']='Initial bound only; training p99.9 observer subsequently tracks the learned coefficient distribution.'
    (root/'calibration.json').write_text(json.dumps(calibration,indent=2))
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    (root/'prepared.json').write_text(json.dumps({'manifest_sha256':protocol['manifest_sha256'],
        'protocol_sha256':sha(root/'protocol.json'),'initializations':protocol['initializations']},indent=2))
    print(json.dumps({'root':str(root),'range_policy':protocol['range_policy'],
                     'initializations':protocol['initializations']},indent=2))


def prepare(root):
    root.mkdir(parents=True,exist_ok=True)
    if (root/'prepared.json').exists():
        print('Already prepared',root);return
    if list(root.glob('*_initial.pt')):
        raise RuntimeError('Incomplete preparation; inspect before replacing initialization')
    old=json.loads(Path('outputs/mdctcodec_fair_comparison/manifest.json').read_text())
    excluded=set(old['excluded_stems'])
    excluded.update(Path(p).stem for item in old['items'] for p in item['source_paths'])
    audio_root=Path('/workspace/Projects/data/vctk/wav48_silence_trimmed')
    files=sorted(audio_root.glob('*/*_mic2.flac'))
    training=[str(p.resolve()) for p in files if p.parent.name not in SPEAKERS]
    assert len(training)==40936
    rng=random.Random(2026091203)
    prior_validation=json.loads(Path('outputs/mdctcodec_recovery/oyg7smih/validation_comparison/manifest.json').read_text())['validation']
    val,test=[],[]
    for speaker in SPEAKERS:
        pool=[p for p in files if p.parent.name==speaker and p.stem not in excluded and sf.info(p).duration>=2]
        rng.shuffle(pool)
        retained=[p for p in prior_validation if Path(p).parent.name==speaker]
        additional=16-len(retained)
        assert 0<=additional and len(pool)>=additional+25,(speaker,len(pool))
        val.extend(retained+[str(p.resolve()) for p in pool[:additional]])
        test.extend(str(p.resolve()) for p in pool[additional:additional+25])
    assert not set(val)&set(test) and not (set(val)|set(test))&set(training)
    manifest={'seed':1234,'selection_seed':2026091203,'sample_rate':48000,
              'train':training,'validation':val,'test':test,'heldout_speakers':SPEAKERS,
              'excluded_stems':sorted(excluded),
              'evaluation_file_sha256':{p:sha(p) for p in val+test},
              'protocol':'128 validation / fresh 200 test recordings; 16 / 25 per held-out speaker. '
                         'Validation extends the existing test-disjoint 116-file set. New test files exclude '
                         'all files reserved in the preceding recovered and corrected evaluations.'}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    with initialize_config_dir(config_dir=str(Path('configs').resolve()),version_base=None):
        cfg=compose(config_name='vctk_mdctcodec_stage1_6kbps')
    kwargs=laser_model_kwargs(cfg.model,cfg.train,in_channels=1,image_size=128,audio_sample_rate=48000)
    kwargs.update(log_images_every_n_steps=0,enable_val_latent_visuals=False,
                  coefficient_quantization_bits=0,coefficient_quantization_max=None)
    pl.seed_everything(1234,workers=True)
    laser=MatchedModel(**kwargs)
    shared={name:{k:v.detach().cpu().clone() for k,v in getattr(laser,name).state_dict().items()}
            for name in ['encoder','decoder','discriminator','pre_bottleneck','post_bottleneck']}
    shared_hashes={name:tensor_hash(state) for name,state in shared.items()}
    # Determine a fixed signed127 range using ONLY the fresh encoder on training crops.
    laser.to('cuda:0').eval()
    dataset=PairedAudioDataset(training,seed=manifest['seed'])
    calibration_ids=random.Random(19013).sample(range(len(dataset)),1024)
    coefficients=[]
    with torch.inference_mode():
        for start in range(0,len(calibration_ids),32):
            x=torch.stack([dataset[(-1,i)][0] for i in calibration_ids[start:start+32]]).to('cuda:0')
            codes=laser.encode(x)[2]
            coefficients.append(codes.values.abs().float().flatten().cpu())
    bound=float(torch.quantile(torch.cat(coefficients),0.999))
    assert math.isfinite(bound) and bound>0
    laser.bottleneck.coefficient_quantization_bits=7
    laser.bottleneck.coefficient_quantization_max=bound
    laser.hparams['coefficient_quantization_bits']=7
    laser.hparams['coefficient_quantization_max']=bound
    laser.cpu()
    # The second architecture consumes a different RNG amount; explicitly copy ALL
    # common state, including the discriminator, rather than relying on seed alone.
    rvq_kwargs=dict(kwargs,bottleneck_type='mdctcodec_rvq',num_embeddings=1024,
                    rq_code_depth=4,dictionary_loss_weight=10.0,
                    coefficient_quantization_bits=0,coefficient_quantization_max=None)
    pl.seed_everything(1234,workers=True)
    rvq=MatchedModel(**rvq_kwargs)
    for name,state in shared.items():
        getattr(rvq,name).load_state_dict(state,strict=True)
        assert tensor_hash(getattr(laser,name).state_dict())==tensor_hash(getattr(rvq,name).state_dict())
    initial={}
    for name,model in [('laser',laser),('rvq',rvq)]:
        path=root/f'{name}_initial.pt'
        torch.save({'state_dict':model.state_dict(),'hyper_parameters':dict(model.hparams)},path)
        initial[name]={'path':str(path.resolve()),'sha256':sha(path),
                       'bottleneck':model.bottleneck_type,'parameter_count':sum(p.numel() for p in model.parameters())}
    calibration={'coefficient_max':bound,'quantile':0.999,'crops':1024,
                 'source':'fresh shared encoder; random fresh LASER dictionary; training split only',
                 'paths':[training[i] for i in calibration_ids],
                 'bounds_policy':'fixed throughout this experiment; clipping logged; no test recalibration'}
    (root/'calibration.json').write_text(json.dumps(calibration,indent=2))
    protocol={'name':'mdctcodec-matched-scratch-6kbps','seed':1234,
              'generator_updates':200000,'discriminator_updates':200000,
              'batch_size':48,'drop_last':True,'batches_per_epoch':len(training)//48,
              'crop_samples':7960,'sample_rate':48000,'precision':'bf16-mixed',
              'optimizer':'AdamW','weight_decay':0.01,'lr':0.0002,'betas':[0.8,0.99],
              'lr_decay_per_epoch':0.999,'discriminator_first':True,
              'common_losses':{'mdct_mse':250,'mel_mae_plus_mse':45,'adversarial':0.1,'feature_matching':0.1,'commitment':2.5},
              'rvq':'authors unchanged 4 x 1024, projected dimension32; codebook loss10, no dropout',
              'laser':'8192 shared atoms, K2, signed127 coefficients from update0; alternating residual updates',
              'shared_initialization_sha256':shared_hashes,'initializations':initial,
              'manifest_sha256':sha(root/'manifest.json'),'coefficient_max':bound,
              'upstream_quantizer_sha256':sha('src/models/mdctcodec_quantize.py'),
              'validation':'128 fixed full recordings every epoch; checkpoint selection only by ViSQOL audio48k',
              'test':'200 separate fixed full recordings, evaluated after both runs complete',
              'checkpoint_upload':'latest + top3 validation ViSQOL, every5 completed epochs and successful exit',
              'scope':'One paired seed. Equal common initialization, examples, crops, update counts and waveform losses. '
                      'Bottleneck architecture, parameter count, auxiliary losses and learning algorithms differ. '
                      'The released pretrained model is an evaluation reference, not an equal-budget arm.'}
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    (root/'prepared.json').write_text(json.dumps({'manifest_sha256':protocol['manifest_sha256'],
        'protocol_sha256':sha(root/'protocol.json'),'initializations':initial},indent=2))
    print(json.dumps({'root':str(root),'training':len(training),'validation':len(val),'test':len(test),
          'coefficient_max':bound,'shared_hashes':shared_hashes},indent=2))


def train(args, *, model_class=MatchedModel, extra_callbacks_factory=None):
    root=args.root
    protocol=json.loads((root/'protocol.json').read_text())
    manifest=json.loads((root/'manifest.json').read_text())
    assert sha(root/'manifest.json')==protocol['manifest_sha256']
    output=args.output or root/args.arm
    output.mkdir(parents=True,exist_ok=True)
    if (output/'run.json').exists() and not args.resume:
        raise RuntimeError('Existing run: resume explicitly or use a new output directory')
    source=protocol['initializations'][args.arm]
    assert sha(source['path'])==source['sha256']
    initial=torch.load(source['path'],map_location='cpu',weights_only=False)
    model=model_class(**{**initial['hyper_parameters'], **protocol.get('continuation_hparams', {})})
    model.load_state_dict(initial['state_dict'],strict=True)
    model.output_path=str(output)
    model.metric_workers=args.metric_workers
    for name,digest in protocol['shared_initialization_sha256'].items():
        assert tensor_hash(getattr(model,name).state_dict())==digest,name
    # Model construction/calibration cannot perturb the training seed or crop stream.
    pl.seed_everything(protocol['seed'],workers=True)
    budget=args.updates or protocol['generator_updates']
    data=MatchedData(manifest,workers=args.workers,batch_size=protocol['batch_size'],
                     validation_limit=args.validation_limit)
    previous=json.loads((output/'run.json').read_text()) if args.resume and (output/'run.json').exists() else {}
    logger=WandbLogger(entity='helloimlixin-rutgers',project='laser',
        name=f'{protocol["name"]}-{args.arm}-'+('preflight' if args.smoke else f'{budget//1000}k'),
        group=protocol.get('group','mdctcodec-matched-scratch-6kbps-20260912'),save_dir=str(output),
        id=previous.get('id'),resume='must' if previous else None,mode=args.mode,
        log_model=False,config={**protocol,'arm':args.arm,'actual_budget':budget,'smoke':args.smoke},
        tags=['mdctcodec','6kbps-target' if protocol.get('rate_is_target_only') else '6kbps',
              'matched','scratch',args.arm,'preflight' if args.smoke else 'stage1'])
    run=logger.experiment
    if args.resume:
        restored=torch.load(args.resume,map_location='cpu',weights_only=False)
        restored_updates=int(restored['state_dict']['_manual_train_step'])
        order_path=output/'data_order.jsonl'
        if order_path.exists():
            lines=[line for line in order_path.read_text().splitlines() if line.strip()]
            retained=[line for line in lines if json.loads(line)['generator_updates']<=restored_updates]
            if len(lines)!=len(retained):
                order_path.with_suffix('.before_resume.jsonl').write_text('\n'.join(lines)+'\n')
                order_path.write_text('\n'.join(retained)+'\n')
        if (output/'completion.json').exists():
            (output/'completion.json').rename(output/'completion_before_resume.json')
        run.config.update({'actual_budget':budget, 'last_resume_generator_updates':restored_updates,
                           'last_resume_checkpoint':str(args.resume), 'last_resume_sha256':sha(args.resume)},allow_val_change=True)
        if previous and args.mode == 'online' and protocol.get('continuation'):
            recovery_source = wandb.Artifact(f'mdctcodec-continuation-recovery-{run.id}', type='experiment',
                metadata={'restored_generator_updates':restored_updates,'checkpoint_sha256':sha(args.resume)})
            recovery_source.add_file(str(args.resume), name='resume.ckpt')
            for path in [Path(__file__), Path('scripts/tools/continue_mdctcodec_matched.py'),
                         Path('scripts/tools/run_mdctcodec_long.py'), Path('src/training/mdctcodec_continuation.py')]:
                recovery_source.add_file(str(path), name='source/' + path.name)
            run.log_artifact(recovery_source, aliases=['latest', f'from-step-{restored_updates}']).wait()
    run.summary['status']='running'
    (output/'run.json').write_text(json.dumps({'id':run.id,'url':run.url,'arm':args.arm,
        'generator_update_budget':budget,'manifest_sha256':protocol['manifest_sha256']},indent=2))
    if args.mode=='online' and not previous:
        artifact=wandb.Artifact(f'mdctcodec-matched-protocol-{run.id}',type='experiment',metadata={'arm':args.arm})
        for path in [root/'protocol.json',root/'manifest.json',root/'calibration.json',Path(source['path']),
                     Path(__file__),Path('src/mdctcodec_matched.py'),Path('src/models/mdctcodec_rvq.py'),
                     Path('src/models/mdctcodec_quantize.py'),Path('src/models/laser.py')]:
            artifact.add_file(str(path),name=path.name)
        for name in ['source.tar.gz','source_files.json','environment.txt','preflight_verified.json','restore_verified.json']:
            if (root/name).is_file():artifact.add_file(str(root/name),name=name)
        for path in protocol.get('runtime_source_files', []):
            artifact.add_file(path, name=path)
        if protocol.get('continuation'):
            for path in ['scripts/tools/continue_mdctcodec_matched.py', 'src/training/mdctcodec_continuation.py',
                         'src/audio_research_media.py', 'src/training/common.py']:
                artifact.add_file(path, name=path)
            artifact.add_file(str(root/'prepared.json'), name='prepared.json')
            artifact.add_file(str(args.resume), name='resume_200k.ckpt')
            parent = protocol['continuation']['lineage'][args.arm]['parent_run']['id']
            run.use_artifact(f'helloimlixin-rutgers/laser/model-{parent}-selected-checkpoints:complete')
        run.log_artifact(artifact).wait()
    checkpoint=ModelCheckpoint(dirpath=str(output/'checkpoints'),monitor='val/audio_visqol_audio48k',
        mode='max',save_top_k=3,save_last=True,every_n_epochs=1,save_on_train_epoch_end=False,
        filename=f'{args.arm}-{{epoch:03d}}-{{step:07d}}')
    upload=_make_selected_checkpoint_artifact_callback(pl.Callback)(checkpoint,every_n_epochs=5)
    callbacks=[checkpoint,upload,PairedAudit(output,budget)]
    if extra_callbacks_factory:
        callbacks.extend(extra_callbacks_factory(output, manifest, args.arm))
    if args.arm=='laser' and protocol.get('range_policy'):
        callbacks.append(TrainingCoefficientRange(output,**protocol['range_policy']))
    if protocol.get('continuation') and not args.smoke:
        from src.training.mdctcodec_continuation import AudioContinuationMedia, GracefulBudget
        callbacks.extend([AudioContinuationMedia(output, manifest, args.arm), GracefulBudget(output)])
    trainer=pl.Trainer(accelerator='gpu',devices=1,precision='bf16-mixed',
        max_steps=budget*2,max_epochs=math.ceil(budget/protocol['batches_per_epoch'])+2,
        callbacks=callbacks,logger=logger,
        enable_progress_bar=False,enable_model_summary=False,num_sanity_val_steps=0,
        log_every_n_steps=20,deterministic='warn',benchmark=False,
        limit_train_batches=args.train_batches if args.train_batches else 1.0)
    if args.train_batches:
        trainer.fit_loop.max_epochs=math.ceil(budget/args.train_batches)
    trainer.fit(model,datamodule=data,ckpt_path=str(args.resume) if args.resume else None)
    actual=int(model._manual_train_step)
    if actual!=budget:
        if getattr(model, 'continuation_stopped', False):
            trainer.save_checkpoint(str(output/'checkpoints/last.ckpt'))
            record = {'status':'paused_budget','generator_updates':actual,'target':budget,'url':run.url}
            (output/'completion.json').write_text(json.dumps(record,indent=2))
            run.summary.update(record)
            logger.finalize('success'); wandb.finish()
            return
        raise RuntimeError(f'Incomplete update budget: {actual}/{budget}')
    trainer.save_checkpoint(str(output/'checkpoints/final.ckpt'))
    if args.smoke and checkpoint.best_model_score is None:
        # A two-update resume ends mid-epoch; explicitly exercise full-audio validation.
        metrics = trainer.validate(model, datamodule=data, verbose=False)[0]
        checkpoint.best_model_score = torch.tensor(metrics['val/audio_visqol_audio48k'])
        checkpoint.best_model_path = str(output/'checkpoints/final.ckpt')
    if not checkpoint.best_model_path:
        raise RuntimeError('Training completed without a validation-selected checkpoint')
    completion={'status':'complete','generator_updates':actual,'lightning_global_step':trainer.global_step,
        'completed_full_epochs':actual//protocol['batches_per_epoch'],
        'best_checkpoint':checkpoint.best_model_path,'best_validation_visqol':float(checkpoint.best_model_score),
        'best_three':{p:float(v) for p,v in checkpoint.best_k_models.items()},
        'latest_checkpoint':checkpoint.last_model_path,'url':run.url,'manifest_sha256':protocol['manifest_sha256']}
    # Explicit wait makes completion mean all restartable selected checkpoints are online.
    if args.mode=='online':
        artifact=wandb.Artifact(f'model-{run.id}-selected-checkpoints',type='model',metadata=completion)
        for path in [checkpoint.last_model_path,*checkpoint.best_k_models]:
            artifact.add_file(path,name=Path(path).name)
        committed=run.log_artifact(artifact,aliases=['latest','best-plus-last','complete']).wait()
        completion['artifact']=committed.qualified_name
    (output/'completion.json').write_text(json.dumps(completion,indent=2))
    run.summary.update(completion)
    logger.finalize('success')
    wandb.finish()
    print('MATCHED_COMPLETE',json.dumps(completion),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--prepare-range-fix-from',type=Path)
    parser.add_argument('--arm',choices=['laser','rvq'])
    parser.add_argument('--output',type=Path)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--mode',choices=['online','disabled'],default='online')
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--metric-workers',type=int,default=8)
    parser.add_argument('--updates',type=int,default=0)
    parser.add_argument('--validation-limit',type=int,default=0)
    parser.add_argument('--train-batches',type=int,default=0)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if not args.prepare and not args.prepare_range_fix_from and not args.arm:parser.error('--arm required for training')
    if (args.updates or args.validation_limit or args.train_batches) and not args.smoke:
        parser.error('Reduced-budget options require --smoke')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=True
    if args.prepare_range_fix_from:prepare_range_fix(args.root,args.prepare_range_fix_from)
    elif args.prepare:prepare(args.root)
    else:train(args)


if __name__=='__main__':main()
