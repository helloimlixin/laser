#!/usr/bin/env python3
"""Resume the hard-rate codec pair with longer crops and preserved optimizer state."""
import argparse
import json
import math
import os
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO))
import torch
from archive.scripts.train_mdctcodec_matched import train,sha
from scripts.tools.continue_mdctcodec_matched import prepare as prepare_optimizer_continuation
from src.mdctcodec_matched import tensor_hash
from src.mdctcodec_hard6k import HardRateLASER,HardRateRVQ,HardRateStatistics,reconstruct_hard6k
from src.training.mdctcodec_continuation import AudioContinuationMedia,GracefulBudget

SOURCE=REPO/'outputs/mdctcodec_k4_a4096_hard6k_20260913'
ROOT=REPO/'outputs/mdctcodec_k4_a4096_hard6k_long_20260913'


def prepare(root,source,target):
    if (root/'long_prepared.json').exists():
        saved=json.loads((root/'long_prepared.json').read_text())
        assert saved['protocol_sha256']==sha(root/'protocol.json') and saved['target']==target
        return
    if root.exists() and list(root.iterdir()):raise RuntimeError('Inspect incomplete preparation before retrying')
    parent_protocol=json.loads((source/'protocol.json').read_text())
    assert parent_protocol['hard_rate_cap_bps']==6000
    prepare_optimizer_continuation(root,source,target)
    protocol=json.loads((root/'protocol.json').read_text())
    manifest=json.loads((root/'manifest.json').read_text())
    epochs=[]
    for arm in ('laser','rvq'):
        path=root/arm/'resume_200k.ckpt';state=torch.load(path,map_location='cpu',weights_only=False)
        # This is a new, explicitly bounded extension; parent compute remains in its lineage.
        state['callbacks']['HardRateStatistics']={'elapsed_seconds':0.}
        torch.save(state,path)
        parent=torch.load(source/arm/'checkpoints/last.ckpt',map_location='cpu',weights_only=False)
        assert tensor_hash(state['state_dict'])==tensor_hash(parent['state_dict'])
        line=protocol['continuation']['lineage'][arm]
        line['resume_sha256']=sha(path);epochs.append(state['epoch'])
    assert len(set(epochs))==1
    protocol.update(name='mdctcodec-k4-a4096-hard6k-long-crops',group='mdctcodec-k4-a4096-hard6k-long-20260913',
        crop_samples=31960,batch_size=12,batches_per_epoch=len(manifest['train'])//12,
        compute_ceiling_gpu_hours_per_arm=24,
        training_entrypoint='scripts/tools/continue_mdctcodec_hard6k.py',
        scope='Continuation of both matched 200k hard-rate codecs to 600k total updates. Same continuation crops, batches, losses and LR schedules for both arms. '
              'Parent model and optimizer states are preserved. Longer crops change the training curriculum; this is not a training-duration-only ablation.',
        crop_curriculum={'parent_samples':7960,'parent_batch':48,'continuation_samples':31960,'continuation_batch':12,
            'audio_samples_per_update_before':7960*48,'audio_samples_per_update_after':31960*12,
            'purpose':'Reduce short-crop header overhead and train with longer temporal context; keep audio seconds/update nearly constant.'})
    protocol['max_epochs']=epochs[0]+math.ceil(protocol['continuation']['additional_generator_updates_per_arm']/protocol['batches_per_epoch'])+2
    protocol['continuation']['frozen_tts_codec']='New K4 stage2 is pending compatible token caches; old K2 priors remain held.'
    protocol['continuation']['parent_batches_per_epoch']=parent_protocol['batches_per_epoch']
    protocol['continuation']['join']='Both continue saved epoch235 at batch0; completed parent data audit is retained. New epochs use 12 x 31960 samples.'
    protocol['runtime_source_files']=list(dict.fromkeys(protocol['runtime_source_files']+[
        'scripts/tools/continue_mdctcodec_hard6k.py','scripts/tools/continue_mdctcodec_matched.py',
        'scripts/tools/run_mdctcodec_hard6k_pair.py','src/mdctcodec_matched.py',
        'docs/mdctcodec-hard6k-long-2026-09-13.md']))
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    record={'protocol_sha256':sha(root/'protocol.json'),'target':target,'lineage':protocol['continuation']['lineage'],
            'model_and_optimizers_preserved':True,'parent_elapsed_time_reset_for_new_extension':True}
    (root/'prepared.json').write_text(json.dumps(record,indent=2))
    (root/'long_prepared.json').write_text(json.dumps(record,indent=2))
    print('HARD6K_LONG_PREPARED',json.dumps(record),flush=True)


def smoke_checkpoint(source,output):
    state=torch.load(source,map_location='cpu',weights_only=False)
    output.mkdir(parents=True,exist_ok=True)
    # Reduced validation is solely a runtime check and cannot rank production checkpoints.
    for key in list(state['callbacks']):
        if key.startswith('ModelCheckpoint'):del state['callbacks'][key]
    path=output/'preflight_resume.ckpt';torch.save(state,path)
    return path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--source',type=Path,default=SOURCE)
    p.add_argument('--target',type=int,default=600000);p.add_argument('--prepare',action='store_true')
    p.add_argument('--arm',choices=['laser','rvq']);p.add_argument('--resume',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--smoke',action='store_true');p.add_argument('--mode',choices=['online','disabled'],default='online')
    args=p.parse_args();os.chdir(REPO);args.root=args.root.resolve();args.source=args.source.resolve()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    if args.prepare:prepare(args.root,args.source,args.target);return
    if not args.arm:p.error('--arm required')
    protocol=json.loads((args.root/'protocol.json').read_text())
    args.resume=args.resume or args.root/args.arm/'resume_200k.ckpt'
    args.output=args.output or args.root/(f'preflight_{args.arm}' if args.smoke else args.arm)
    if args.smoke:args.resume=smoke_checkpoint(args.resume,args.output)
    restored=int(torch.load(args.resume,map_location='cpu',weights_only=False)['state_dict']['_manual_train_step'])
    args.updates=restored+12 if args.smoke else 0
    args.workers=args.metric_workers=2 if args.smoke else 8
    args.validation_limit=2 if args.smoke else 0;args.train_batches=2 if args.smoke else 0
    def callbacks(output,manifest,arm):
        return [HardRateStatistics(output=output,gpu_hour_ceiling=protocol['compute_ceiling_gpu_hours_per_arm']),
            GracefulBudget(output),AudioContinuationMedia(output,manifest,arm,reconstruct_fn=reconstruct_hard6k,
                rate_label='hard 6 kbps maximum; longer-context continuation')]
    train(args,model_class=HardRateLASER if args.arm=='laser' else HardRateRVQ,extra_callbacks_factory=callbacks)


if __name__=='__main__':main()
