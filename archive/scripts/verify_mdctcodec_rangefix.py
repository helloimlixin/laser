#!/usr/bin/env python3
"""Verify the range-fix GPU preflights and resume their remote epoch-five snapshots."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
import wandb
from src.mdctcodec_matched import MatchedModel, PairedAudioDataset, reconstruct_serialized
from src.models.laser import LASER


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def optimizer_steps(checkpoint):
    return [sorted({int(item['step']) for item in optimizer['state'].values() if 'step' in item})
            for optimizer in checkpoint['optimizer_states']]


def download_and_launch(root):
    api=wandb.Api()
    verified={}
    streams={arm:rows(root/f'preflight_{arm}'/'data_order.jsonl') for arm in ['laser','rvq']}
    assert streams['laser']==streams['rvq'] and sum(r['batches'] for r in streams['laser'])==1200
    for arm in ['laser','rvq']:
        output=root/f'preflight_{arm}'
        completion=read(output/'completion.json')
        assert completion['status']=='complete' and completion['generator_updates']==1200
        run=read(output/'run.json')
        artifact=api.artifact(f'helloimlixin-rutgers/laser/model-{run["id"]}-selected-checkpoints:epoch-005')
        files=sorted(f.name for f in artifact.files())
        assert artifact.state=='COMMITTED' and len(files)==4 and 'last.ckpt' in files
        assert artifact.metadata['epoch']==5 and artifact.metadata['global_step']==2000
        target=root/'remote_restore'/arm
        downloaded=Path(artifact.get_path('last.ckpt').download(root=str(target)))
        checkpoint=torch.load(downloaded,map_location='cpu',weights_only=False)
        assert int(checkpoint['state_dict']['_manual_train_step'])==1000
        assert optimizer_steps(checkpoint)==[[1000],[1000]]
        entry={'run':run,'epoch5_artifact':artifact.qualified_name,'state':artifact.state,
               'files':files,'downloaded_last':str(downloaded),
               'sha256':hashlib.sha256(downloaded.read_bytes()).hexdigest(),
               'epoch5_optimizer_steps':optimizer_steps(checkpoint),'data_order_exact_match':True,
               'preflight_updates':1200,'preflight_epochs':6,'batches_per_smoke_epoch':200}
        if arm=='laser':
            bounds=rows(output/'coefficient_range.jsonl')
            after=[r for r in bounds if r['generator_updates']>=1000]
            assert after and max(r['clipping_window_mean'] for r in after)<=0.05
            state=checkpoint['callbacks']['TrainingCoefficientRange']
            bound=checkpoint['hyper_parameters']['coefficient_quantization_max']
            assert state['last_bound']==bound and state['last_step']==1000
            initial=torch.load(root/'laser_initial.pt',map_location='cpu',weights_only=False)
            model=MatchedModel(**initial['hyper_parameters'])
            model.on_load_checkpoint(checkpoint)
            model.load_state_dict(checkpoint['state_dict'],strict=True)
            assert model.bottleneck.coefficient_quantization_max==bound
            restored=LASER.load_from_checkpoint(downloaded,map_location='cpu',strict=True).eval()
            model.eval()
            manifest=read(root/'manifest.json')
            x=PairedAudioDataset(manifest['train'])[0][0][None]
            with torch.inference_mode():
                y,payload=reconstruct_serialized(model,x)
                y2,payload2=reconstruct_serialized(restored,x)
            torch.testing.assert_close(y,y2,rtol=0,atol=0)
            assert payload==payload2 and len(payload)%5==0
            assert model.bottleneck.coefficient_quantization_max==restored.bottleneck.coefficient_quantization_max==bound
            entry.update({'max_clipping_window_mean_after_1000':max(r['clipping_window_mean'] for r in after),
                'range_at_1200':bounds[-1],'epoch5_saved_bound':bound,'observer_step':state['last_step'],
                'base_model_and_training_restore_decode_identically':True,
                'inference_keeps_checkpoint_bound':True,'payload_bytes_for_7960_samples':len(payload)})
        verified[arm]=entry
    (root/'preflight_verified.json').write_text(json.dumps(verified,indent=2))
    processes={}
    for gpu,arm in enumerate(['laser','rvq']):
        output=root/f'restore_{arm}'
        if output.exists():raise RuntimeError(f'Restore output already exists: {output}')
        output.mkdir()
        shutil.copyfile(root/f'preflight_{arm}'/'run.json',output/'run.json')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        cmd=[sys.executable,'-u','scripts/train_mdctcodec_matched.py','--root',str(root),
             '--arm',arm,'--output',str(output),'--resume',verified[arm]['downloaded_last'],
             '--mode','disabled','--smoke','--updates','1200','--train-batches','200',
             '--validation-limit','2','--workers','4','--metric-workers','2']
        with (root/f'restore_{arm}.log').open('w') as log:
            process=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL,start_new_session=True)
        processes[arm]={'pid':process.pid,'command':cmd}
    (root/'restore_launch.json').write_text(json.dumps(processes,indent=2))
    print(json.dumps({'verification':verified,'restores':processes},indent=2))


def verify_restores(root):
    verified={}
    for arm in ['laser','rvq']:
        output=root/f'restore_{arm}'
        completion=read(output/'completion.json')
        assert completion['status']=='complete' and completion['generator_updates']==1200
        original=rows(root/f'preflight_{arm}'/'data_order.jsonl')
        restored=rows(output/'data_order.jsonl')
        assert restored==original[-2:]  # Pending epoch-four audit, then epoch five.
        checkpoint=torch.load(output/'checkpoints/final.ckpt',map_location='cpu',weights_only=False)
        assert optimizer_steps(checkpoint)==[[1200],[1200]]
        assert all(torch.isfinite(t).all() for t in checkpoint['state_dict'].values() if t.is_floating_point())
        entry={'remote_epoch5_updates':1000,'restored_updates':1200,
               'optimizer_steps':optimizer_steps(checkpoint),'next_epoch_matches_uninterrupted_run':True,
               'data_order':restored,'all_saved_weights_finite':True}
        if arm=='laser':
            state=checkpoint['callbacks']['TrainingCoefficientRange']
            assert state['last_step']==1200 and len(state['clipping'])==100
            assert state['last_bound']==checkpoint['hyper_parameters']['coefficient_quantization_max']
            remote=torch.load(root/'remote_restore'/arm/'last.ckpt',map_location='cpu',weights_only=False)
            first=rows(output/'coefficient_range.jsonl')[0]
            assert first['generator_updates']==1100
            assert sum(state['clipping'])/100<=0.05
            entry.update({'remote_bound':remote['hyper_parameters']['coefficient_quantization_max'],
                'restored_bound':state['last_bound'],'observer_resumed_through_step':state['last_step'],
                'final_clipping_mean':sum(state['clipping'])/100})
        verified[arm]=entry
    (root/'restore_verified.json').write_text(json.dumps(verified,indent=2))
    print(json.dumps(verified,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('outputs/mdctcodec_matched_6kbps_rangefix'))
    parser.add_argument('--verify-restores',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4)
    if args.verify_restores:verify_restores(args.root)
    else:download_and_launch(args.root)
