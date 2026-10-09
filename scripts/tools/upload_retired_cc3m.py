"""Drain a retired run's preserved full checkpoints without occupying GPUs."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import time

from omegaconf import OmegaConf
import torch

from src.training.cc3m_text import (upload_checkpoint_file, verify_checkpoint_progress,
                                  retryable_upload_error)
from src.training.checkpoint_upload_queue import CheckpointFile, CheckpointUploadQueue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--local', type=Path, required=True)
    parser.add_argument('--api-key-file', type=Path, required=True)
    args = parser.parse_args()
    os.environ['WANDB_API_KEY'] = args.api_key_file.read_text().strip()
    os.environ['WANDB_DIR'] = str(args.local)
    os.environ['WANDB_CACHE_DIR'] = str(args.local/'wandb-cache')
    os.environ['WANDB_DATA_DIR'] = str(args.local/'wandb-data')
    options = OmegaConf.to_container(OmegaConf.load(args.base/'recipe.yaml').options, resolve=True)
    groups = {}
    metrics = {}
    for slot in ['best-fid.pt', 'best-clip.pt', 'last.pt']:
        path = args.local/'checkpoints/return-to-original-preserved'/slot
        state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        verify_checkpoint_progress(state, options['train_items']//options['total_batch_size'],
                                   options['accumulation'], 8)
        with path.open('rb') as source:
            md5 = base64.b64encode(hashlib.file_digest(source,'md5').digest()).decode()
        if md5 not in groups:
            groups[md5] = [path, state['global_step'], state['epoch'],path.stat().st_size,md5,[]]
        groups[md5][-1].append(slot)
        if state.get('metrics') is not None:
            metrics[slot] = state['metrics']
        del state
    import wandb
    wb = wandb.init(entity=options['wandb_entity'], project=options['wandb_project'],
        id=options['wandb_id'], resume='allow', mode='online', config=options, allow_val_change=True)
    wb.summary.update({'pipeline/phase':'superseded_by_original_best_resume',
        'pipeline/stop_reason':'user chose original best-FID continuation',
        'pipeline/successor_run':'cc3m-rfid421-physicalpairs-650m-8h100-20261004'})
    if 'best-fid.pt' in metrics:
        wb.summary['best/fid'] = metrics['best-fid.pt']['fid']
    if 'best-clip.pt' in metrics:
        wb.summary['best/clip_score'] = metrics['best-clip.pt']['clip_score']
    uploader = CheckpointUploadQueue(lambda item, slots:
        upload_checkpoint_file(item,slots,options,wb),retry_delay=30,
        retryable=retryable_upload_error)
    try:
        for values in sorted(groups.values(),key=lambda x:x[1]):
            uploader.submit(CheckpointFile(*values[:-1],tuple(values[-1])))
        uploader.close()
        report = dict(timestamp=time.time(),retired_run=options['wandb_id'],
            online_verified=True,checkpoints=[dict(step=x[1],epoch=x[2],bytes=x[3],md5=x[4],slots=x[5])
                for x in groups.values()])
        record = args.base/'return-to-original-final-upload.json'
        record.write_text(json.dumps(report,indent=2)+'\n')
        wb.save(str(record),base_path=str(args.base),policy='now')
        print(json.dumps(report),flush=True)
    finally:
        wb.finish()


if __name__ == '__main__':
    main()
