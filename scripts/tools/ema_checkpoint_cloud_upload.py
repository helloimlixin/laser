"""Publish full raw/EMA recovery states with matching verified metric sidecars."""
import json
from pathlib import Path

import torch

from src.training.full_resume_upload import recovery_metadata
from verified_wandb_checkpoint_upload import VerifiedCloudUpload


class EMACheckpointCloudUpload:
    def __init__(self, run_path, receipt):
        self.receipt = Path(receipt)
        self.upload = VerifiedCloudUpload(run_path, self.receipt)
        self.metadata_upload = VerifiedCloudUpload(run_path,
            self.receipt.with_name(self.receipt.stem + '-metadata.json'))

    def __call__(self, paths, epoch):
        self.upload(paths, epoch)
        verified = {item['name']: item for item in json.loads(self.receipt.read_text())['files']}
        sidecars = []
        for path in paths:
            raw = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
            metadata = recovery_metadata(raw)
            selected = 'ema' if 'best-ema-' in path.name else 'raw'
            metrics = raw.get('ema_original_rqtransformer_metrics') if selected == 'ema' else raw.get('original_rqtransformer_metrics')
            if selected == 'ema' and raw.get('parameter_ema') is None:
                # At initialization EMA is an exact copy of the pinned epoch77 raw model.
                metrics = dict(raw['original_rqtransformer_metrics'], weight_state='ema',
                               ema_updates=0, initialized_from_raw_exactly=True)
            metadata.update(checkpoint_file=path.name, checkpoint_bytes=verified[path.name]['bytes'],
                            checkpoint_md5=verified[path.name]['md5'],
                            selected_inference_weights=selected, selected_official_metrics=metrics)
            sidecar = path.with_suffix('.json')
            sidecar.write_text(json.dumps(metadata, indent=2, default=str) + '\n')
            sidecars.append(sidecar)
            del raw
        self.metadata_upload(sidecars, epoch)
