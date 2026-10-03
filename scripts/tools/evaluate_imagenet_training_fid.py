"""Evaluate the epoch-45 baseline and current prior against all training images."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import time

import torch
import torch.distributed as dist
import yaml

ROOT = Path(os.environ.get('LASER_RUNTIME_ROOT', Path(__file__).resolve().parents[2]))
sys.path[:0] = [str(ROOT), str(ROOT / 'runtime')]
from src.training import rqtransformer as training
from src.training.fid_reference import load_torchmetrics_reference
import torchmetrics.image.fid as fid_module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--current', type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ['RANK'])
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    dist.init_process_group('nccl', timeout=timedelta(minutes=40))
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    options = yaml.safe_load(args.config.read_text())['options']
    reference = load_torchmetrics_reference(args.reference, expected_samples=1281167)
    args.output.mkdir(parents=True, exist_ok=True)
    label = None
    original_metric = fid_module.FrechetInceptionDistance
    class CapturedFID(original_metric):
        def compute(self):
            # TorchMetrics synchronizes these states before invoking compute.
            assert int(self.real_features_num_samples) == 1281167
            assert int(self.fake_features_num_samples) == 50000
            result = super().compute()
            if rank == 0:
                torch.save({key:getattr(self,key).cpu() for key in (
                    'fake_features_sum','fake_features_cov_sum','fake_features_num_samples')},
                    args.output/(label+'-generated-moments.pt'))
                print(json.dumps({'phase':'fid_counts_verified','checkpoint':label,
                    'real_images':int(self.real_features_num_samples),
                    'generated_images':int(self.fake_features_num_samples)}),flush=True)
            return result
    fid_module.FrechetInceptionDistance = CapturedFID
    model = training.build_model(18432, 16384, coeff_vocab_size=2048,
        sparsity_level=4, physical_pair_context=True, model_preset='imagenet-1400m')
    for block in model.head_transformer.blocks:
        block.attn.short_attention_backend = 'compiled'
    aux = training.LaserAux(Path(options['checkpoint']),16384,2048,3.,6.4,
        attn_resolutions=(8,),coeff_scales=options['coeff_scales'],
        soft_target_physical=False,clamp_coeffs=False,sparsity_level=4).to(device).eval()
    results = []
    for label,path in (('epoch45',args.baseline),('step30880',args.current)):
        payload = torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        model.load_state_dict(payload['state_dict'],strict=True)
        model.to(device).eval()
        torch.manual_seed(options['seed']+rank)
        torch.cuda.manual_seed(options['seed']+rank)
        started=time.monotonic()
        fid,score,score_std=training.evaluate_generation_metrics(model,aux,None,50000,512,
            num_condition_classes=1000,atom_temperature=options['atom_temperature'],
            atom_top_k=options['atom_top_k'],atom_top_p=options['atom_top_p'],
            coeff_temperature=options['coeff_temperature'],coeff_top_k=options['coeff_top_k'],
            coeff_top_p=options['coeff_top_p'],compute_inception_score=True,
            metric_backend='torchmetrics',fid_reference_stats=args.reference)
        result=dict(checkpoint=label,source=str(path),epoch=payload['epoch'],
            checkpoint_batch_idx=payload.get('batch_idx',0),
            progress_epoch=payload['epoch']+payload.get('batch_idx',0)/1252,
            global_step=payload['global_step'],fid=float(fid),inception_score=float(score),
            inception_score_std=float(score_std),real_images=1281167,
            generated_images=50000,real_split='train',metric_backend='torchmetrics',
            reference=str(args.reference),reference_network_sha256=reference['metadata']['inception_state_sha256'],
            generation_seed=options['seed'],sampling_recipe='atoms T0.9 k0 p0.9; coeff T1 k0 p0.85',
            elapsed_seconds=time.monotonic()-started,evaluated_unix=time.time())
        if rank==0:
            (args.output/(label+'-evaluation.json')).write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(dict(phase='full_training_fid_evaluation',**result)),flush=True)
        results.append(result)
        del payload
    if rank==0:(args.output/'evaluations.json').write_text(json.dumps(results,indent=2)+'\n')
    dist.barrier()
    dist.destroy_process_group()


if __name__=='__main__':main()
