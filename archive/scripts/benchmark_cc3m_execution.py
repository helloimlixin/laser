"""Profile execution-only CC3M controls on a disposable model, without W&B."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import statistics
import time

import torch
import yaml

from src.training.cc3m_joint import (build_model, build_aux, dataset, objective,
    configure_execution, execution_options, attention_options, seed_all, optimizer_for)
from src.training.rq_sdpa_backend import sdpa_attention
from src.training.rq_precision import native_bf16_autocast


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--isolated', action='store_true')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32 = False
    cfg = yaml.safe_load(args.config.read_text())
    device = torch.device('cuda', 0)
    seed_all(cfg['seed'])
    aux, model = build_aux(cfg, device), build_model(cfg, device)
    state = torch.load(Path(cfg['local_checkpoints'])/'best-fid.pt',
        map_location='cpu', weights_only=False, mmap=True)
    model.load_state_dict(state['model'], strict=True)
    del state
    optimizer = optimizer_for(model, cfg)
    data = dataset(cfg, 'validation')
    rows = [data[i] for i in range(16)]
    copies = (args.batch+15)//16
    images = torch.stack([r[0] for r in rows]).repeat(copies,1,1,1)[:args.batch].to(device)
    text = torch.stack([r[1] for r in rows]).repeat(copies,1)[:args.batch].to(device)
    rows = []
    variants = [
        ('baseline', {}),
        ('short-math', dict(short_sequence_math=True)),
        ('larger-attention', dict(attention_batch_limit=8192)),
        ('encoder32', dict(encoder_batch=32)),
        ('encoder64', dict(encoder_batch=64)),
        ('ce8192', dict(ce_chunk_size=8192)),
        ('combined', dict(short_sequence_math=True,encoder_batch=64,ce_chunk_size=8192)),
        ('baseline-repeat', {}),
    ]
    if args.isolated:
        variants = [('baseline', {}),
            ('encoder32', dict(encoder_batch=32)),
            ('encoder64', dict(encoder_batch=64)),
            ('encoder128', dict(encoder_batch=128)),
            ('encoder256', dict(encoder_batch=256)),
            ('combined64', dict(short_sequence_math=True,encoder_batch=64,ce_chunk_size=8192)),
            ('combined128', dict(short_sequence_math=True,encoder_batch=128,ce_chunk_size=8192)),
            ('baseline-repeat', {})]
    with native_bf16_autocast():
        for name, controls in variants:
            configure_execution(controls)
            durations = []
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                with sdpa_attention(model, **attention_options(cfg)):
                    for iteration in range(4):
                        optimizer.zero_grad(set_to_none=True)
                        torch.cuda.synchronize()
                        start = time.monotonic()
                        loss, _ = objective(model,aux,images,text,cfg)
                        loss.backward()
                        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip'],error_if_nonfinite=True)
                        optimizer.step()
                        torch.cuda.synchronize()
                        if iteration:
                            durations.append(time.monotonic()-start)
                    row = dict(name=name,controls=controls,seconds=statistics.median(durations),
                        trials=durations,images_per_second=args.batch/statistics.median(durations),
                        peak_gib=torch.cuda.max_memory_allocated()/2**30,
                        loss=float(loss.detach()),gradient_norm=float(norm),passed=True)
            except torch.cuda.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                row = dict(name=name,controls=controls,passed=False,error='out of memory')
            rows.append(row)
            print(json.dumps(row),flush=True)
            args.output.write_text(json.dumps(dict(batch=args.batch,gpu=torch.cuda.get_device_name(),
                live_training_concurrent=not args.isolated,variants=rows),indent=2)+'\n')
        if args.verify:
            optimizer.zero_grad(set_to_none=True)
            model.eval()
            verification = {}
            with torch.no_grad(), torch.autocast('cuda',enabled=False):
                validation = torch.stack([data[i][0] for i in range(64)]).to(device)
                reference = torch.cat([aux.encode(x.float()) for x in validation.split(8)])
                for batch in [32,64,128,256]:
                    actual = torch.cat([aux.encode(x.float()) for x in validation.split(batch)])
                    torch.testing.assert_close(actual,reference,rtol=2e-4,atol=2e-5)
                    verification[str(batch)] = dict(max_absolute_error=float((actual-reference).abs().max()),
                        relative_l2_error=float((actual-reference).norm()/reference.norm()),passed=True)
                    if batch == 64:
                        reference_targets, reference_codes = aux.quantizer.get_soft_codes(reference[:16],temp=cfg['temperatures'],stochastic=False)
                        actual_targets, actual_codes = aux.quantizer.get_soft_codes(actual[:16],temp=cfg['temperatures'],stochastic=False)
                        torch.testing.assert_close(actual_targets,reference_targets,rtol=1e-3,atol=1e-5)
                        # Near-tied argmax choices can differ despite matching
                        # probabilities. Training samples the full soft targets.
                        mean_l1 = float((actual_targets-reference_targets).abs().sum(-1).mean())
                        assert mean_l1 < 1e-4
                        verification['teacher'] = dict(
                            greedy_code_mismatch_fraction=float((actual_codes!=reference_codes).float().mean()),
                            max_probability_error=float((actual_targets-reference_targets).abs().max()),
                            mean_target_l1=mean_l1,passed=True)
                        del reference_targets,actual_targets,reference_codes,actual_codes
            args.output.with_suffix('.verification.json').write_text(json.dumps(verification,indent=2)+'\n')
            print('ENCODER_VERIFICATION',json.dumps(verification),flush=True)
        if args.profile:
            configure_execution({})
            optimizer.zero_grad(set_to_none=True)
            with sdpa_attention(model, **attention_options(cfg)), torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                loss, _ = objective(model,aux,images,text,cfg)
                loss.backward()
                optimizer.step()
                torch.cuda.synchronize()
            args.output.with_suffix('.profile.txt').write_text(
                prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=45))


if __name__ == '__main__':
    main()
