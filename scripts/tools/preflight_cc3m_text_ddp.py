"""One real global-batch update on every production GPU, with consensus checks."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import OmegaConf

from src.training.cc3m_text import build_model, configure_performance, objective, capture_rng, cpu_snapshot
from src.training.cc3m_compound import make_aux
from scripts.tools.build_cc3m_compound_cache import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    options = OmegaConf.to_container(OmegaConf.load(args.config).options, resolve=True)
    rank, local, world = [int(os.environ[x]) for x in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE')]
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    dist.init_process_group('nccl', timeout=timedelta(minutes=20))
    checkpoint_group = dist.new_group(backend='gloo', timeout=timedelta(minutes=20))
    torch.manual_seed(1077)
    data = torch.load(options['validation_cache'], weights_only=True, map_location='cpu', mmap=True)
    aux = make_aux(options, options['coeff_scales'], device)
    prior = build_model(options).to(device)
    configure_performance(prior, options)
    model = DDP(prior, device_ids=[local], broadcast_buffers=False, gradient_as_bucket_view=True)
    optimizer = torch.optim.AdamW(prior.parameters(), lr=options['lr'], betas=(.9, .95), weight_decay=1e-4, fused=True)
    torch.manual_seed(1077 + rank)
    batch, accumulation = options['batch_size'], options['accumulation']
    assert batch * accumulation * world == options['total_batch_size']
    optimizer.zero_grad(set_to_none=True)
    from contextlib import nullcontext
    torch.cuda.reset_peak_memory_stats()
    for micro in range(accumulation):
        begin = (micro * world + rank) * batch
        rows = torch.arange(begin, begin + batch).remainder(len(data['atoms']))
        atoms = data['atoms'][rows].long().to(device)
        coeffs = data['coeffs'][rows].to(device)
        text = data['text_ids'][rows].long().to(device)
        sync = micro == accumulation - 1
        with (nullcontext() if sync else model.no_sync()), torch.autocast('cuda', dtype=torch.bfloat16):
            loss, _ = objective(model, aux, atoms, coeffs, text, options['coeff_target_temperature'], accumulation)
            loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(prior.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    assert torch.isfinite(loss) and torch.isfinite(norm)
    snapshot = cpu_snapshot(dict(model=prior.state_dict(), optimizer=optimizer.state_dict(),
        cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(device)))

    def next_update():
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accumulation):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                next_loss, _ = objective(model, aux, atoms, coeffs, text,
                    options['coeff_target_temperature'], accumulation)
                next_loss.backward()
        torch.nn.utils.clip_grad_norm_(prior.parameters(), 1., error_if_nonfinite=True)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        return next_loss.detach().cpu()

    expected_loss = next_update()
    expected = cpu_snapshot(prior.state_dict())
    prior.load_state_dict(snapshot['model'], strict=True)
    optimizer.load_state_dict(snapshot['optimizer'])
    torch.set_rng_state(snapshot['cpu_rng'])
    torch.cuda.set_rng_state(snapshot['cuda_rng'], device)
    actual_loss = next_update()
    torch.testing.assert_close(expected_loss, actual_loss, rtol=0, atol=0)
    actual = cpu_snapshot(prior.state_dict())
    assert all(torch.equal(expected[key], actual[key]) for key in expected), 'Distributed resume changes the next update'
    del snapshot, expected, actual
    probe = torch.cat([value.flatten()[:8].detach().float() for value in prior.parameters()])
    probes = [torch.empty_like(probe) for _ in range(world)]
    dist.all_gather(probes, probe)
    assert all(torch.equal(probes[0], item) for item in probes)
    rng = capture_rng(device, checkpoint_group)
    report = dict(rank=rank, loss=float(loss.detach()) * accumulation,
        gradient_norm=float(norm), peak_gib=torch.cuda.max_memory_allocated(device)/2**30)
    reports = [None] * world if rank == 0 else None
    dist.gather_object(report, reports, dst=0, group=checkpoint_group)
    if rank == 0:
        write_json(args.output, dict(passed=True, world_size=world, global_batch=options['total_batch_size'],
            batch_size=batch, accumulation=accumulation, parameter_probes_equal=True,
            rng_states=len(rng), ranks=reports, weights_discarded=True, next_ddp_update_bitwise_identical=True))
        print(json.dumps(dict(passed=True, ranks=reports)), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
