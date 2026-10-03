"""Train-only, CPU comparison of finite complete-vector candidate coverage."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import sys
import time

import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from src.training.combination_search import search_combinations, wide_pair_pool
from src.training.expanded_vector_teacher import ExpandedVectorTeacher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', type=Path, default=Path('/tmp/laser-church-vector96-repeat90-20261001'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--images', type=int, default=32)
    parser.add_argument('--sites-per-image', type=int, default=4)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--extension-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    sys.path.insert(0, str(args.runtime))
    from fast_teacher_cached_norm import CachedNormVectorTeacher
    from vector_targets import build_vector_targets
    plan = json.loads((args.runtime/'plan.json').read_text())
    source = torch.load(plan['tokenizer'], map_location='cpu', weights_only=False, mmap=True)
    dictionary = F.normalize(source['state_dict']['quantizer.dictionary'].float(), dim=0)
    scales = torch.tensor(plan['coeff_scales'], dtype=torch.float32)
    grids = torch.linspace(-3, 3, 2048)[None] * scales[:, None]
    aux = SimpleNamespace(dictionary=dictionary, coeff_scales=scales, coeff_bins=torch.linspace(-3, 3, 2048))
    cache = torch.load(plan['token_cache'], map_location='cpu', weights_only=False, mmap=True)
    split = torch.load(plan['split_path'], map_location='cpu', weights_only=False)
    rows = split['train'][:args.images * 2]
    assert not torch.isin(rows, split['validation']).any()
    generator = torch.Generator().manual_seed(2026100141)
    positions = torch.stack([torch.randperm(64, generator=generator)[:args.sites_per_image]
                              for _ in range(len(rows))])
    records = []
    for group, selected_rows, selected_positions in (
            ('fit-train', rows[:args.images], positions[:args.images]),
            ('check-disjoint-train', rows[args.images:], positions[args.images:])):
        a = cache['atoms'][selected_rows].reshape(-1, 64, 4).long()
        c = cache['coeffs'][selected_rows].reshape(-1, 64, 4).float()
        local_rows = torch.arange(len(a))[:, None]
        atoms = a[local_rows, selected_positions].reshape(-1, 4)
        normalized = c[local_rows, selected_positions].reshape(-1, 4)
        physical = normalized * scales
        z = (dictionary.T[atoms] * physical[..., None]).sum(-2)
        print(json.dumps({'phase':group,'sites':len(atoms)}), flush=True)
        started = time.perf_counter()
        legacy, telemetry = CachedNormVectorTeacher(aux, plan)(atoms, normalized,
            generator=torch.Generator().manual_seed(179))
        elapsed = time.perf_counter() - started
        stochastic_atoms = legacy.atoms[:, -4:]
        different = (stochastic_atoms.sort(-1).values != atoms[:, None].sort(-1).values).any(-1)
        baseline = dict(group=group, setting='legacy-four-independent-pools', seconds_cpu=elapsed,
            sites=len(atoms), changed_support_draw_fraction=float(different.float().mean()),
            draw_distortion=float(legacy.errors[:, -4:].mean()),
            context_expected_error=float(telemetry['context_expected_error']))
        records.append(baseline)
        print(json.dumps(baseline), flush=True)
        if args.extension_only:
            teacher = ExpandedVectorTeacher(aux, plan, baseline_class=CachedNormVectorTeacher,
                                            target_builder=build_vector_targets)
            started = time.perf_counter()
            expanded, expanded_telemetry = teacher(atoms, normalized,
                generator=torch.Generator().manual_seed(179))
            elapsed = time.perf_counter()-started
            valid_extra = expanded.valid[:, -4:]
            error_extra = expanded.errors[:, -4:][valid_extra]
            record = dict(group=group, setting='preserved-context-plus-four-novel-support-events',
                sites=len(atoms), seconds_cpu=elapsed,
                baseline_context_bitwise_equal=bool(torch.equal(legacy.context_atoms,expanded.context_atoms)
                    and torch.equal(legacy.context_coefficient_ids,expanded.context_coefficient_ids)),
                baseline_candidates_preserved=bool(torch.equal(legacy.atoms,expanded.atoms[:, :28])
                    and torch.equal(legacy.valid,expanded.valid[:, :28])),
                valid_new_candidates_per_site=float(valid_extra.float().sum(-1).mean()),
                new_candidate_error_median=float(error_extra.median()) if error_extra.numel() else None,
                new_candidate_error_p95=float(torch.quantile(error_extra,.95)) if error_extra.numel() else None,
                new_candidate_mean_kernel_weight=float(torch.exp(-error_extra/plan['vector_kernel_temperature']).mean()) if error_extra.numel() else None,
                **{k:float(v) for k,v in expanded_telemetry.items()})
            records.append(record)
            print(json.dumps(record),flush=True)
            continue
        for alternatives in (2, 4):
            started = time.perf_counter()
            pool = wide_pair_pool(atoms, physical, dictionary, grids,
                temperature=plan['teacher_temperature'], alternatives_per_depth=alternatives,
                generator=torch.Generator().manual_seed(831), site_chunk_size=32)
            proposal_seconds = time.perf_counter()-started
            for width in (16, 32, 64):
                started = time.perf_counter()
                bank = search_combinations(z, dictionary, pool.atoms, pool.coefficient_ids,
                    grids, beam_width=width, site_chunk_size=32)
                search_seconds = time.perf_counter()-started
                probability = bank.probabilities(plan['teacher_temperature'])
                sorted_support = bank.atoms.sort(-1).values
                new_support = (sorted_support != atoms[:, None].sort(-1).values).any(-1) & bank.valid
                same_support = (sorted_support[:, :, None] == sorted_support[:, None]).all(-1)
                earlier = torch.tril(torch.ones(width, width, dtype=torch.bool), diagonal=-1)
                unique = bank.valid & ~(same_support & earlier & bank.valid[:, None]).any(-1)
                draw_a, draw_b = bank.sample(plan['teacher_temperature'],
                    generator=torch.Generator().manual_seed(569))
                draw_changed = (draw_a.sort(-1).values != atoms[:, None].sort(-1).values).any(-1)
                sampled_values = grids[torch.arange(4), draw_b]
                drawn_z = (dictionary.T[draw_a] * sampled_values[..., None]).sum(-2)
                draw_error = (drawn_z-z[:, None]).square().sum(-1)
                record = dict(group=group, setting='wide-pool-two-slot-search',
                    alternative_atoms_per_depth=alternatives, beam_width=width,
                    support_quota=width//2,
                    sites=len(atoms), seconds_cpu=proposal_seconds+search_seconds,
                    proposal_seconds_cpu=proposal_seconds, search_seconds_cpu=search_seconds,
                    mean_unique_supports=float(unique.sum(-1).float().mean()),
                    mean_retained_candidates=float(bank.valid.sum(-1).float().mean()),
                    mean_candidate_evaluations=float(bank.evaluated_candidates.float().mean()),
                    changed_support_probability=float((probability * new_support).sum(-1).mean()),
                    changed_support_draw_fraction=float(draw_changed.float().mean()),
                    draw_distortion=float(draw_error.mean()),
                    expected_distortion=float((probability * bank.errors).sum(-1).mean()),
                    useful_alternative_candidates=float((new_support & (bank.errors <= plan['teacher_temperature'])).sum(-1).float().mean()),
                    source_energy=float(z.square().sum(-1).mean()))
                records.append(record)
                print(json.dumps(record), flush=True)
    report = dict(records=records, device='cpu', threads=args.threads,
        train_rows=rows.tolist(), spatial_positions=positions.tolist(),
        heldout_used=False, fid_used=False, optimizer_steps=0,
        scope='CPU prototype coverage and cost, not a GPU throughput or FID result. Spatial context and student loss kernel are unchanged in the intended candidate-only ablation.',
        module_sha256=hashlib.sha256((PROJECT/'src/training/combination_search.py').read_bytes()).hexdigest(),
        source_assets={k:plan[k] for k in ('tokenizer_sha256','cache_sha256','split_sha256')})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'completed':True,'result':str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
