"""Paired likelihood analysis and measured FID summary for the four cases."""
import json
from pathlib import Path

import numpy as np

ROOT = Path('/workspace/Projects/laser/outputs/imagenet-laser-pair-ablation-20261015')


def main():
    pair = 'pair-raw' if (ROOT/'pair-raw-soft/result.json').exists() else 'pair'
    hard, soft = pair+'-hard', pair+'-soft'
    names = ['scalar-hard', 'scalar-soft', hard, soft]
    results = {name: json.loads((ROOT/name/'result.json').read_text()) for name in names}
    assert all(r['global_step'] == 3884 and r['parent_step'] == 3756 for r in results.values())
    assert all(r['same_data_order_sha256_by_rank'] == results['scalar-hard']['same_data_order_sha256_by_rank']
               for r in results.values())
    rows = {name: json.loads((ROOT/name/'heldout-step3884.json').read_text())['paired_rows'] for name in names}
    reference_indices = [r['index'] for r in rows['scalar-hard']]
    assert len(reference_indices) == 512 and len(set(reference_indices)) == 512
    assert all([r['index'] for r in case] == reference_indices for case in rows.values())
    assert abs(results['scalar-hard']['before']['ce'] - results['scalar-soft']['before']['ce']) < 1.e-5
    assert abs(results[hard]['before']['ce'] - results[soft]['before']['ce']) < 1.e-5
    values = {name: np.array([r['ce'] for r in case]) for name, case in rows.items()}
    generator = np.random.default_rng(261015)
    bootstrap = generator.integers(0, 512, size=(4000, 512))
    contrasts = {}
    for label, left, right in [('decoder_hard', hard, 'scalar-hard'),
                               ('decoder_soft', soft, 'scalar-soft'),
                               ('coefficient_policy_scalar', 'scalar-soft', 'scalar-hard'),
                               ('coefficient_policy_pair', soft, hard),
                               ('combined', soft, 'scalar-hard')]:
        difference = values[left] - values[right]
        interval = np.quantile(difference[bootstrap].mean(-1), [.025, .975])
        contrasts[label] = dict(candidate=left, control=right,
            mean_ce_difference=float(difference.mean()),
            paired_image_bootstrap_95_percent_interval=interval.tolist(), lower_is_better=True)
    summary = dict(passed=True, heldout_images=512, source_step=3756, tested_step=3884,
        additional_updates_per_case=128, global_batch=2048,
        curves={name:dict(initial_ce=r['before']['ce'], final_ce=r['after']['ce'],
                         support_nll=r['after']['support_nll'], coefficient_nll=r['after']['coefficient_nll'])
                for name,r in results.items()}, contrasts=contrasts,
        limitations=['Short architecture continuations do not compare training each architecture from scratch.',
                     'The pair model has 1.458B parameters versus 1.392B for scalar; capacity changes with the decoder.',
                     'The tested pair model omits the additional global coefficient-history decoder used by the historical joint/orthogonal runs.',
                     'The pair transfer preserves the scalar spatial projection (project after summing); historical RQ pair models project each pair before summing, which differs for a nonzero projection bias.',
                     'All four cases use identical epoch-6 sample order and epoch-6 training crops; ExactGlobalBatchSampler.set_epoch forwards the epoch to the dataset.',
                     'Coefficient policy changes both soft labels and sampled coefficient contexts, keeping supports fixed.',
                     'Likelihood uncertainty intervals do not quantify FID uncertainty.'])
    if (ROOT/'comparison.json').exists():
        summary['fid_comparison'] = json.loads((ROOT/'comparison.json').read_text())
        fids = summary['fid_comparison']['fid']
        summary['fid_point_differences'] = {
            label:dict(candidate=item['candidate'],control=item['control'],
                       candidate_minus_control=fids[item['candidate']]-fids[item['control']],
                       lower_is_better=True)
            for label,item in contrasts.items()}
        summary['fid_decoder_policy_interaction_point_difference'] = (
            fids[soft]-fids[hard]-(fids['scalar-soft']-fids['scalar-hard']))
        summary['limitations'].append('FID differences and the decoder-policy interaction are single-seed point estimates, with no FID confidence interval.')
    (ROOT/'paired-analysis.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['curves','limitations']},indent=2))


if __name__ == '__main__':
    main()
