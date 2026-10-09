"""Summarize completed noise forks while preserving encoder hash caveats."""
import argparse
import json
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trial', type=Path, required=True)
    args = p.parse_args()
    root = args.trial
    plan = json.loads((root/'plan.json').read_text())
    audit = json.loads((root/'matched-input-and-rng-audit.json').read_text())
    assert audit['input_and_rng_matching']
    assert json.loads((root/'config-difference-proof.json').read_text())['passed']
    scores = {}
    checkpoints = {}
    for branch in ('low25', 'control200'):
        completion = json.loads((root/branch/'completed.json').read_text())
        assert completion['passed']
        score = completion['evaluation']
        assert score['train/global_step'] == 5634 and score['train/epoch'] == 9
        assert score['generated_images'] == 50000
        assert score['fid_protocol'] == 'original_rqvae_training_reference'
        scores[branch] = score
        verification = json.loads((root/branch/'final-checkpoint-verification.json').read_text())
        assert verification['passed'] and verification['wandb_md5_matching']
        checkpoints[branch] = verification
    restored = json.loads((root/'parent-restoration-verification.json').read_text())
    assert restored['passed']
    fid_change = scores['low25']['eval/fid_original_train50k']-scores['control200']['eval/fid_original_train50k']
    is_change = scores['low25']['eval/inception_score']-scores['control200']['eval/inception_score']
    result = dict(status='completed_with_encoder_target_variation' if not audit['token_hashes_all_identical']
                  else 'completed', plan=plan, scores=scores,
        candidate_minus_control_fid=fid_change, candidate_minus_control_inception_score=is_change,
        observed_lower_noise_fid_better=fid_change < 0,
        source_state_step=5008, final_step=5634, updates_each=626,
        input_and_rng_audit=audit, recovery_checkpoints=checkpoints,
        parent_restoration=restored, automatic_full_restart=False,
        coefficient_distribution=str(root/'distribution-diagnostics/results.json'),
        sampled_coefficient_diagnostics=str(root/'sampled-coefficient-diagnostics/comparison.json'),
        limitations='One-epoch continuation, one training/sampling seed, no confidence interval. '
            'Fresh OMP target hashes differ on two ranks despite identical image and RNG hashes. '
            'The exact live target difference is unknown; replay demonstrates sensitivity to '
            'floating-point encoder policies. This is not a bitwise paired target experiment '
            'and does not establish eventual scratch-training FID.',
        time=time.time())
    (root/'audited-comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(status=result['status'], scores=scores,
        candidate_minus_control_fid=fid_change,
        candidate_minus_control_inception_score=is_change,
        parent_restored=True)), flush=True)


if __name__ == '__main__':
    main()
