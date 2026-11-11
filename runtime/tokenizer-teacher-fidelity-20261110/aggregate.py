"""Aggregate exact paired populations and original-RQ Inception features."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path('/workspace/Projects/laser')
sys.path.insert(0, str(ROOT))
from src.rqvae_metrics import frechet_distance


def moments(directory, name, records):
    total = np.zeros(2048, dtype=np.float64)
    cross = np.zeros((2048, 2048), dtype=np.float64)
    count = 0
    for row in records:
        features = np.load(directory / f'{name}-rank{row["rank"]}.npy', mmap_mode='r')
        assert features.shape == (row['images'], 2048)
        for first in range(0, len(features), 2048):
            block = np.asarray(features[first:first + 2048], dtype=np.float64)
            assert np.isfinite(block).all()
            total += block.sum(0)
            cross += block.T @ block
            count += len(block)
    return total / count, (cross - np.outer(total, total) / count) / (count - 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', default='full50k')
    parser.add_argument('--world', type=int, default=8)
    parser.add_argument('--images', type=int, default=50000)
    args = parser.parse_args()
    output = ROOT / 'outputs/tokenizer-teacher-fidelity-20261110' / args.tag
    rows = [json.loads((output / f'result-rank{rank}.json').read_text()) for rank in range(args.world)]
    indices = [index for row in rows for index in row['image_indices']]
    assert sorted(indices) == list(range(args.images)) and len(set(indices)) == args.images
    assert all(row['variants'] == rows[0]['variants'] for row in rows)
    assert len({row['native_codebook_sha256'] for row in rows}) == 1
    assert len({row['original_checkpoint_sha256'] for row in rows}) == 1
    directory = Path(rows[0]['feature_directory'])
    real_mean, real_cov = moments(directory, 'real', rows)
    result = {}
    started = time.monotonic()
    for name, (kind, temperature) in rows[0]['variants'].items():
        mean, covariance = moments(directory, name, rows)
        fid = frechet_distance(real_mean, real_cov, mean, covariance)
        stats = np.sum([row['statistics'][name] for row in rows], 0)
        result[name] = dict(kind=kind, temperature=temperature, rfid=fid,
                            latent_mse=float(stats[0] / stats[1]),
                            entropy_by_depth=(stats[2:6] / (args.images * 64)).tolist(),
                            token_flip_fraction_vs_greedy_by_depth=(stats[6:10] / (args.images * 64)).tolist())
        result[name]['mean_entropy'] = float(np.mean(result[name]['entropy_by_depth']))
        greedy_name = 'laser_tau0' if kind == 'laser' else 'rq_original_greedy'
        if greedy_name in result:
            result[name]['latent_mse_ratio_to_greedy'] = result[name]['latent_mse'] / result[greedy_name]['latent_mse']
            result[name]['rfid_increase_over_greedy'] = fid - result[greedy_name]['rfid']
        print(json.dumps(dict(name=name, **result[name])), flush=True)
    report = dict(images=args.images, split='official ImageNet2012 validation',
                  metric='original RQ-VAE TensorFlow-FID-compatible Inception; paired real/reconstructed populations',
                  validation_used_for_fitting=False,
                  current_codebook_sha256=rows[0]['native_codebook_sha256'],
                  original_rq_stage1_sha256=rows[0]['original_checkpoint_sha256'],
                  original_rq_source_url='https://twg.kakaocdn.net/brainrepo/models/RQVAE/7518a004fe39120fcffbba76005dc6c3/imagenet_480M.tar.gz',
                  variants=result, encoder_decoder_frozen=True, training_state_modified=False,
                  sampling_seeds_by_rank=[row['seeds'] for row in rows],
                  seconds_for_aggregation=time.monotonic() - started)
    (output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
