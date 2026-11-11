"""Produce final rFIDs only after all official validation images are covered."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import yaml

ROOT = Path('/workspace/Projects/laser')
sys.path.insert(0, str(ROOT))
from src.rqvae_metrics import frechet_distance


def moments(directory, name, rows):
    count = 0
    total = np.zeros(2048, dtype=np.float64)
    cross = np.zeros((2048, 2048), dtype=np.float64)
    for row in rows:
        values = np.load(directory / f'{name}-rank{row["rank"]}.npy', mmap_mode='r')
        assert values.shape == (row['images'], 2048)
        for first in range(0, len(values), 2048):
            block = np.asarray(values[first:first + 2048], dtype=np.float64)
            assert np.isfinite(block).all()
            count += len(block)
            total += block.sum(0)
            cross += block.T @ block
    assert count == 50000
    return total / count, (cross - np.outer(total, total) / count) / (count - 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    output = Path(config['output_directory'])
    folder = output / 'validation50k'
    rows = [json.loads((folder / f'result-rank{rank}.json').read_text()) for rank in range(8)]
    indices = [index for row in rows for index in row['image_indices']]
    assert len(indices) == 50000 and sorted(indices) == list(range(50000))
    assert all(row['codebook_hashes'] == rows[0]['codebook_hashes'] for row in rows)
    assert all(row['source_cnn_digest'] == rows[0]['source_cnn_digest'] for row in rows)
    assert all(row['frozen_encoder_decoder_verified'] and row['frozen_vector_tables_verified'] and row['integer_lookup_parity_verified'] for row in rows)
    directory = Path(rows[0]['feature_directory'])
    real_mean, real_covariance = moments(directory, 'real', rows)
    results = []
    for vocabulary in config['vocabularies']:
        mean, covariance = moments(directory, f'T{vocabulary}', rows)
        fid = frechet_distance(real_mean, real_covariance, mean, covariance)
        assert np.isfinite(fid) and fid > 0
        result = dict(vocabulary=vocabulary, depth=config['depth'], rfid=fid, images=50000,
                      codebook_sha256=rows[0]['codebook_hashes'][str(vocabulary)])
        results.append(result)
        print(json.dumps(result), flush=True)
    previous = json.loads((ROOT / 'outputs/imagenet-rq24576-tokenizer-20261108/full-validation.json').read_text())
    control = next(row for row in results if row['vocabulary'] == 24576)
    reproduction_error = abs(control['rfid'] - previous['greedy_rfid'])
    # Quantization boundary changes at floating-point roundoff can slightly
    # change rFID across refits; a larger discrepancy needs investigation.
    assert reproduction_error < .01, f'24,576-vector control failed: {reproduction_error}'
    report = dict(passed=True, images=50000, split=config['evaluation']['split'], results=results,
                  continuous_stage1_rfid=4.213210725808494, frozen_encoder_decoder_verified=True,
                  frozen_vector_tables_verified=True, integer_lookup_parity_verified=True,
                  complete_disjoint_image_coverage_verified=True,
                  stage1_checkpoint_sha256=config['stage1_checkpoint_sha256'],
                  source_cnn_digest=rows[0]['source_cnn_digest'],
                  fitting_images=config['fitting_images'], fitting_passes=config['fitting_passes'],
                  vector_gain=config['vector_gain'], validation_used_for_fitting=False,
                  metric='original RQ-VAE TensorFlow-FID-compatible Inception; paired real/reconstruction statistics',
                  encoding=config['encoding'], reference_24576_rfid=previous['greedy_rfid'],
                  control_24576_rfid_difference=reproduction_error, preprocessing=rows[0]['preprocessing'])
    (output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
