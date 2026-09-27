"""Fail the allocation before production unless both stages and resume passed."""
import argparse
import json
import os
from pathlib import Path

import torch


def finite_tree(value, path=''):
    if torch.is_tensor(value) and value.is_floating_point():
        if path.endswith('.attn_bias_for_masking'):
            assert ((value == 0) | torch.isneginf(value)).all(), 'Invalid causal attention mask'
        else:
            assert torch.isfinite(value).all(), f'Nonfinite checkpoint tensor: {path}'
    elif isinstance(value, dict):
        for name, child in value.items():
            finite_tree(child, f'{path}.{name}')
    elif isinstance(value, (tuple, list)):
        for i, child in enumerate(value):
            finite_tree(child, f'{path}.{i}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    args = parser.parse_args()
    if int(os.environ.get('SLURM_PROCID', 0)):
        return
    source = torch.load(args.run/'initial-tokenizer.pt', map_location='cpu', weights_only=False)
    stage1 = torch.load(args.run/'tokenizer-last.pt', map_location='cpu', weights_only=False)
    assert stage1['progress']['step'] == 3
    assert len(stage1['rng']) == 4
    assert stage1['discriminator_optimizer']['state'], 'No discriminator update occurred'
    for key in ['encoder.conv_in.weight', 'decoder.conv_out.weight', 'quantize.dictionary.dictionary']:
        assert not torch.equal(source['model'][key], stage1['model'][key]), f'No learning in {key}'
    finite_tree(stage1)
    del source, stage1
    stage2 = torch.load(args.run/'prior-last.pt', map_location='cpu', weights_only=False)
    assert stage2['progress']['step'] == 4, 'Prior did not resume through a new optimizer update'
    assert len(stage2['rng']) == 4
    assert stage2['model']['class_emb.weight'].shape[0] == 2, 'Incorrect face label vocabulary'
    assert 'coefficient_query.weight' in stage2['model']
    finite_tree(stage2)
    del stage2
    assert (args.run/'generated-epoch000.png').is_file()
    for rank in range(4):
        assert json.loads((args.run/f'tokenizer-roundtrip-rank{rank}.json').read_text())['passed']
    receipt = dict(passed=True, job_id=os.environ.get('SLURM_JOB_ID'), run=str(args.run),
                   tokenizer_updates=3, resumed_prior_updates=4, world_size=4,
                   adversarial_active=True, checkpoints_finite=True, roundtrip_passed=True)
    (args.base/'gpu-smoke-passed.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
