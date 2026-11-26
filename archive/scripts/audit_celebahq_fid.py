"""Quantify sample-count FID bias using disjoint real CelebA-HQ images."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.data.var_images import Images, load_manifests
from src.training.var_laser import get_inception_model, frechet_distance


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(4)
    manifests = load_manifests(args.base/'manifests', 'celebahq')
    inception = get_inception_model().eval().to(args.device)
    features = {}
    for split in ['train', 'val']:
        path = args.base/f'fid-real-{split}-features.npy'
        if path.exists():
            features[split] = np.load(path)
            continue
        dataset = Images(args.base/'data'/split, manifests[split], False, 0, 'celebahq')
        chosen = np.random.default_rng(7919).permutation(len(dataset))[:2000]
        result = []
        for i, (images, _) in enumerate(DataLoader(Subset(dataset, chosen), batch_size=16, num_workers=2)):
            pixels = ((images.to(args.device)+1)*127.5).round().clamp(0,255).byte()
            result.append(inception(pixels.float()/255).cpu().numpy())
            if i % 20 == 0:
                print(f'{split}: {sum(len(x) for x in result)} features', flush=True)
        features[split] = np.concatenate(result)
        np.save(path, features[split])
    reference = features['val'].astype(np.float64)
    mu, sigma = reference.mean(0), np.cov(reference, rowvar=False)
    scores = []
    for count in [128, 512, 2000]:
        # Match the original preview sample count and the repaired count,
        # always against the same held-out reference. No overlapping images.
        value = features['train'][:count].astype(np.float64)
        fid = float(frechet_distance(value.mean(0), np.cov(value, rowvar=False), mu, sigma))
        row = dict(generated_count=count, reference_count=len(reference), real_vs_real_fid=fid)
        scores.append(row)
        print(json.dumps(row), flush=True)
    (args.base/'fid-sample-count-audit.json').write_text(json.dumps(dict(
        source='disjoint real train vs validation images', scores=scores), indent=2)+'\n')


if __name__ == '__main__':
    main()
