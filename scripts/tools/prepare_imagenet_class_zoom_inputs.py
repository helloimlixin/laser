#!/usr/bin/env python3
"""Select four official validation examples each of Samoyed and cheeseburger."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image
from scipy.io import loadmat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--devkit', type=Path, default=Path('/workspace/Projects/data/imagenet/archives/ILSVRC2012_devkit_t12.tar.gz'))
    parser.add_argument('--archive', type=Path, default=Path('/mnt/laser-imagenet-zooms/imagenet-val-complete.tar'))
    args = parser.parse_args()
    with tarfile.open(args.devkit) as archive:
        members = archive.getmembers()
        ground_truth = next(m for m in members if m.name.endswith('ILSVRC2012_validation_ground_truth.txt'))
        labels = np.array([int(x) for x in archive.extractfile(ground_truth).read().decode().split()])
        meta = next(m for m in members if m.name.endswith('meta.mat'))
        synsets = loadmat(io.BytesIO(archive.extractfile(meta).read()), squeeze_me=True)['synsets']
        classes = {int(s['ILSVRC2012_ID']): (str(s['WNID']), str(s['words'])) for s in synsets}
    expected = [('n02111889', 'Samoyed'), ('n07697313', 'cheeseburger')]
    permutation = np.random.default_rng(20260924).permutation(len(labels)) + 1
    samples = []
    for synset, name in expected:
        matches = [idx for idx, (wnid, words) in classes.items() if wnid == synset]
        assert len(matches) == 1 and name.lower() in classes[matches[0]][1].lower()
        label = matches[0]
        assert np.count_nonzero(labels == label) == 50
        ids = [int(idx) for idx in permutation if labels[idx-1] == label][:4]
        for idx in ids:
            filename = f'ILSVRC2012_val_{idx:08d}.JPEG'
            samples.append(dict(index=41+len(samples), image_id=filename[:-5], source_file=filename,
                                synset=synset, class_name=classes[label][1], subset='requested'))
    args.work.mkdir(parents=True, exist_ok=True)
    (args.work/'inputs').mkdir(exist_ok=True)
    selection = dict(seed=20260924, count=8, new_count=8, classes=2, samples_per_class=4,
                     selection='For each requested class, take the first four matching images in a seeded permutation of all 50,000 official validation IDs. Selection is fixed before inference; no model metric or reconstruction is used. Indices 41–48 continue the previous 40-example gallery.',
                     main_figure_indices=[41,45], samples=samples,
                     region_overrides={'41':[169,153,233,217],'42':[184,120,248,184],
                                       '43':[116,65,180,129],'44':[47,83,111,147]},
                     region_selection='Samoyed crops manually locate the faces in the reference photos; cheeseburger crops use the earlier reference-only central-detail gradient rule. One identical 64×64 region is shared across all models, enlarged to the full-image display size. No model-quality ranking is used.',
                     parent_gallery='https://wandb.ai/helloimlixin-rutgers/laser/runs/imagenet-rfid421-paper40-20260924')
    manifest = args.work/'selection.json'
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        assert [(s['index'],s['image_id']) for s in previous['samples']] == [(s['index'],s['image_id']) for s in samples]
    manifest.write_text(json.dumps(selection,indent=2)+'\n')
    print(json.dumps({'frozen_selection':[(s['index'],s['image_id'],s['class_name']) for s in samples]}),flush=True)
    with args.archive.open('rb') as f:
        digest = hashlib.file_digest(f,'md5').hexdigest()
    assert digest == '29b22e2961454d5413ddabcf34fc5622',digest
    with tarfile.open(args.archive,'r:') as archive:
        members = {Path(m.name).name:m for m in archive.getmembers() if m.isfile()}
        assert len(members) == 50000
        for sample in samples:
            data = archive.extractfile(members[sample['source_file']]).read()
            Image.open(io.BytesIO(data)).load()
            (args.work/'inputs'/sample['source_file']).write_bytes(data)
            sample['jpeg_sha256'] = hashlib.sha256(data).hexdigest()
    selection.update(archive_md5=digest,archive_url='https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar')
    manifest.write_text(json.dumps(selection,indent=2)+'\n')
    print('Extracted eight verified validation JPEGs.',flush=True)


if __name__ == '__main__':
    main()
