#!/usr/bin/env python3
"""Reuse the exact eight examples and native pixels from the original W&B run."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shutil

from PIL import Image
import wandb

SOURCE = 'helloimlixin-rutgers/laser/imagenet-rfid421-stage1-zoom-comparison:v0'
SOURCE_RUN = 'helloimlixin-rutgers/laser/imagenet-rfid421-stage1-zooms-20260924'
MODELS = ['vqgan16','rq4','laser2','laser4']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    root = args.work.parent
    args.work.mkdir(parents=True,exist_ok=True)
    for name in ['pixels','inputs','source-metadata']:
        (args.work/name).mkdir(exist_ok=True)
    artifact = wandb.Api(timeout=90).artifact(SOURCE)
    verification = []

    def copy_verified(name, destination):
        source = root/name
        expected = artifact.manifest.entries[name].digest
        def md5(path):
            return base64.b64encode(hashlib.md5(path.read_bytes()).digest()).decode()
        if not source.exists() or md5(source) != expected:
            source = Path(artifact.get_path(name).download(root=str(root/'original-artifact-cache')))
        assert md5(source) == expected,name
        shutil.copyfile(source,destination)
        assert md5(destination) == expected
        verification.append(dict(source=name,destination=str(destination.relative_to(args.work)),md5_base64=expected))

    copy_verified('figures/fixed-regions.json',args.work/'source-metadata/fixed-regions.json')
    copy_verified('figures/manifest.json',args.work/'source-metadata/zoom-manifest.json')
    copy_verified('source/figures/manifest.json',args.work/'source-metadata/inference-manifest.json')
    regions = json.loads((args.work/'source-metadata/fixed-regions.json').read_text())
    source = json.loads((args.work/'source-metadata/inference-manifest.json').read_text())
    prior = json.loads((root/'paper/selection.json').read_text())
    samples = [s for s in prior['samples'] if s['subset']=='original']
    assert len(samples) == len(regions['images']) == 8
    for sample,original in zip(samples,regions['images']):
        assert sample['image_id'] == original['image_id']
        index = sample['index']
        assert index == original['index']+1
        jpeg = root/'paper/inputs'/sample['source_file']
        assert hashlib.sha256(jpeg.read_bytes()).hexdigest() == sample['jpeg_sha256']
        shutil.copyfile(jpeg,args.work/'inputs'/sample['source_file'])
        for model in ['reference']+MODELS:
            target = args.work/f'pixels/{index:02d}-{model}.png'
            copy_verified(f'figures/pixels/example-{index:02d}-{model}.png',target)
            assert Image.open(target).size == (256,256)
    selection = dict(count=8,new_count=0,classes=8,samples=samples,main_figure_indices=[1,2],
        source_run=SOURCE_RUN,source_artifact=SOURCE,
        selection='Reuse all eight images from the explicitly requested W&B run in their original order. No new selection or inference.',
        region_overrides={str(i['index']+1):i['boxes'][0] for i in regions['images']},
        region_selection='Keep the first (A) reference-selected 64×64 box from the source run, shared across all models. Enlarge it to the same displayed size as the full image.',
        inference_note='Exact native reference and reconstruction PNGs from the source W&B artifact, verified against its server-side digests. No inference was rerun and no reconstruction pixels were changed.',
        archive_md5=prior['archive_md5'],archive_url=prior['archive_url'])
    (args.work/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    for model in MODELS:
        report=dict(model=model,count=8,checkpoint_sha256=source['metrics'][model]['checkpoint_sha256'],
                    source_artifact=SOURCE,inference_performed=False,exact_saved_pixels_reused=True)
        (args.work/f'inference-{model}.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.work/'source-verification.json').write_text(json.dumps(dict(source_artifact=SOURCE,verified_files=len(verification),files=verification),indent=2)+'\n')
    print(json.dumps(dict(examples=8,native_pngs=40,source_artifact=SOURCE,verified_files=len(verification))),flush=True)


if __name__ == '__main__':
    main()
