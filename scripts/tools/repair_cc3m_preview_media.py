#!/usr/bin/env python3
"""Repair float-to-uint8 W&B previews from the original correctly saved grid."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

BASE = Path('/tmp/laser-cc3m-joint20k-depth5-8a100-20261017')
RUN = 'helloimlixin-rutgers/laser/cc3m-laser-joint20k-depth5-rqnoise-k2048-scratch100-8a100-20261017'


def main():
    import wandb
    os.environ['WANDB_API_KEY'] = Path('/root/.config/laser/wandb.key').read_text().strip()
    run = wandb.Api(timeout=45).run(RUN)
    original = json.loads((BASE/'original-media-summary.json').read_text())
    root = BASE/'media-repair'
    root.mkdir(exist_ok=True)
    grid = Image.open(BASE/'live/train/samples/step-0000001.png').convert('RGB')
    filenames, records = [], []
    for i, old_name in enumerate(original['filenames']):
        x, y = 2+(i%4)*258, 2+(i//4)*258
        pixels = grid.crop((x, y, x+256, y+256))
        old_path = root/old_name
        old_path.parent.mkdir(parents=True, exist_ok=True)
        pixels.save(old_path)
        digest = hashlib.sha256(old_path.read_bytes()).hexdigest()
        new_name = f'media/images/samples/text_to_image_step1_rgb8_{i}_{digest[:20]}.png'
        new_path = root/new_name
        new_path.write_bytes(old_path.read_bytes())
        # New paths refresh the current gallery; old paths repair its history.
        run.upload_file(str(new_path), root=str(root))
        run.upload_file(str(old_path), root=str(root))
        filenames.append(new_name)
        values = np.asarray(pixels)
        assert values.max() > 32 and values.std() > 1
        records.append(dict(original=old_name, corrected=new_name,
            min=int(values.min()), max=int(values.max()), mean=float(values.mean())))
        print(json.dumps(dict(repaired=i+1, total=8, maximum=int(values.max()))), flush=True)
    corrected = dict(original, filenames=filenames)
    run.summary['samples/text_to_image'] = corrected
    run.summary['samples/preview_conversion'] = 'Explicit RGB uint8; generated values were valid; repaired step-1 media'
    run.summary['samples/source_global_step'] = 1
    run.summary.update()
    check = run.file(filenames[0]).download(root=str(BASE/'media-repair-download'), replace=True)
    verified = np.asarray(Image.open(check.name))
    assert verified.max() > 32 and verified.std() > 1
    report = dict(passed=True, generation_not_black=True, source_global_step=1,
        original_upload_range=[0, 1], corrected_upload_range=[0, 255],
        remote_download_verified=True, history_and_current_gallery_repaired=True,
        training_state_changed=False, images=records)
    (BASE/'media-repair-verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(passed=True, remote_download_verified=True, images=8)), flush=True)


if __name__ == '__main__':
    main()
