#!/usr/bin/env python3
"""Verify full-size uninterrupted versus resumed training before production."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,required=True)
    args = p.parse_args()
    root = args.run_root.resolve()
    sys.path[:0] = [str(root/'runtime/upstream'),str(root/'runtime')]
    torch.set_num_threads(4)
    statuses = [json.loads((root/name/'status.json').read_text()) for name in ['preflight','preflight-resume']]
    assert all(x['phase']=='preflight_complete' and x['optimizer_step']==3 for x in statuses)
    assert statuses[0]['final_weights_sha256'] == statuses[1]['final_weights_sha256']
    a,b = [torch.load(root/name/'last.pt',map_location='cpu',weights_only=False,mmap=True)
           for name in ['preflight','preflight-resume']]
    tensors = 0
    def identical(left,right,path='checkpoint'):
        nonlocal tensors
        assert type(left) is type(right),path
        if isinstance(left,torch.Tensor):
            assert left.dtype == right.dtype and torch.equal(left,right),path
            tensors += 1
        elif isinstance(left,np.ndarray):
            assert np.array_equal(left,right),path
        elif isinstance(left,dict):
            assert left.keys() == right.keys(),path
            for key in left:
                identical(left[key],right[key],path+'.'+str(key))
        elif isinstance(left,(tuple,list)):
            assert len(left) == len(right),path
            for i,(x,y) in enumerate(zip(left,right)):
                identical(x,y,path+f'[{i}]')
        else:
            assert left == right,path
    identical(a,b)
    assert a['step'] == a['attempts'] == 3 and a['skipped_amp_updates'] == 0
    assert a['initial_weights_sha256'] != statuses[0]['final_weights_sha256']
    assert a['scheduler']['after']['T_max'] == 18600
    assert abs(a['optimizer']['param_groups'][0]['lr']-.0005*(1+np.cos(np.pi*3/18600))/2) < 1e-15
    for name in ['preflight','preflight-resume']:
        val = json.loads((root/name/'validation-preflight.json').read_text())
        assert val['images'] == 300 and all(np.isfinite(x) for x in val.values())
    preview = json.loads((root/'preflight/preview-latest.json').read_text())
    assert preview['samples']==100 and preview['rows']==preview['columns']==10 and preview['training_rng_preserved']
    with Image.open(preview['path']) as im:
        assert im.size == (2582,2582) and im.mode == 'RGB'
        im.verify()
    manifest = root/'runtime-manifest.json'
    for name,digest in json.loads(manifest.read_text()).items():
        assert hashlib.sha256((root/'runtime'/name).read_bytes()).hexdigest() == digest,name
    report = dict(passed=True,resumed_step3_matches_uninterrupted=True,
        model_optimizer_scheduler_scaler_rng_and_cursor_identical=True,tensors_compared=tensors,
        initial_weights_sha256=a['initial_weights_sha256'],step3_weights_sha256=statuses[0]['final_weights_sha256'],
        uninterrupted=statuses[0],resumed=statuses[1],preview_100_verified=True,
        validation_images=300,finite_inception_features_verified_at_startup=True,
        runtime_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        verified_unix=time.time())
    (root/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
