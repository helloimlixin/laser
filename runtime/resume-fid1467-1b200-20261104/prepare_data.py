"""Verify and index the original ImageNet training archive on local NVMe."""
import importlib.util
import json
from pathlib import Path
import time

ROOT = Path('/workspace/Projects/laser')
BASE = Path('/tmp/laser-fid1467-b200/frozen')
OUT = ROOT / 'outputs/imnet-j16384-a4096-fid1467-resume100-1b200-20261104'


def main():
    spec = importlib.util.spec_from_file_location(
        'prepare_original', ROOT / 'scripts/tools/prepare_original_rq_imagenet_data.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.BASE, module.DATA, module.OUT = BASE, BASE / 'data', OUT
    module.DATA.mkdir(parents=True, exist_ok=True)
    receipt = module.copy_archive(
        module.SOURCE / 'ILSVRC2012_img_train.verified-20261005.tar',
        module.DATA / 'ILSVRC2012_img_train.tar',
        '1d675b47d978889d74fa0da5fadfb00e', workers=24)
    classes = module.index_train(module.DATA / 'ILSVRC2012_img_train.tar')
    report = dict(ready=True, training_images=1281167, classes=len(classes),
                  archive=receipt, time=time.time())
    module.record('train-ready.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
