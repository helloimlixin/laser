import importlib.util,json,time
from pathlib import Path
p=Path('/workspace/Projects/laser/scripts/tools/prepare_original_rq_imagenet_data.py')
spec=importlib.util.spec_from_file_location('prep',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.DATA=Path('/tmp/laser-beam16-a100-20261105/frozen/data');m.OUT=Path('/tmp/laser-beam16-a100-20261105/data-status');m.DATA.mkdir(parents=True,exist_ok=True)
target=m.DATA/'ILSVRC2012_img_train.tar'
receipt=m.copy_archive(m.SOURCE/'ILSVRC2012_img_train.verified-20261005.tar',target,'1d675b47d978889d74fa0da5fadfb00e',workers=32)
if not (m.DATA/'train-index.npz').exists():
 m.index_train(target)
m.record('train-ready.json',dict(ready=True,training_images=1281167,classes=1000,archive=receipt,time=time.time()))
print('READY',flush=True)
