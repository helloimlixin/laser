import importlib.util,json,time
from pathlib import Path
root=Path('/workspace/Projects/laser')
spec=importlib.util.spec_from_file_location('prep',root/'scripts/tools/prepare_original_rq_imagenet_data.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.BASE=Path('/tmp/laser-resume-h100-20261105/frozen');m.DATA=m.BASE/'data';m.OUT=root/'outputs/imnet-j16384-best14574-continue100-1b300-20261104/resume-h100-20261105'
m.DATA.mkdir(parents=True,exist_ok=True)
receipt=m.copy_archive(m.SOURCE/'ILSVRC2012_img_train.verified-20261005.tar',m.DATA/'ILSVRC2012_img_train.tar','1d675b47d978889d74fa0da5fadfb00e',workers=24)
classes=m.index_train(m.DATA/'ILSVRC2012_img_train.tar')
m.record('train-ready.json',dict(ready=True,training_images=1281167,classes=len(classes),archive=receipt,time=time.time()))
