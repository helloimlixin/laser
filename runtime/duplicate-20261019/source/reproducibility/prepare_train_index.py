from pathlib import Path
import time,importlib.util,json
b=Path('/tmp/laser-sharedjoint20k-repair-2b300-20261018');d=b/'data'
while not (d/'ILSVRC2012_img_train.verified.json').exists():time.sleep(5)
s=importlib.util.spec_from_file_location('prep','/workspace/Projects/laser/scripts/tools/prepare_original_rq_imagenet_data.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.DATA=d;m.OUT=b/'data-verification'
wnids=m.index_train(d/'ILSVRC2012_img_train.tar')
report=dict(ready=True,training_images=1281167,classes=len(wnids),archive=json.loads((d/'ILSVRC2012_img_train.verified.json').read_text()),time=time.time())
m.record('train-ready.json',report);print(json.dumps(report),flush=True)
