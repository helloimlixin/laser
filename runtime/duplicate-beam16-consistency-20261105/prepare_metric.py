import torch,hashlib,json
from pathlib import Path
base=Path('/tmp/laser-beam16-duplicate-20261105')
folder=base/'frozen/torch-cache/hub/checkpoints';folder.mkdir(parents=True,exist_ok=True)
name='pt_inception-2015-12-05-6726825d.pth'
p=folder/name
if not p.exists():torch.hub.download_url_to_file('https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/'+name,str(p),hash_prefix='6726825d',progress=False)
print('INCEPTION_WEIGHTS_READY',p.stat().st_size)
