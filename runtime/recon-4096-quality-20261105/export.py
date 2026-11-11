"""Export verified reports, compact book and safe-class reconstruction examples."""
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path('/workspace/Projects/laser')
SOURCE=Path('/tmp/laser-j32769-a8192-20261105/frozen')
BASE=Path('/tmp/laser-recon-4096-20261105')
PUBLIC=ROOT/'outputs/imagenet-native-joint-atom-reduction-20261105'
sys.path[:0]=[str(SOURCE/'source'),str(SOURCE)]
os.environ['RQ_UPSTREAM_ROOT']=str(SOURCE/'upstream')
os.environ['TORCH_HOME']=str(BASE/'torch-cache')
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from src import rq_upstream
from src.models.laser_joint_tokenizer import FrozenJointLaserTokenizer,JointAtomCoefficientRQ
from src.training.rq_imagenet_runtime import TarImageNet
from rqvae.utils.config import load_config
from rqvae.img_datasets.transforms import create_transforms


def main():
    report=json.loads((BASE/'full-results.json').read_text())
    selected=report['selected']
    compact=Path(report['results'][selected]['codebook_path'])
    checkpoint=PUBLIC/'compact4096-reconstruction-selected.pt'
    shutil.copyfile(compact,checkpoint)
    for name in ['full-results.json','screen-results.json','refinement-verification.json','validation-ready.json']:
        shutil.copyfile(BASE/name,PUBLIC/name)
    plan=json.loads((PUBLIC/'plan.json').read_text())
    plan.update(status='reconstruction_evaluation_completed',selected_atoms=4096,selected_vocabulary=16385,
        selected_codebook=str(checkpoint),rfid_preserved_within_two_percent=report['rfid_preserved_within_two_percent'],
        transformer_training_started=False,encoder_decoder_frozen=True)
    (PUBLIC/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda',0)
    control=Path(report['results']['control8192']['codebook_path'])
    raw=Path(report['results']['compact4096-gain110']['codebook_path'])
    paths=[control,raw,compact]
    aux=FrozenJointLaserTokenizer(SOURCE/'inputs/stage1-model.pt',control).to(device).eval().requires_grad_(False)
    books=[]
    for p in paths:
        spec=torch.load(p,map_location='cpu',weights_only=True)
        books.append(JointAtomCoefficientRQ(spec['dictionary'],spec['atom_ids'],spec['coefficients']).to(device))
    config=load_config(SOURCE/'inputs/stage2-training-config.yaml')
    data=TarImageNet(BASE/'data','val',create_transforms(config.dataset,split='val'))
    # Respect the saved preference: only these explicitly chosen safe classes.
    classes=[('cheeseburger',933),('teapot',849),('samoyed',258),('bald eagle',22),
             ('red fox',277),('hotpot',926),('wombat',106),('timber wolf',269)]
    indices=[int(np.flatnonzero(data.labels==label)[0]) for _,label in classes]
    images=torch.stack([data[i][0] for i in indices]).to(device)
    with torch.inference_mode():
        latent=aux.encode(images)
        columns=[images.mul(.5).add(.5).clamp(0,1).cpu()]
        for q in books:
            decoded=aux.decode(q.quantize(latent,chunk_size=1024)['quantized'])
            columns.append(decoded.mul(.5).add(.5).clamp(0,1).cpu())
    cell=256;gap=8;header=40;left=130
    canvas=Image.new('RGB',(left+4*(cell+gap)+gap,header+len(classes)*(cell+gap)+gap),'white')
    draw=ImageDraw.Draw(canvas)
    font_path=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    font=ImageFont.truetype(str(font_path),18) if font_path.exists() else ImageFont.load_default()
    for col,title in enumerate(['Original','8,192 atoms','4,096 atoms raw','4,096 selected']):
        draw.text((left+col*(cell+gap),10),title,font=font,fill='black')
    for row,(name,_) in enumerate(classes):
        draw.text((4,header+row*(cell+gap)+110),name,font=font,fill='black')
        for col,values in enumerate(columns):
            pixels=values[row].mul(255).round().byte().permute(1,2,0).numpy()
            canvas.paste(Image.fromarray(pixels),(left+col*(cell+gap),header+row*(cell+gap)))
    canvas.save(PUBLIC/'reconstruction-comparison.png')
    (PUBLIC/'preview-policy.json').write_text(json.dumps(dict(classes=classes,validation_indices=indices,
        excluded=['insects','spiders','other arachnids'],displayed_images=len(classes)),indent=2)+'\n')
    print(json.dumps(dict(selected_codebook=str(checkpoint),preview=str(PUBLIC/'reconstruction-comparison.png'),
        full_results=str(PUBLIC/'full-results.json'))),flush=True)


if __name__=='__main__':main()
