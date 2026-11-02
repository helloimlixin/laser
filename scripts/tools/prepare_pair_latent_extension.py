"""Extend the disjoint ImageNet pair pilot using indexed, frozen encoder latents."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import sys


def extract(args):
    classes=json.loads(args.classes.read_text())
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'images').mkdir(exist_ok=True)
    fd=os.open(args.archive,os.O_RDONLY)
    def one(item):
        label,(name,offset,size)=item;position=offset;index=0;records=[]
        while position<offset+size:
            header=os.pread(fd,512,position)
            if not header.strip(b'\0'):break
            filename=header[:100].split(b'\0')[0].decode()
            length=int(header[124:136].rstrip(b'\0 ').lstrip(b' ') or b'0',8)
            if filename.endswith('.JPEG') and header[156:157] in (b'0',b'\0'):
                if index!=3:
                    existing=args.original_images/filename
                    target=existing if existing.exists() else args.output/'images'/filename
                    if not target.exists():
                        image=os.pread(fd,length,position+512)
                        if len(image)!=length:raise EOFError(filename)
                        target.write_bytes(image)
                    records.append(dict(filename=filename,path=str(target),label=label))
                index+=1
                if index==args.per_class+1:return records
            position+=512+((length+511)//512)*512
        raise RuntimeError('not enough images in '+name)
    rows=[]
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
            for label,items in enumerate(pool.map(one,enumerate(classes))):
                rows.extend(items)
                if label%100==0:print('extracted classes',label,flush=True)
    finally:os.close(fd)
    (args.output/'manifest.json').write_text(json.dumps(rows))
    print('images',len(rows),flush=True)


def encode(args):
    sys.path[:0]=[str(args.teacher_source),str(args.teacher_source/'runtime')]
    import torch
    from PIL import Image
    from torch.utils.data import Dataset,DataLoader
    from src.training.orthogonal_gmm_runtime import OrthogonalGMMAux
    from src.training.imagenet_compound_runtime import SCALES
    from src.training.rqtransformer import val_image_transform
    torch.set_num_threads(8);torch.manual_seed(261029)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rows=json.loads((args.output/'manifest.json').read_text())
    indices=list(range(args.rank,len(rows),args.world))
    class Images(Dataset):
        def __init__(self):self.transform=val_image_transform()
        def __len__(self):return len(indices)
        def __getitem__(self,index):
            row=rows[indices[index]]
            with Image.open(row['path']) as image:return self.transform(image.convert('RGB'))
    teacher=OrthogonalGMMAux(args.teacher,16384,512,3.,coeff_scales=SCALES,sparsity_level=4).cuda().eval()
    latents=[]
    with torch.no_grad():
        for step,images in enumerate(DataLoader(Images(),batch_size=32,num_workers=8,pin_memory=True)):
            latent=teacher.quant_conv(teacher.encoder(images.cuda(non_blocking=True))).permute(0,2,3,1).float()
            latents.append(latent.cpu())
            if step%50==0:print('encoded',args.rank,step*32+len(images),flush=True)
    torch.save(dict(latent=torch.cat(latents),indices=torch.tensor(indices),dictionary=teacher.dictionary.cpu()),
        args.output/f'shard-{args.rank}.pt')
    print('encoded shard',args.rank,len(indices),flush=True)


def merge(args):
    import torch
    old=torch.load(args.original_cache,map_location='cpu',mmap=True,weights_only=False)
    rows=json.loads((args.output/'manifest.json').read_text())
    latent=torch.empty(len(rows),8,8,256)
    covered=torch.zeros(len(rows),dtype=torch.bool)
    for rank in range(args.world):
        shard=torch.load(args.output/f'shard-{rank}.pt',map_location='cpu',mmap=True,weights_only=True)
        torch.testing.assert_close(shard['dictionary'],old['dictionary'],atol=0,rtol=0)
        latent[shard['indices']]=shard['latent'];covered[shard['indices']]=True
    assert covered.all()
    filenames=[row['filename'] for row in rows]
    assert len(filenames)==len(set(filenames)) and not set(filenames)&set(old['heldout']['filenames'])
    lookup={name:i for i,name in enumerate(filenames)}
    # Re-encoding old training images verifies the same exact input pipeline.
    previous=torch.tensor([lookup[name] for name in old['train']['filenames'][:64]])
    delta=float((latent[previous]-old['train']['latent'][:64]).abs().max())
    torch.testing.assert_close(latent[previous],old['train']['latent'][:64],atol=3e-5,rtol=3e-5)
    result=dict(dictionary=old['dictionary'],coeff_scales=old['coeff_scales'],
        source=f'ImageNet train archive: first {args.per_class+1} entries/class, entry 3 reserved for heldout; frozen FP32 encoder, deterministic center crop',
        encoder_overlap_max_absolute_difference=delta,
        train=dict(latent=latent,labels=torch.tensor([row['label'] for row in rows]),filenames=filenames),
        heldout={key:old['heldout'][key] for key in ('latent','labels','filenames')})
    torch.save(result,args.output/'latents.pt')
    print(json.dumps(dict(train_images=len(rows),heldout_images=len(result['heldout']['latent']),
        encoder_overlap_max_absolute_difference=delta,cache=str(args.output/'latents.pt'))),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['extract','encode','merge'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--classes',type=Path,default=Path('/tmp/laser-investigation/train-classes.json'))
    p.add_argument('--archive',type=Path,default=Path('/workspace/Projects/data/imagenet/archives/ILSVRC2012_img_train.verified-20261005.tar'))
    p.add_argument('--original-images',type=Path,default=Path('/tmp/laser-investigation/images'))
    p.add_argument('--original-cache',type=Path,default=Path('/tmp/laser-investigation/balanced-soft-targets.pt'))
    p.add_argument('--teacher-source',type=Path,default=Path('/tmp/laser-investigation/source'))
    p.add_argument('--teacher',type=Path,default=Path('/tmp/laser-investigation/inputs/stage1.pt'))
    p.add_argument('--per-class',type=int,default=33)
    p.add_argument('--rank',type=int,default=0)
    p.add_argument('--world',type=int,default=2)
    args=p.parse_args()
    if args.per_class<4 or args.world<1 or not 0<=args.rank<args.world:p.error('invalid image count or shard')
    globals()[args.mode](args)


if __name__=='__main__':main()
