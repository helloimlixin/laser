#!/usr/bin/env python3
"""Minimal, lossless ImageNet reconstruction panels with vector PDF labels."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import flip_evaluator as flip
import matplotlib
matplotlib.use("Agg")
from matplotlib import colormaps
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from render_imagenet_stage1_zooms import choose_regions

METHODS = [("reference", "Input"), ("vqgan16", "VQGAN (16×16)"), ("rq4", "RQ-VAE (D=4)"),
           ("laser2", "LASER (D=2)"), ("laser4", "LASER (D=4)")]
CROP_COLOR = "#007eaf"
DISPLAY_SIZE = 384
COL, GAP, PAD, HEAD, ROW = 392, 12, 12, 48, 794
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "pdf.fonttype": 42})


def native(path):
    return Image.open(path).convert("RGB")


def zoom(image, box):
    crop = image.crop(box)
    assert crop.size == (64, 64)
    result = crop.resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.Resampling.NEAREST)
    factor = DISPLAY_SIZE // crop.width
    assert np.array_equal(np.asarray(result), np.repeat(np.repeat(np.asarray(crop),factor,0),factor,1))
    return result


def magma(error):
    return Image.fromarray(np.rint(colormaps["magma"](error)[...,:3]*255).astype(np.uint8))


def panel(indices, pixels, regions, methods=METHODS, errors=None):
    width = 2*PAD + len(methods)*COL + (len(methods)-1)*GAP
    canvas = Image.new("RGB", (width, HEAD+len(indices)*ROW-10), "white")
    for r,index in enumerate(indices):
        assert len(regions[index]) == 1, 'Exactly one shared crop per example is required.'
        box = regions[index][0]
        y = HEAD+r*ROW
        for c,(key,_) in enumerate(methods):
            x = PAD+c*(COL+GAP)
            source = native(pixels/f"{index:02d}-{key}.png")
            full = source.resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.Resampling.NEAREST)
            draw = ImageDraw.Draw(full)
            draw.rectangle((box[0]*1.5,box[1]*1.5,box[2]*1.5-1,box[3]*1.5-1),outline=CROP_COLOR,width=2)
            canvas.paste(full,(x+4,y))
            crop_source = magma(errors[index,key]) if errors is not None and key!='reference' else source
            xx,yy = x+(COL-DISPLAY_SIZE)//2,y+DISPLAY_SIZE+6
            canvas.paste(zoom(crop_source,box),(xx,yy))
            # Border lies outside the pixels, preserving every zoom pixel.
            ImageDraw.Draw(canvas).rectangle((xx-1,yy-1,xx+DISPLAY_SIZE,yy+DISPLAY_SIZE),outline=CROP_COLOR,width=1)
    return canvas


def write_panel(canvas, methods, output, pdf=None):
    raster = canvas.copy()
    draw = ImageDraw.Draw(raster)
    font = ImageFont.truetype(FONT, 33 if len(methods)==5 else 27)
    for c,(_,title) in enumerate(methods):
        x = PAD+c*(COL+GAP)+COL/2
        draw.text((x,HEAD/2-1),title,font=font,fill="black",anchor="mm")
    raster.save(output.with_suffix(".png"))
    inches = 6.5
    fig = plt.figure(figsize=(inches,inches*canvas.height/canvas.width))
    ax = fig.add_axes([0,0,1,1])
    ax.imshow(np.asarray(canvas), interpolation="none")
    ax.set_xlim(-.5,canvas.width-.5)
    ax.set_ylim(canvas.height-.5,-.5)
    ax.axis("off")
    for c,(_,title) in enumerate(methods):
        x = PAD+c*(COL+GAP)+COL/2
        ax.text(x,HEAD/2-1,title,ha="center",va="center",fontsize=9 if len(methods)==5 else 8)
    fig.savefig(output.with_suffix(".pdf"), dpi=300)
    if pdf is not None:
        pdf.savefig(fig,dpi=300)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work",type=Path,required=True)
    args=p.parse_args()
    selection=json.loads((args.work/'selection.json').read_text())
    samples=selection['samples']
    pixels=args.work/'pixels'
    output=args.work/'figures';output.mkdir(exist_ok=True)
    supplemental=args.work/'flip';supplemental.mkdir(exist_ok=True)
    # Retain the original first (central-detail) region for every example.
    regions={s['index']:choose_regions(native(pixels/f"{s['index']:02d}-reference.png"))[:1] for s in samples}
    # Subject crops may be specified from reference photos, without model ranking.
    for index,box in selection.get('region_overrides',{}).items():
        index=int(index)
        assert index in regions and len(box)==4
        x0,y0,x1,y1=box
        assert all(isinstance(v,int) for v in box)
        assert x1-x0==y1-y0==64 and 0<=x0<x1<=256 and 0<=y0<y1<=256
        regions[index]=[box]
    (args.work/'regions.json').write_text(json.dumps(regions,indent=2)+'\n')
    # Preserve selection order within each subset; never rank by model quality.
    priority={'requested':0,'new':1,'original':2}
    order=[s['index'] for s in sorted(samples,key=lambda s:priority.get(s['subset'],0))]
    main_indices=selection.get('main_figure_indices',order[:2])
    assert main_indices and all(index in order for index in main_indices)
    pages=[order[i:i+4] for i in range(0,len(order),4)]
    with PdfPages(output/'imagenet-reconstructions-all.pdf') as pdf:
        for page,indices in enumerate(pages,1):
            canvas=panel(indices,pixels,regions)
            write_panel(canvas,METHODS,output/f'comparison-{page:02d}',pdf)
            print(f'Rendered comparison page {page}/{len(pages)}',flush=True)
    write_panel(panel(main_indices,pixels,regions),METHODS,output/'imagenet-reconstructions-main')
    with PdfPages(output/'imagenet-laser421-all.pdf') as pdf:
        for page,indices in enumerate(pages,1):
            methods=[METHODS[0],METHODS[-1]]
            left=panel(indices[::2],pixels,regions,methods)
            right=panel(indices[1::2],pixels,regions,methods)
            offset=2*(COL+GAP)
            canvas=Image.new('RGB',(left.width+offset,max(left.height,right.height)),'white')
            canvas.paste(left,(0,0));canvas.paste(right,(offset,0))
            write_panel(canvas,methods*2,output/f'laser421-{page:02d}',pdf)
    for index in order:
        write_panel(panel([index],pixels,regions),METHODS,output/f'example-{index:02d}')

    errors={};metrics=[]
    for s in samples:
        index=s['index'];ref=np.asarray(native(pixels/f'{index:02d}-reference.png'),dtype=np.float32)/255
        for key,_ in METHODS[1:]:
            rec=np.asarray(native(pixels/f'{index:02d}-{key}.png'),dtype=np.float32)/255
            error,mean,params=flip.evaluate(ref,rec,'LDR',applyMagma=False)
            error=np.asarray(error).squeeze(-1)
            assert np.isfinite(error).all() and error.min()>=0 and error.max()<=1
            errors[index,key]=error
            np.save(supplemental/f'{index:02d}-{key}.npy',error)
            magma(error).save(supplemental/f'{index:02d}-{key}.png')
            metrics.append(dict(index=index,image_id=s['image_id'],model=key,flip_mean=float(mean),ppd=float(params['ppd'])))
    with (args.work/'flip-metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
    with PdfPages(output/'imagenet-flip-supplement.pdf') as pdf:
        for page,indices in enumerate(pages,1):
            canvas=panel(indices,pixels,regions,errors=errors)
            # One compact quantitative scale, shared by every error map.
            extended=Image.new('RGB',(canvas.width,canvas.height+40),'white')
            extended.paste(canvas,(0,0))
            bar=magma(np.tile(np.linspace(0,1,200,dtype=np.float32),(12,1)))
            bx=canvas.width//2-100;by=canvas.height+10
            extended.paste(bar,(bx,by))
            d=ImageDraw.Draw(extended);f=ImageFont.truetype(FONT,23)
            d.text((bx-18,by+6),'0',font=f,fill='black',anchor='mm')
            d.text((bx+220,by+6),'1',font=f,fill='black',anchor='mm')
            write_panel(extended,METHODS,output/f'flip-{page:02d}',pdf)
    old=json.loads((args.work.parent/'source/figures/manifest.json').read_text())
    result=dict(
        count=len(samples),new_images=sum(s['subset']!='original' for s in samples),classes=len({s['synset'] for s in samples}),
        selection=selection,display_order=order,comparison_pages=pages,main_figure_indices=main_indices,
        crop_size=64,zoom_factor=6,display_size=DISPLAY_SIZE,zoom_display_size=DISPLAY_SIZE,crops_per_image=1,region_selection=selection.get('region_selection','First central-detail region from the earlier reference-only gradient rule; one identical region across all models, enlarged beneath each full image at the same display size.'),
        figure_text='Only Input and model/grid/depth column headings. VQGAN explicitly identifies its 16×16 latent grid. No titles, filenames, rFID numbers, crop-coordinate labels, legends, or explanatory footers in reconstruction figures.',
        pdf='6.5-inch width; lossless native raster embedding and vector model headings',
        checkpoints={k:{'sha256':v['checkpoint_sha256']} for k,v in old['metrics'].items()},
        models=[dict(key=k,label=label,latent_shape=([16,16,1] if k=='vqgan16' else [8,8,2 if k=='laser2' else 4]),dictionary_size=16384) for k,label in METHODS[1:]],
        unavailable_exact_rows=old['unavailable_exact_rows'],
        vqgan_release=old['provenance']['vqgan16'],
        inference=[json.loads((args.work/f'inference-{key}.json').read_text()) for key,_ in METHODS[1:]],
        inference_note=selection.get('inference_note',f'All {len(samples)} examples reconstructed in CPU FP32 using the same verified checkpoint weights as the earlier gallery. Discrete code choices and pixels may differ across numerical backends. Comparisons with archived examples, when present, are recorded in the inference reports.'),
        flip=dict(implementation='NVIDIA flip-evaluator 1.7',dynamic_range='LDR',ppd=metrics[0]['ppd'],scale=[0,1],computed_before_crop=True),
    )
    (args.work/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    hashes={str(f.relative_to(args.work)):hashlib.sha256(f.read_bytes()).hexdigest() for folder in ['pixels','figures'] for f in (args.work/folder).iterdir() if f.is_file()}
    (args.work/'sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(json.dumps(dict(figures=str(output),count=len(samples),pages=len(pages))),flush=True)


if __name__=='__main__':
    main()
