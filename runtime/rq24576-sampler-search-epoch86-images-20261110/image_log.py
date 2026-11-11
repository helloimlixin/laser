"""Log fixed, unfiltered class panels without disturbing metric generation."""
import hashlib
import json
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def write(path, value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_name(path.name+f'.{os.getpid()}.pending')
    pending.write_text(json.dumps(value,indent=2)+'\n')
    pending.replace(path)


class ImageLogger:
    def __init__(self, plan, run):
        self.plan,self.run=plan,run
        self.local,self.output=Path(plan['local']),Path(plan['output'])
        self.records={}
        path=self.local/'image-logging.json'
        if path.exists(): self.records=json.loads(path.read_text())['panels']
        self.font=ImageFont.truetype('DejaVuSans.ttf',20)
        self.small=ImageFont.truetype('DejaVuSans.ttf',17)

    def ready(self):
        for folder in sorted((self.local/'previews').glob('*')):
            key=folder.name
            if key in self.records: continue
            paths=[folder/f'rank-{r}.json' for r in range(self.plan['world_size'])]
            if not all(p.exists() for p in paths): continue
            receipts=[json.loads(p.read_text()) for p in paths]
            rows=[]
            for rank,receipt in enumerate(receipts):
                assert receipt['rank']==rank and receipt['key']==key
                assert receipt['checkpoint_sha256']==self.plan['checkpoint_sha256']
                assert receipt['unfiltered'] and receipt['excluded_from_metric_count'] and receipt['sampling_rng_restored']
                path=folder/f'rank-{rank}.png'
                assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt['sha256']
                with Image.open(path) as im: rows.append(im.convert('RGB'))
            setting=receipts[0]['sampling']
            assert all(r['sampling']==setting for r in receipts)
            width=max(im.width for im in rows)
            height=120+sum(im.height+32 for im in rows)
            panel=Image.new('RGB',(width,height),(244,244,244));draw=ImageDraw.Draw(panel)
            draw.text((10,8),f"Epoch 86 | {key} | 64 unfiltered samples",font=self.font,fill='black')
            draw.text((10,39),f"T={setting['temperature']}  K={setting['top_k']}  P={setting['top_p']}",font=self.font,fill='black')
            draw.text((10,71),'Fixed class/preview seeds across configs; these images are excluded from FID/IS counts.',font=self.small,fill='black')
            y=120
            for im,receipt in zip(rows,receipts):
                c=receipt['class_info']
                draw.text((10,y+2),f"{c['name']} (class {c['imagenet_id']}) | preview seed {receipt['preview_seed']}",font=self.font,fill='black')
                y+=32;panel.paste(im,(0,y));y+=im.height
            destination=self.output/'previews'/(key+'.png');destination.parent.mkdir(parents=True,exist_ok=True)
            panel.save(destination)
            caption=f"Epoch 86 {key}; T={setting['temperature']}, K={setting['top_k']}, P={setting['top_p']}. 8 fixed classes x 8 unfiltered images, matched preview seeds; excluded from metric counts."
            if self.run is None: raise RuntimeError('Image logging requires an online W&B run')
            import wandb
            self.run.log({f'sampling/images/{key}':wandb.Image(str(destination),caption=caption)})
            record=dict(key=key,policy=receipts[0]['base_policy'],generation_seed=receipts[0]['generation_seed'],
                sampling=setting,images=sum(r['images'] for r in receipts),unfiltered=True,
                file=str(destination),sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                wandb_key=f'sampling/images/{key}',logged=True,receipts=receipts)
            self.records[key]=record
            evidence=dict(images_logged=True,panels=self.records,panel_count=len(self.records),
                total_images=sum(r['images'] for r in self.records.values()),run_url=self.run.url)
            write(self.local/'image-logging.json',evidence);write(self.output/'image-logging.json',evidence)
            print(json.dumps(dict(event='images_logged',key=key,images=record['images'])),flush=True)

    def table(self, screens, confirmations):
        self.ready()
        import wandb
        screen={r['policy']:r for r in screens}
        confirmation={(r['base_policy'],r['seed']):r for r in confirmations}
        table=wandb.Table(columns=['config','generation_seed','temperature','top_k','top_p','screen_mmd_not_fid','screen_is','fid50k','is50k','is_split_std','64_unfiltered_images'])
        for key,r in sorted(self.records.items()):
            s=screen.get(r['policy'],{});c=confirmation.get((r['policy'],r['generation_seed']),{})
            p=r['sampling']
            table.add_data(r['policy'],r['generation_seed'],json.dumps(p['temperature']),json.dumps(p['top_k']),json.dumps(p['top_p']),
                s.get('moment_mmd'),s.get('inception_score'),c.get('fid'),c.get('inception_score'),c.get('inception_score_std'),wandb.Image(r['file']))
        self.run.log({'sampling/comparison':table})
        self.run.summary.update({'images_logged':True,'image_panel_count':len(self.records),
            'unfiltered_preview_images':sum(r['images'] for r in self.records.values())})
