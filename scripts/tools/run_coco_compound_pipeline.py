"""Resume-capable COCO LASER stage1 -> selected tokenizer -> cache -> text prior."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(value, indent=2)+'\n'); temp.replace(path)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def export_tokenizer(source, target, config):
    import torch
    from src.training.rqtransformer import LaserAux
    from src.stage1_setup import build_stage1_model
    from src.training.cli import load_config
    from src.data.coco2014 import COCO2014Dataset
    from scripts.tools.build_coco_compound_cache import transform
    torch.set_num_threads(4)
    state = torch.load(source, map_location='cpu', weights_only=False)
    prefixes = ('encoder.','decoder.','pre_bottleneck.','post_bottleneck.','bottleneck.dictionary')
    weights = {k:v for k,v in state['state_dict'].items() if k.startswith(prefixes)}
    target = Path(target); target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=weights, epoch=state['epoch'], global_step=state['global_step']), target)
    cfg = load_config(config)
    native = build_stage1_model(cfg.model,cfg.train,cfg.data).cuda().eval()
    missing, unexpected = native.load_state_dict(state['state_dict'], strict=False)
    if missing or any(not k.startswith(('val_rfid.','test_fid.')) for k in unexpected):
        raise ValueError(f'Stage1 reload failed: {missing}, {unexpected}')
    aux = LaserAux(target,16384,2048,3.,attn_resolutions=(16,),sparsity_level=4,clamp_coeffs=False).cuda()
    dataset = COCO2014Dataset(cfg.data.data_dir, transform=transform())
    images = torch.stack([dataset[i][0] for i in range(2)]).cuda()
    with torch.inference_mode():
        z_native = native.pre_bottleneck(native.encoder(images))
        z_aux = aux.quant_conv(aux.encoder(images))
        torch.testing.assert_close(z_native,z_aux,rtol=0,atol=0)
        native_pixels = native.decoder(native.post_bottleneck(z_native))
        aux_pixels = aux.decoder(aux.post_quant_conv(z_aux))
        torch.testing.assert_close(native_pixels,aux_pixels,rtol=0,atol=0)
        atoms, coeffs = aux.encode_sparse_components(images)
        assert atoms.shape == (2,8,8,4) and torch.isfinite(coeffs).all()
    result = dict(source=str(source), source_sha256=digest(source), stage1_sha256=digest(target),
        epoch=int(state['epoch']), global_step=int(state['global_step']),
        encoder_decoder_exact=True, latent_shape=[8,8,4])
    write(target.with_suffix('.json'),result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base',type=Path,required=True)
    args = p.parse_args(); base = args.base.resolve(); base.mkdir(parents=True,exist_ok=True)
    lock = (base/'pipeline.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest = json.loads((base/'runtime-manifest.json').read_text())
    for relative, expected in manifest.items():
        if digest(ROOT/relative) != expected:
            raise RuntimeError(f'Snapshotted runtime changed: {relative}')
    config1 = ROOT/'configs/stage1/coco2014-laser-k4-4a100.yaml'
    config2 = base/'stage2-resolved.yaml'
    from omegaconf import OmegaConf
    from src.training.cli import load_config
    stage1 = load_config(config1)
    options2 = load_config(ROOT/'configs/stage2/coco2014-ffhq-compound-k4-4a100.yaml')
    (base/'assets').mkdir(exist_ok=True)
    env = {**os.environ, 'WANDB_MODE':'online','WANDB_ENTITY':'helloimlixin-rutgers',
           'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4',
           'TOKENIZERS_PARALLELISM':'false','PYTHONUNBUFFERED':'1',
           'TMPDIR':'/tmp/laser-coco',
           'WANDB_RUN_GROUP':'coco2014-laser-k4-compound-20260923'}
    for key in ('WANDB_SERVICE','_WANDB_SERVICE'):
        env.pop(key,None)
    Path(env['TMPDIR']).mkdir(parents=True,exist_ok=True)
    def status(phase,**extra):
        write(base/'status.json',dict(phase=phase,pid=os.getpid(),updated_unix=time.time(),**extra))
    def command(phase,argv):
        with (base/f'{phase}.log').open('ab') as log:
            process = subprocess.Popen(argv,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            write(base/f'{phase}-launch.json',dict(pid=process.pid,command=argv,started_unix=time.time()))
            while process.poll() is None:
                status(phase,child_pid=process.pid)
                time.sleep(15)
        if process.returncode:
            raise RuntimeError(f'{phase} failed with exit {process.returncode}; inspect {phase}.log')
    distributed = [sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=4']
    try:
        if not (base/'stage1-complete.json').exists():
            stage1_output = base/'stage1'
            latest = list((stage1_output/'checkpoints').glob('run_*/laser/last.ckpt'))
            overrides = [f'output_dir={stage1_output}']
            if latest:
                overrides.append(f'ckpt_path={max(latest,key=lambda x:x.stat().st_mtime)}')
            command('stage1',[sys.executable,str(ROOT/'train.py'),'--config',str(config1),*overrides])
            best = stage1_output/'wandb_checkpoints/best-01.ckpt'
            if not best.is_file():
                raise RuntimeError('Stage1 completed without a best reconstruction-FID checkpoint')
            write(base/'stage1-complete.json',dict(best=str(best),sha256=digest(best)))
        chosen = json.loads((base/'stage1-complete.json').read_text())
        local_checkpoint = Path(options2.options.checkpoint)
        provenance = base/'assets/stage1.json'
        if not provenance.is_file() or not local_checkpoint.is_file():
            status('export_tokenizer')
            # Keep CUDA initialization outside the persistent supervisor process.
            command('export_tokenizer',[sys.executable,'-c',
                'from scripts.tools.run_coco_compound_pipeline import export_tokenizer; import sys; export_tokenizer(*sys.argv[1:])',
                chosen['best'],str(local_checkpoint),str(config1)])
            shutil.copyfile(local_checkpoint,base/'assets/stage1.pt')
            shutil.copyfile(local_checkpoint.with_suffix('.json'),provenance)
        info = json.loads(provenance.read_text())
        options2.options.stage1_sha256 = info['stage1_sha256']
        resolved = OmegaConf.to_container(options2,resolve=True)
        resolved['defaults'] = ['_self_']
        OmegaConf.save(OmegaConf.create(resolved),config2)
        if not (base/'cache/complete.json').is_file():
            command('cache',distributed+[str(ROOT/'scripts/tools/build_coco_compound_cache.py'),
                '--config',str(config2),'--output',str(base/'cache')])
        else:
            for name in ('token_cache','validation_cache'):
                target=Path(options2.options[name])
                if not target.is_file():
                    target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(base/'assets'/target.name,target)
        if not (base/'production-preflight/stage2.json').is_file():
            command('stage2-preflight',distributed+[str(ROOT/'scripts/tools/preflight_coco_compound.py'),
                '--config',str(config2),'--output',str(base/'production-preflight')])
        proof=json.loads((base/'production-preflight/stage2.json').read_text())
        if not proof['passed'] or proof['stage1_sha256'] != info['stage1_sha256']:
            raise RuntimeError('Stage2 preflight does not match the selected tokenizer')
        import wandb
        wb=wandb.init(entity=options2.options.wandb_entity,project='laser',id=options2.options.wandb_id,
            name=options2.options.wandb_name,resume='allow',mode='online',config=resolved,allow_val_change=True)
        wb.summary['pipeline/phase']='cache_ready'
        for path in (config2,provenance,base/'recipe.md',base/'runtime-manifest.json',base/'production-preflight/stage2.json'):
            wb.save(str(path),base_path=str(base),policy='now')
        wb.finish()
        command('stage2',distributed+[str(ROOT/'train.py'),'--config',str(config2)])
        status('completed')
    except BaseException as error:
        status('failed',error=str(error))
        raise


if __name__ == '__main__':
    main()
