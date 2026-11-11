"""Isolate the verified joint16k codec and native RQ training recipe."""
import hashlib
import json
import os
from pathlib import Path
import shutil

from omegaconf import OmegaConf

ROOT = Path('/workspace/Projects/laser')
RUN = 'imagenet-beam16-joint1694m-native-scratch200-4h200-20261107'
BASE = Path('/tmp') / RUN
OUT = ROOT / 'outputs' / RUN
SOURCE = Path('/tmp/laser-beam-recovery/frozen')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    assert not BASE.exists(), 'Do not overwrite an existing experiment'
    report = json.loads((ROOT / 'outputs/imagenet-beam16-validation50k-20261107/results.json').read_text())
    assert report['passed'] and report['validation_rfid_remeasured'] and report['images'] == 50000
    BASE.mkdir(); OUT.mkdir(exist_ok=True)
    for name in ('source', 'upstream', 'runtime'):
        shutil.copytree(SOURCE / name, BASE / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copyfile(Path(__file__).parent/'asynchronous_checkpoints.py', BASE/'runtime/asynchronous_checkpoints.py')
    inputs = BASE / 'inputs'; inputs.mkdir()
    for name in ('joint-codebook.pt', 'reconstruction-codebook.pt', 'imagenet_256_train.npz',
                 'preview-content-policy.json', 'paired-tokenizer-artifact.json'):
        shutil.copyfile(SOURCE / 'inputs' / name, inputs / name)
    os.link(SOURCE / 'inputs/stage1-model.pt', inputs / 'stage1-model.pt')
    config = OmegaConf.load(SOURCE / 'inputs/stage2-training-config.yaml')
    config.arch.embed_dim = 1664
    config.arch.body.n_layer = 44
    config.arch.body.block.n_head = 26
    config.arch.head.block.n_head = 26
    config.optimizer.max_gn = 1.0
    config.experiment.epochs = 200
    config.experiment.batch_size = 256
    for name in ('stage2-training-config.yaml', 'stage2-archive-config.yaml'):
        OmegaConf.save(config, inputs / name)
    recipe = OmegaConf.load(SOURCE / 'inputs/recipe.yaml')
    recipe.name = 'ImageNet beam16 joint16k | 1.69B | native soft T0.5 | fresh200 | 4H200'
    recipe.run_id = RUN
    recipe.epochs = recipe.lr_schedule_epochs = 200
    recipe.microbatch = 256
    recipe.tokenizer.measured_rfid = report['results']['joint_beam16']['rfid']
    recipe.tokenizer.native_greedy_rfid = report['results']['joint_greedy']['rfid']
    recipe.lineage = dict(source_run=report['source_run'], fresh_stage2=True,
                         source_reconstruction_verification='imagenet-beam16-validation50k-20261107',
                         source_ar_weights_loaded=False, source_optimizer_loaded=False)
    OmegaConf.save(recipe, inputs / 'recipe.yaml')
    data = BASE / 'data'; data.mkdir()
    train = Path('/tmp/imagenet-laser-continuous-compound-native-scratch100-4h200-20261107/data')
    val = Path('/tmp/laser-beam-recovery/validation-data')
    for name in ('ILSVRC2012_img_train.tar', 'train-index.npz', 'verified.json'):
        (data / name).symlink_to(train / name)
    for name in ('ILSVRC2012_img_val.tar', 'val-index.npz'):
        (data / name).symlink_to(val / name)
    (BASE / 'torch-cache').symlink_to(SOURCE / 'torch-cache', target_is_directory=True)
    for name in ('wandb', 'checkpoint-staging', 'checkpoint-cache'):
        (BASE / name).mkdir()
    driver = BASE / 'source/src/training/rq_imagenet_shared.py'
    text = driver.read_text()
    text = text.replace('len(model.body_transformer.blocks)==42', 'len(model.body_transformer.blocks)==44')
    text = text.replace('assert model.config.embed_dim==1536 and model.config.vocab_size_cond==1000',
                        'assert model.config.embed_dim==1664 and model.config.vocab_size_cond==1000')
    # A fresh run has no inherited Inception-score floor.
    # Keep every lower-FID evaluation while reporting IS alongside it.
    text = text.replace("lr_amplitude_factor=float(os.environ.get('LASER_LR_AMPLITUDE_FACTOR','0.15'))",
                        "lr_amplitude_factor=float(os.environ.get('LASER_LR_AMPLITUDE_FACTOR','1.0'))")
    text = text.replace("name=PROFILE.name,\n            resume=",
                        "name=PROFILE.name,settings=wandb.Settings(console='off',disable_git=True,x_disable_stats=True),\n            resume=")
    text = text.replace('        runtime_support.workspace(run,PROFILE,OUT)',
                        "        try:\n            runtime_support.workspace(run,PROFILE,OUT)\n        except Exception as error:\n            write_json(OUT/'workspace-error.json',dict(error=str(error)))")
    text = text.replace('                    runtime_support.sample_panels(PROFILE,model,aux,device,rank,world,global_step,run,OUT)\n                    checkpoint(epoch,epoch,absolute+1)',
                        '                    runtime_support.sample_panels(PROFILE,model,aux,device,rank,world,global_step,run,OUT)\n                    if global_step==10 or global_step%1000==0:\n                        checkpoint(epoch,epoch,absolute+1)')
    text = text.replace("        if run:run.summary['training/status']='early_stopped_quality_regression' if quality_stopped else 'completed';run.finish()",
                        "        if run:\n            from runtime.asynchronous_checkpoints import finish_uploads\n            finish_uploads()\n            run.summary['training/status']='early_stopped_quality_regression' if quality_stopped else 'completed'\n            run.finish()")
    first = text.index('            if first==0 and run is not None:\n')
    last = text.index('        assert int(metric.fake_count)==local_count',first)
    text = text[:first]+text[last:]
    driver.write_text(text)
    acceleration = BASE / 'runtime/training_acceleration.py'
    text = acceleration.read_text().replace('or self.attn_drop.p:', 'or self.attn_drop.p or x.shape[1] <= 8:')
    acceleration.write_text(text)
    support = BASE / 'source/src/training/joint_runtime_support.py'
    text = support.read_text()
    text = text.replace("metadata=dict(native_rq=True,fresh=False,", "metadata=dict(native_rq=True,fresh=True,")
    text = text.replace("get('source_epoch',65)", "get('source_epoch',0)")
    text = text.replace("title='ImageNet FID14.5739 original unscaled cosine continue100 4H100 20261105'", "title=' '.join(profile.name.replace('|',' ').split())")
    text = text.replace("def workspace(run,profile,out):\n", "def workspace(run,profile,out):\n    if (Path(out)/'workspace.json').exists():return\n")
    text = text.replace("assert len(restored['optimizer']['state'])==780", "assert len(restored['optimizer']['state'])==len(state['optimizer']['state'])")
    text = text.replace('optimizer_states=780,', "optimizer_states=len(state['optimizer']['state']),")
    # Serialize to NVMe, atomically commit the persistent copy, and upload from the local buffer.
    old = "    try:\n        torch.save(state,tmp);tmp.replace(path)\n    except BaseException:\n        tmp.unlink(missing_ok=True)\n        raise"
    new = "    from src.training.k4_checkpoint_io import atomic_torch_save\n    atomic_torch_save(state,path)"
    assert old in text
    text = text.replace(old, new)
    text = text.replace("artifact.add_file(str(path),name='last.pt',policy='immutable',skip_cache=True)",
                        "from src.training.k4_checkpoint_io import _checkpoint_upload_source\n        artifact.add_file(str(_checkpoint_upload_source(path)),name='last.pt',policy='immutable',skip_cache=True)")
    old = "    if run is not None:\n        for old in folder.glob('step-*.pt'):\n            if old!=path:old.unlink()"
    new = "    if run is not None:\n        from src.training.k4_checkpoint_io import prune_local_checkpoint_cache\n        keep={path}\n        best_receipt=folder/'best.json'\n        if best_receipt.exists():keep.add(Path(json.loads(best_receipt.read_text())['path']))\n        prune_local_checkpoint_cache(keep)\n        for old in folder.glob('step-*.pt'):\n            if old not in keep:old.unlink()"
    assert old in text
    text = text.replace(old, new)
    support.write_text(text)
    worker = '''import os,sys
from pathlib import Path
root=Path(__file__).resolve().parent
sys.path[:0]=[str(root/'source'),str(root/'upstream'),str(root)]
os.environ['RQ_UPSTREAM_ROOT']=str(root/'upstream')
os.environ['TORCH_HOME']=str(root/'torch-cache')
import torch
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=True
torch.set_float32_matmul_precision('highest')
from runtime.precision import install_bf16_training
install_bf16_training()
from runtime.training_acceleration import install
install()
from runtime.asynchronous_checkpoints import install as install_async_checkpoints
install_async_checkpoints()
from src.training.rq_imagenet_shared import main
if __name__=='__main__':main('laser_joint',root)
'''
    (BASE / 'train.py').write_text(worker)
    execution = dict(fresh_stage2=True, source_epoch=0, target_generation_fid=11.0,
        source_generation_fid=14.1633995743926, measured_validation_beam_rfid=report['results']['joint_beam16']['rfid'],
        teacher='unchanged native stochastic soft T0.5', reconstruction_beam_width=16,
        model=dict(embed_dim=1664, body_layers=44, head_layers=6, attention_heads=26),
        epochs=200, cosine_schedule_epochs=200, global_batch=2048, gradient_clip_max_norm=1.0,
        checkpoint_every_updates=1000, preview_every_updates=250,
        checkpoint_cloud_upload='separate executor, evaluated checkpoints only',
        reference='50k generated vs full ImageNet train, original RQ Inception, no rejection',
        official_rq_config='https://github.com/kakaobrain/rq-vae-transformer/blob/main/configs/imagenet256/stage2/in256-rqtransformer-8x8x4-1400M.yaml',
        stage1_hashes=report['hashes'], attention='spatial SDPA for training only; native depth and sampling')
    write(inputs / 'execution.json', execution)
    write(OUT / 'plan.json', execution)
    repro = OUT / 'reproducibility'; repro.mkdir()
    for path in inputs.iterdir():
        if path.stat().st_size < 50 * 1024**2: shutil.copyfile(path, repro / path.name)
    write(BASE / 'prepared.json', dict(base=str(BASE), output=str(OUT), run=RUN))
    print(json.dumps(dict(base=str(BASE), output=str(OUT), prepared=True)), flush=True)


if __name__ == '__main__': main()
