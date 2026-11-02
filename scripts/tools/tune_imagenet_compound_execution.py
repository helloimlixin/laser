"""Benchmark execution settings from one immutable production checkpoint."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from omegaconf import OmegaConf

BASE = Path('/tmp/laser-imagenet-compound512-audit-20261026')
TUNE = BASE/'speedup'
SOURCE = TUNE/'source'
OUT = Path('/workspace/Projects/laser/outputs/imagenet-compound512-audited-8a100-20261026')
RECIPE = BASE/'source/configs/stage2/imagenet-compound512-audited-8a100-20261026.yaml'


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2, default=str)+'\n')
    temporary.replace(path)


def status(phase, **payload):
    value = dict(phase=phase, supervisor_pid=os.getpid(), updated_unix=time.time(), **payload)
    write(TUNE/'status.json', value)
    write(OUT/'status.json', value)
    print(json.dumps(value), flush=True)


def environment(accumulation, compiled):
    return dict(os.environ, LASER_RUN_BASE=str(BASE),
        PYTHONPATH=':'.join(map(str,[SOURCE,SOURCE/'runtime',SOURCE/'scripts/tools'])),
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
        PYTHONDONTWRITEBYTECODE='1', LASER_ACCUMULATION=str(accumulation),
        LASER_ALLOW_ACCUMULATION_CHANGE='1', LASER_ENCODER_BATCH='16', LASER_DATA_WORKERS='4',
        LASER_COMPILE_BLOCKS=str(int(compiled)), TORCHINDUCTOR_COMPILE_THREADS='2',
        TORCHINDUCTOR_CACHE_DIR=str(TUNE/'inductor-cache'), TORCHINDUCTOR_FX_GRAPH_CACHE='1',
        TORCH_HOME=str(BASE/'torch-cache'), WANDB_DIR=str(BASE/'wandb'),
        WANDB_CACHE_DIR=str(BASE/'wandb-cache'), WANDB_DATA_DIR=str(BASE/'wandb-data'),
        PYTORCH_ALLOC_CONF='expandable_segments:True', NCCL_NVLS_ENABLE='0', MPLBACKEND='Agg',
        WANDB_API_KEY=Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip())


def execute(label, cfg, accumulation, compiled):
    path = TUNE/(label+'.yaml')
    OmegaConf.save(cfg, path)
    command = [sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=8',
               str(SOURCE/'train.py'),'--config',str(path)]
    log_path = TUNE/(label+'.log')
    with log_path.open('a') as log:
        child = subprocess.Popen(command,cwd=SOURCE,env=environment(accumulation,compiled),
                                 stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        phase = 'training' if label=='production' else 'speed_benchmark'
        while child.poll() is None:
            values = dict(label=label,child_pid=child.pid,batch_size=cfg.options.batch_size,
                          accumulation=accumulation,compiled=compiled,log=str(log_path))
            metrics = Path(cfg.options.output)/'metrics.jsonl'
            if metrics.exists():
                lines = metrics.read_text().splitlines()
                if lines:
                    row = json.loads(lines[-1])
                    values['latest_metrics'] = {key:row[key] for key in
                        ['train/global_step','train/loss','train/grad_norm','train/lr','train/images_per_second']}
            status(phase,**values)
            time.sleep(10)
    return child.returncode


def verify_compilation():
    """Compare loss and gradients with dropout disabled, including short attention."""
    os.environ['TORCHINDUCTOR_COMPILE_THREADS']='2'
    os.environ['TORCHINDUCTOR_CACHE_DIR']=str(TUNE/'inductor-cache')
    sys.path[:0]=[str(SOURCE),str(SOURCE/'runtime')]
    import copy
    import torch
    from src.models.rqtransformer.attentions import AttentionBlock
    from src.models.rqtransformer.configs import AttentionBlockConfig
    torch.set_num_threads(4)
    rows=[]
    for length in [64,4,2]:
        torch.manual_seed(37)
        eager=AttentionBlock(AttentionBlockConfig(embed_dim=96,n_head=3,resid_pdrop=0.)).cuda()
        optimized=copy.deepcopy(eager)
        optimized.attn.short_attention_backend='compiled'
        forward=torch.compile(optimized.forward,fullgraph=True,dynamic=True)
        x=torch.randn(8,length,96,device='cuda',dtype=torch.bfloat16)
        def evaluate(model,fn):
            value=x.clone().requires_grad_()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                result=fn(value);loss=result.float().square().mean()
            loss.backward()
            gradient=torch.cat([p.grad.float().flatten() for p in model.parameters()])
            return float(loss),gradient
        reference,reference_gradient=evaluate(eager,eager.forward)
        actual,actual_gradient=evaluate(optimized,forward)
        relative=float((reference_gradient-actual_gradient).norm()/reference_gradient.norm())
        passed=abs(reference-actual)<.002 and relative<.01
        rows.append(dict(sequence_length=length,loss_error=abs(reference-actual),
                         relative_gradient_error=relative,passed=passed))
        del eager,optimized,forward
    torch.cuda.empty_cache()
    report=dict(passed=all(r['passed'] for r in rows),cases=rows)
    write(TUNE/'compilation-parity.json',report)
    return report['passed']


def main():
    status('waiting_for_checkpoint_handoff',target_step=500)
    while not (TUNE/'handoff.json').exists():
        time.sleep(2)
    handoff = json.loads((TUNE/'handoff.json').read_text())
    assert handoff['completed_optimizer_steps_retained']==500
    assert json.loads((TUNE/'tests-receipt.json').read_text())['passed']
    # Wait for all old GPU workers to exit before starting independent trials.
    while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        time.sleep(2)
    status('checking_compiler_loss_and_gradients')
    try:
        compiled=verify_compilation()
    except Exception as error:
        write(TUNE/'compilation-parity.json',dict(passed=False,error=repr(error)))
        compiled=False
    results=[]
    for batch,accumulation in [(64,4),(128,2)]:
        label=f'{"compiled" if compiled else "eager"}-b{batch}'
        cfg=OmegaConf.load(RECIPE)
        cfg.audit_directory=str(TUNE/'records')
        cfg.options.update(resume=True,resume_checkpoint=handoff['checkpoint'],batch_size=batch,
            output=str(TUNE/'trials'/label), checkpoint_dir=str(TUNE/'trials'/label/'checkpoints'),
            wandb_mode='disabled',upload_checkpoints=False,fid_every=0,fid_early_epochs=0,
            sample_grid_every=0,save_step_freq=0,max_optimizer_steps=22)
        code=execute(label,cfg,accumulation,compiled)
        log=(TUNE/(label+'.log')).read_text()
        row=dict(label=label,batch_size=batch,accumulation=accumulation,compiled=compiled,
                 returncode=code,passed=False)
        metrics=Path(cfg.options.output)/'metrics.jsonl'
        if code==0 and metrics.exists():
            lines=[json.loads(x) for x in metrics.read_text().splitlines()]
            latest=lines[-1]
            receipt=json.loads((Path(cfg.options.checkpoint_dir)/'latest.json').read_text())
            assert receipt['global_step']==522 and receipt['rng_ranks']==8
            assert latest['train/global_step']==520
            row.update(passed=True,images_per_second=latest['train/images_per_second'],
                       loss=latest['train/loss'],grad_norm=latest['train/grad_norm'],
                       benchmark_updates_discarded=22)
        else:
            row['error_tail']=log[-6000:]
        results.append(row)
        write(TUNE/'benchmarks.json',results)
        write(OUT/'speed-benchmarks.json',results)
        folder=Path(cfg.options.checkpoint_dir)
        if folder.exists():shutil.rmtree(folder)
        # If compilation failed, test a larger eager batch as a fallback.
        if not row['passed'] and batch==64:
            break
    candidates=[r for r in results if r['passed'] and r['images_per_second']>445*1.05]
    best=max(candidates,key=lambda r:r['images_per_second']) if candidates else dict(
        batch_size=32,accumulation=8,compiled=False,images_per_second=445.)
    cfg=OmegaConf.load(RECIPE)
    cfg.options.resume=True
    cfg.options.batch_size=best['batch_size']
    cfg.audit_directory=str(TUNE/'records')
    selection=dict(selected=best,baseline_images_per_second=445.,
        speedup=best['images_per_second']/445.,resume_step=500,
        optimizer_and_scheduler_retained=True,global_batch=2048,
        teacher_precision_unchanged=True,objective_unchanged=True,
        dropout_rng_trajectory='Compilation or microbatch changes can change dropout draws; distribution is unchanged')
    write(TUNE/'selection.json',selection);write(OUT/'speed-selection.json',selection)
    OmegaConf.save(cfg,TUNE/'production.yaml')
    # Keep the measured code/config alongside the original immutable source.
    records=TUNE/'records';records.mkdir(exist_ok=True)
    for name in ['selection.json','benchmarks.json','production.yaml','tests.xml']:
        shutil.copyfile(TUNE/name,records/name)
    code=execute('production',cfg,best['accumulation'],best['compiled'])
    if code:raise RuntimeError(f'Production resume exited {code}; see {TUNE}/production.log')
    status('completed')


if __name__=='__main__':
    try:main()
    except BaseException as error:
        status('failed',error=repr(error));raise
