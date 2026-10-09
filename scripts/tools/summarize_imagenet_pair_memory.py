"""Summarize paired ImageNet measurements and fixed training-cache diagnostics."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

OUT = Path('/workspace/Projects/laser/outputs/imagenet-pair-memory-investigation-20261007')
PAIR = Path('/workspace/Projects/laser/outputs/imagenet-pair-memory-trial-20261007')
PRODUCTION = Path('/workspace/Projects/laser/outputs/imagenet-rfid421-classcond-8h100-20261007')
ROOT = Path('/workspace/Projects/laser')
METRICS = ('atom_nll','coefficient_kl','atom_top1','coefficient_physical_mae')


def read(path):
    return json.loads(path.read_text())


def summarize():
    import numpy as np
    cross = read(OUT / 'cross-teacher-probe-1000.json')
    mlp = read(OUT / 'mlp-teacher-probe-1000.json')
    assert cross['indices'] == mlp['indices']
    full = np.array(cross['results']['full']['per_image_and_depth'])
    def differences(array):
        delta = array-full
        images = delta.mean(1)
        return {name:dict(mean=float(images[:,i].mean()),
            paired_standard_error_across_images=float(images[:,i].std(ddof=1)/len(images)**.5),
            by_depth=delta[:,:,i].mean(0).tolist()) for i,name in enumerate(METRICS)}
    control = {'mlp_minus_cross':differences(np.array(mlp['results']['full']['per_image_and_depth']))}
    baseline_path = OUT / 'baseline-teacher-probe-1000.json'
    if baseline_path.is_file():
        baseline = read(baseline_path)
        assert baseline['indices'] == cross['indices']
        control['plain_minus_cross'] = differences(np.array(baseline['results']['full']['per_image_and_depth']))
    ablations = {mode:differences(np.array(value['per_image_and_depth']))
                 for mode,value in cross['results'].items() if mode != 'full'}
    memory_path = OUT / 'cross-teacher-probe-1000-memory.json'
    memory = read(memory_path) if memory_path.is_file() else cross
    if memory_path.is_file():
        assert memory['indices'] == cross['indices']
        np.testing.assert_allclose(memory['results']['full']['per_image_and_depth'],full,atol=1e-5,rtol=0)
        for mode,value in memory['results'].items():
            if mode != 'full':
                ablations[mode] = differences(np.array(value['per_image_and_depth']))
    comparison_path = OUT / 'comparison.json'
    comparison = read(comparison_path) if comparison_path.is_file() else None
    decompositions = {}
    for branch in ('baseline','cross','mlp'):
        for seed in (261001,271001):
            path = OUT / 'verification' / branch / f'seed-{seed}' / 'fid-decomposition.json'
            if path.is_file():
                decompositions[f'{branch}/{seed}'] = read(path)
    bos_decomposition = OUT / 'verification/cross/seed-271001-memory-bos/fid-decomposition.json'
    if bos_decomposition.is_file():
        decompositions['cross_BOS/271001'] = read(bos_decomposition)
    report = dict(status='complete' if comparison is not None and baseline_path.is_file() else 'in_progress',
        endpoint_step=3798, original_FID_comparison=read(PAIR / 'comparison.json'),
        repeated_FID_comparison=comparison,FID_decompositions=decompositions,
        training_loss_audit=read(OUT / 'training-loss-audit.json'),
        training_probe=dict(images=1000,classes=1000,shared_indices_verified=True,
            coefficient_sampling_seed=7319,control_minus_cross=control,
            inference_ablations_minus_full=ablations,attention=memory['attention'],
            attention_diagnostic_images=memory.get('diagnostic_attention_images'),
            residual_rms=memory.get('residual_rms'),block_rms=memory.get('block_rms')),
        limits=['Training-cache diagnostics are not held-out validation.',
            'Inference ablations are not independently retrained controls.',
            'A second sampling seed does not establish robustness across training seeds.',
            'Attention mass toward earlier sites must be compared with the number of legal records.',
            'Native optimizer, sampler and RNG state are preserved, but training replay is not bitwise deterministic.'],
        no_automatic_architecture_promotion=True,time=time.time())
    path = OUT / 'investigation-report.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report,indent=2)+'\n')
    temporary.replace(path)
    plot(report)
    return report


def plot(report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    colors = dict(cross='#2463b4',mlp='#7c8492',baseline='#cc7b28')
    fig,axes = plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    comparison = report['repeated_FID_comparison']
    ax=axes[0,0]
    if comparison:
        seeds=(261001,271001)
        for index,branch in enumerate(('baseline','mlp','cross')):
            ax.bar(np.arange(2)+(index-1.5)*.18,[comparison['metrics'][branch][str(s)]['fid'] for s in seeds],
                   width=.18,label=branch,color=colors[branch])
        ax.bar([1+.27],[comparison['bos_memory_inference_ablation']['fid']],width=.18,
               label='cross, BOS memory',color='#9960b0')
        ax.set_xticks([0,1],['Seed 261001','Seed 271001'])
    else:
        metrics=report['original_FID_comparison']['metrics']
        ax.bar(['MLP','Cross'],[metrics[b]['fid'] for b in ('mlp','cross')],color=[colors['mlp'],colors['cross']])
    ax.set_title('FID, 50,000 images, step 3,798');ax.set_ylabel('Lower is better')
    if comparison:ax.legend()
    series={}
    for branch in ('cross','mlp','baseline'):
        paths=([PRODUCTION/'train/metrics.jsonl',OUT/'baseline/train/metrics.jsonl'] if branch=='baseline'
               else [PAIR/branch/'train/metrics.jsonl'])
        values={}
        for path in paths:
            if not path.is_file():continue
            for line in path.read_text().splitlines():
                row=json.loads(line);step=row.get('train/global_step',0)
                if 1750<step<=3798 and 'train/loss'in row:values[step]=row
        series[branch]=values
    for ax,key,title in [(axes[0,1],'train/atom_nll','Training atom NLL'),
                         (axes[0,2],'train/coeff_kl','Training coefficient KL')]:
        for branch,values in series.items():
            steps=sorted(values);y=np.array([values[s][key] for s in steps]);window=21
            if len(y)>=window:ax.plot(steps[window-1:],np.convolve(y,np.ones(window)/window,'valid'),label=branch,color=colors[branch])
        ax.set_title(title+' (21-point mean)');ax.set_xlabel('Optimizer step');ax.legend()
    ax=axes[1,0];probe=report['training_probe'];controls=probe['control_minus_cross']
    for index,(name,value) in enumerate(controls.items()):
        ax.bar(np.arange(4)+(index-(len(controls)-1)/2)*.3,value['atom_nll']['by_depth'],width=.3,label=name.replace('_',' '))
    ax.axhline(0,color='black',linewidth=.7);ax.set_xticks(range(4));ax.set_xlabel('OMP depth');ax.set_ylabel('NLL difference, nats');ax.set_title('Fixed training probe: atom prediction');ax.legend(fontsize=8)
    ax=axes[1,1];modes=[m for m in ('both_latest','both_uniform','both_local','memory_bos','memory_zero') if m in probe['inference_ablations_minus_full']]
    for index,metric in enumerate(('atom_nll','coefficient_kl')):
        values=[probe['inference_ablations_minus_full'][m][metric] for m in modes]
        ax.bar(np.arange(len(modes))+(index-.5)*.35,[v['mean'] for v in values],width=.35,
               yerr=[v['paired_standard_error_across_images'] for v in values],label=metric.replace('_',' '),capsize=2)
    ax.axhline(0,color='black',linewidth=.7);ax.set_xticks(range(len(modes)),[m.replace('both_','').replace('memory_','').replace('_',' ') for m in modes],rotation=20);ax.set_title('Inference restrictions minus full');ax.set_ylabel('Loss difference, nats');ax.legend(fontsize=8)
    ax=axes[1,2];attention=probe['attention'];names=list(attention)
    ax.bar(range(len(names)),[attention[n]['overall']['normalized_entropy'] for n in names],color=colors['cross'])
    ax.set_xticks(range(len(names)),[n.replace('_',' ') for n in names],rotation=20);ax.set_ylim(0,1);ax.set_ylabel('Entropy / log(legal records)');ax.set_title('Attention spread (1 = uniform)')
    fig.suptitle('ImageNet pair-memory investigation — training probes are not held-out validation',fontsize=14)
    fig.savefig(OUT/'investigation.png',dpi=150);plt.close(fig)


def upload():
    import wandb
    os.environ['WANDB_API_KEY']=Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip()
    api=wandb.Api()
    for run_id in ('imagenet-rfid421-pair-memory-cross-8h100-20261007','imagenet-rfid421-plain-matched-3798-20261007'):
        run=api.run('helloimlixin-rutgers/laser/'+run_id)
        for name in ('investigation-report.json','investigation.png','comparison.json'):
            run.upload_file(str(OUT/name),root=str(OUT))
    (OUT/'report-upload.json').write_text(json.dumps(dict(completed=True,time=time.time()))+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');parser.add_argument('--upload-wandb',action='store_true');args=parser.parse_args()
    if args.watch:
        while not (OUT/'comparison.json').is_file():
            status=read(OUT/'launch-status.json')
            if status.get('production'):
                raise RuntimeError('Production resumed before comparison completed; inspect investigation supervisor.')
            time.sleep(5)
        if not (OUT/'baseline-teacher-probe-1000.json').is_file():
            with (OUT/'baseline-teacher-probe.log').open('a') as log:
                subprocess.run([sys.executable,str(ROOT/'scripts/tools/probe_imagenet_pair_memory.py'),
                    '--branch','baseline','--samples','1000','--batch-size','4'],stdout=log,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
    report=summarize()
    if args.upload_wandb:
        assert report['status']=='complete';upload()
    print(json.dumps(dict(status=report['status'],report=str(OUT/'investigation-report.json'))),flush=True)


if __name__=='__main__':
    main()
