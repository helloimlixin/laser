"""Audit checkpoint identity, optimizer clock, evaluator code, and metric reports."""
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

LOCAL=Path('/tmp/laser-rq24576-epoch80-vs82-audit-20261110')

def write(name,value):
    plan=json.loads((LOCAL/'plan.json').read_text())
    for folder in [LOCAL,Path(plan['output'])]:
        p=folder/name;t=p.with_suffix('.pending');t.write_text(json.dumps(value,indent=2)+'\n');t.replace(p)

def tree(path,name):
    nodes=[node for node in ast.walk(ast.parse(Path(path).read_text())) if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name==name]
    assert len(nodes)==1
    return ast.dump(nodes[0],include_attributes=False)

def main():
    plan=json.loads((LOCAL/'plan.json').read_text());base=Path(plan['base'])
    sys.path[:0]=[str(base/'source'),str(base),str(base/'upstream')]
    os.environ['RQ_UPSTREAM_ROOT']=str(base/'upstream')
    import numpy as np
    import torch
    from src import rqvae_metrics as metrics
    torch.set_num_threads(4)
    paths=[Path('/tmp/laser-rq24576-tuned-eval-epoch80-20261109/frozen-checkpoint.pt'),Path(plan['checkpoint'])]
    states=[torch.load(path,map_location='cpu',mmap=True,weights_only=False) for path in paths]
    a,b=states
    assert a['recipe']==b['recipe'] and a['config']['arch']==b['config']['arch']
    assert a['joint_codebook_sha256']==b['joint_codebook_sha256']==plan['codebook_sha256']
    assert list(a['state_dict'])==list(b['state_dict'])
    assert all(a['state_dict'][key].shape==b['state_dict'][key].shape and a['state_dict'][key].dtype==b['state_dict'][key].dtype for key in a['state_dict'])
    checkpoints=[]
    for state,epoch in zip(states,[80,82]):
        step=int(state['global_step']);assert step==epoch*626
        ages={int(value['step']) for value in state['optimizer']['state'].values()};assert ages=={step}
        scheduler=state['scheduler']['after'];assert scheduler['last_epoch']==step
        expected=.0005*.5*(1+math.cos(math.pi*step/62600))
        lrs=[float(group['lr']) for group in state['optimizer']['param_groups']]
        assert all(math.isclose(value,expected,rel_tol=1e-10,abs_tol=1e-12) for value in lrs)
        checkpoints.append(dict(epoch=epoch,step=step,adam_state_count=len(state['optimizer']['state']),all_adam_ages=step,scheduler_last_epoch=scheduler['last_epoch'],learning_rates=lrs,expected_cosine_lr=expected))
    weight_keys=[key for key,value in a['state_dict'].items() if 'weight' in key and value.is_floating_point() and value.ndim>=2 and value.numel()>=1000]
    probes=[]
    for key in [weight_keys[0],weight_keys[len(weight_keys)//2],weight_keys[-1]]:
        x,y=a['state_dict'][key].flatten(),b['state_dict'][key].flatten();stride=max(1,len(x)//262144)
        diff=(y[::stride]-x[::stride]).float()
        probes.append(dict(name=key,sampled_values=len(diff),delta_rms=float(diff.square().mean().sqrt()),delta_abs_max=float(diff.abs().max())))
    assert any(row['delta_rms']>0 for row in probes)
    modules=['runtime/rq24576-sampler-refine-epoch80-20261109/worker.py','runtime/rq24576-tuned-eval-epoch82-20261110/worker.py']
    for name in ['native_settings','generate']:assert tree(modules[0],name)==tree(modules[1],name)
    training=[]
    for line in Path('outputs/imagenet-rq24576-native-scratch100-8h100-20261108/training.log').read_text().splitlines():
        if '"train/loss"' in line:
            try:training.append(json.loads(line))
            except json.JSONDecodeError:pass
    windows=[]
    for epoch in [80,82]:
        rows=[row for row in training if epoch-1<float(row.get('train/epoch',-1))<=epoch and 'train/loss' in row]
        assert rows
        windows.append(dict(epoch=epoch,logged_batches=len(rows),mean_training_loss=statistics.mean(float(row['train/loss']) for row in rows),mean_grad_norm=statistics.mean(float(row['train/grad_norm']) for row in rows)))
    checkpoint_proof=dict(passed=True,checkpoints=checkpoints,architecture_identical=True,training_recipe_identical=True,tokenizer_identical=True,native_generation_function_AST_identical=True,weight_change_probes=probes,training_windows=windows)
    write('training-state-audit.json',checkpoint_proof)
    print(json.dumps(checkpoint_proof),flush=True)
    parent=Path('outputs/imagenet-rq24576-sampler-refine-epoch80-20261109')
    old=json.loads((parent/'report.json').read_text());row80=next(row for row in old['challenger']['results'] if row['seed']==261001)
    parent82=Path('outputs/imagenet-rq24576-tuned-eval-epoch82-20261110');row82=json.loads((parent82/'report.json').read_text())['confirmation'][0]
    assert row80['sampling']==row82['sampling'] and row80['seed']==row82['seed']==261001
    assert row80['reference_sha256']==row82['reference_sha256']==plan['reference_sha256']
    mean,covariance=metrics.load_reference_statistics(base/'inputs/imagenet_256_train.npz')
    recomputed=[]
    for folder,policy,row in [(Path('/tmp/laser-rq24576-sampler-refine-epoch80-20261109'),'t0975_wider_k__s261001',row80),(Path('/tmp/laser-rq24576-tuned-eval-epoch82-20261110'),'refined_t0975_wider',row82)]:
        total=np.zeros(2048);cross=np.zeros((2048,2048));count=0;probabilities=[]
        for rank in range(8):
            with np.load(folder/'confirm'/policy/f'rank-{rank}.npz') as stats:
                total+=stats['total'];cross+=stats['cross'];count+=int(stats['count']);probabilities.append(torch.from_numpy(stats['probabilities'].copy()))
        assert count==50000
        mu=total/count;sigma=(cross-np.outer(total,total)/count)/(count-1)
        fid=metrics.frechet_distance(mean,covariance,mu,sigma)
        probs=torch.cat(probabilities);permutation=torch.randperm(count,generator=torch.Generator().manual_seed(261002))
        score,std=metrics.inception_score(probs[permutation])
        assert abs(fid-row['fid'])<1e-8 and abs(score-row['inception_score'])<1e-5 and abs(std-row['inception_score_std'])<1e-5
        recomputed.append(dict(epoch=row['source_epoch'],fid=fid,inception_score=score,inception_score_std=std,images=count))
    proof=dict(passed=True,matching_sampler=True,matching_generation_seed=True,matching_is_shuffle_seed=261002,matching_reference_statistics=True,metrics_recomputed_from_all_eight_rank_statistics=recomputed)
    write('metric-recalculation-audit.json',proof);print(json.dumps(proof),flush=True)

if __name__=='__main__':main()
