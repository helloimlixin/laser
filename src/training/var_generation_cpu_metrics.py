"""CPU-only FID/IS aggregation; never initializes CUDA or a Torch thread pool."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy import linalg


def numpy_inception_score(probabilities,splits=10,seed=0):
    probabilities=np.asarray(probabilities,dtype=np.float64)
    if probabilities.ndim!=2 or len(probabilities)<splits or not np.isfinite(probabilities).all():raise ValueError('Invalid probabilities')
    if (probabilities<0).any() or not np.allclose(probabilities.sum(1),1.,atol=2e-5):raise ValueError('Probabilities are not normalized')
    order=np.random.default_rng(seed).permutation(len(probabilities));scores=[]
    for split in range(splits):
        part=probabilities[order[split*len(order)//splits:(split+1)*len(order)//splits]]
        marginal=part.mean(0,keepdims=True);logp=np.zeros_like(part);np.log(part,out=logp,where=part>0)
        kl=part*(logp-np.log(np.maximum(marginal,np.finfo(np.float64).tiny)))
        scores.append(np.exp(kl.sum(1).mean()))
    return float(np.mean(scores)),float(np.std(scores,ddof=1))


def frechet_distance(mu1,sigma1,mu2,sigma2):
    difference=mu1-mu2
    try:root=linalg.sqrtm(sigma1.dot(sigma2),disp=False)
    except TypeError:root=linalg.sqrtm(sigma1.dot(sigma2))
    if isinstance(root,tuple):root=root[0]
    if not np.isfinite(root).all():
        offset=np.eye(len(mu1))*1e-6;root=linalg.sqrtm((sigma1+offset).dot(sigma2+offset))
    if np.iscomplexobj(root):
        if not np.allclose(np.diagonal(root).imag,0,atol=1e-3):raise ValueError('Complex FID covariance root')
        root=root.real
    return float(difference.dot(difference)+np.trace(sigma1)+np.trace(sigma2)-2*np.trace(root))


def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--moments',required=True);p.add_argument('--directory',required=True);p.add_argument('--world',type=int,required=True);p.add_argument('--count',type=int,required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    folder=Path(a.directory)
    with np.load(a.reference) as real,np.load(a.moments) as fake:fid=frechet_distance(real['mu'],real['sigma'],fake['mu'],fake['sigma'])
    probabilities=np.concatenate([np.load(folder/f'probabilities-rank{r}.npy') for r in range(a.world)],axis=0)
    if len(probabilities)!=a.count:raise ValueError('Incomplete probability set')
    mean,std=numpy_inception_score(probabilities,seed=a.seed)
    row=dict(fid=fid,inception_score=mean,inception_score_std=std,count=a.count,shuffle_seed=a.seed)
    if not all(np.isfinite(row[k]) for k in ['fid','inception_score','inception_score_std']):raise ValueError('Nonfinite metrics')
    temp=folder/'scores.pending';temp.write_text(json.dumps(row,indent=2)+'\n');temp.replace(folder/'scores.json');print(json.dumps(row),flush=True)
if __name__=='__main__':main()
