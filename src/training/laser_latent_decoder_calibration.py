"""Small, closed-form decoder-input corrections for a frozen integer codebook."""
import torch
from torch.nn import functional as F


@torch.no_grad()
def latent_regression_statistics(quantized,target,*,kernel=3,images_per_chunk=32):
    if (quantized.shape!=target.shape or quantized.ndim!=4 or kernel not in [1,3]
            or images_per_chunk<1):
        raise ValueError('Expected matching NCHW latents and a supported spatial kernel')
    channels=quantized.shape[1];features=channels*kernel*kernel
    gram=torch.zeros(features,features,device=quantized.device,dtype=torch.float64)
    rhs=torch.zeros(features,channels,device=quantized.device,dtype=torch.float64)
    sums=torch.zeros(features,device=quantized.device,dtype=torch.float64)
    residual_sums=torch.zeros(channels,device=quantized.device,dtype=torch.float64)
    count=0
    for first in range(0,len(quantized),images_per_chunk):
        z=quantized[first:first+images_per_chunk]
        u=F.unfold(z,kernel_size=kernel,padding=kernel//2).transpose(1,2).reshape(-1,features).double()
        residual=(target[first:first+len(z)]-z).permute(0,2,3,1).reshape(-1,channels).double()
        gram.addmm_(u.T,u);rhs.addmm_(u.T,residual)
        sums+=u.sum(0);residual_sums+=residual.sum(0);count+=len(u)
    return dict(gram=gram,rhs=rhs,feature_sum=sums,residual_sum=residual_sums,count=count,
        channels=channels,kernel=kernel)


@torch.no_grad()
def select_center_statistics(statistics):
    channels,kernel=statistics['channels'],statistics['kernel']
    if kernel!=3:raise ValueError('Expected three-by-three source statistics')
    ids=torch.arange(channels,device=statistics['gram'].device)*9+4
    return dict(statistics,gram=statistics['gram'][ids][:,ids],rhs=statistics['rhs'][ids],
        feature_sum=statistics['feature_sum'][ids],kernel=1)


@torch.no_grad()
def solve_latent_correction(statistics,*,ridge_fraction=.01):
    if ridge_fraction<=0:raise ValueError('Positive ridge regularization required')
    n=statistics['count'];mean=statistics['feature_sum']/n;residual_mean=statistics['residual_sum']/n
    gram=statistics['gram']-n*mean[:,None]*mean[None,:]
    rhs=statistics['rhs']-n*mean[:,None]*residual_mean[None,:]
    gram=.5*(gram+gram.T)
    gram.diagonal().add_(ridge_fraction*gram.diagonal().clamp_min(1.))
    factor=torch.linalg.cholesky(gram)
    solution=torch.cholesky_solve(rhs,factor)
    bias=residual_mean-mean@solution
    c,k=statistics['channels'],statistics['kernel']
    weight=solution.T.reshape(c,c,k,k).float()
    assert torch.isfinite(weight).all() and torch.isfinite(bias).all()
    return dict(weight=weight,bias=bias.float(),kernel=k,ridge_fraction=ridge_fraction)


def correct_latent(latent,adapter):
    return latent+F.conv2d(latent,adapter['weight'],adapter['bias'],padding=adapter['kernel']//2)
