#!/usr/bin/env python3
"""Convert a stochastic compound bank to physical coefficients and shared bins.

Preserves every atom support and continuous physical reconstruction. Fits a new
coefficient vocabulary, which must not be silently substituted into a trained
transformer. This tool only prepares and validates assets; it never trains.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import torch


def atomic_json(path, value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def file_sha256(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def physical_coefficients(coefficients, scales):
    scales=torch.as_tensor(scales,dtype=torch.float32,device=coefficients.device)
    if scales.ndim!=1 or scales.numel()!=coefficients.shape[-1]:
        raise ValueError('scale count must equal coefficient depth')
    if not torch.isfinite(scales).all() or (scales<=0).any():
        raise ValueError('scales must be finite and positive')
    result=coefficients.float()*scales
    if not torch.isfinite(result).all():raise ValueError('coefficients must be finite')
    return result


def fit_scalar_lloyd(values, num_bins=2048, iterations=250, progress=None):
    """Exact empirical 1D Lloyd centroid updates, using sorted prefix sums."""
    if num_bins<2 or iterations<1:raise ValueError('invalid Lloyd fit dimensions')
    x=np.array(torch.as_tensor(values).detach().cpu().reshape(-1).numpy(),dtype=np.float64,copy=True)
    if len(x)<num_bins or not np.isfinite(x).all() or x.min()==x.max():
        raise ValueError('fit needs enough finite, nonconstant values')
    x.sort()
    edges=np.linspace(x[0],x[-1],65537)
    cuts=np.searchsorted(x,edges,side='left');cuts[-1]=len(x)
    histogram=np.diff(cuts)
    density=np.cbrt(histogram)
    cdf=np.concatenate(([0.],np.cumsum(density)))
    centers=np.interp((np.arange(num_bins)+.5)*cdf[-1]/num_bins,cdf,edges)
    prefix=np.empty(len(x)+1,dtype=np.float64);prefix[0]=0
    np.cumsum(x,dtype=np.float64,out=prefix[1:])
    second=float(np.dot(x,x))
    def statistics(c):
        boundaries=(c[:-1]+c[1:])*.5
        edges=np.concatenate(([0],np.searchsorted(x,boundaries,side='right'),[len(x)]))
        counts=np.diff(edges);sums=prefix[edges[1:]]-prefix[edges[:-1]]
        distortion=(second-2*np.dot(c,sums)+np.dot(c*c,counts))/len(x)
        return counts,sums,max(float(distortion),0.)
    trace=[];previous=math.inf
    for iteration in range(iterations):
        counts,sums,error=statistics(centers)
        updated=np.divide(sums,counts,out=centers.copy(),where=counts>0)
        if not np.all(np.diff(updated)>0):raise ValueError('Lloyd centers lost strict order')
        if error>previous+1e-9:raise ValueError('Lloyd distortion increased')
        movement=float(np.max(np.abs(updated-centers)))
        centers=updated;previous=error
        row=dict(iteration=iteration,physical_coefficient_mse=error,max_movement=movement,empty_bins=int((counts==0).sum()))
        trace.append(row)
        if progress and iteration%25==0:progress('fitting_codebook',**row)
        if movement<1e-7:break
    result=torch.tensor(centers,dtype=torch.float32)
    if not torch.all(result[1:]>result[:-1]):raise ValueError('FP32 centers lost strict order')
    return result,dict(fit_values=len(x),iterations=len(trace),trace=trace,
        final_physical_coefficient_mse=statistics(result.double().numpy())[2],
        fit_min=float(x[0]),fit_max=float(x[-1]),objective='unweighted physical coefficient squared error',
        initialization='density^(1/3), 65536 histogram cells; exact empirical centroid updates')


def nearest_centers(values, centers):
    if centers.ndim!=1 or not torch.all(centers[1:]>centers[:-1]):
        raise ValueError('centers must be strictly ordered')
    ids=torch.bucketize(values.contiguous(),(centers[:-1]+centers[1:])*.5)
    return centers[ids],ids


@torch.no_grad()
def evaluate_quantization(atoms, normalized, scales, dictionary, old_centers, new_centers, batch=8, progress=None):
    k=atoms.shape[-1];channels=dictionary.shape[1]
    accum={name:dict(coefficient_squared=torch.zeros(k,dtype=torch.float64),latent_squared=0.,
                    count_sites=0,outside=torch.zeros(k,dtype=torch.int64),hist=torch.zeros(k,len(new_centers),dtype=torch.int64))
           for name in ['depth_normalized','physical_shared']}
    scales=torch.as_tensor(scales,dtype=torch.float32)
    max_conversion_error=0.
    for start in range(0,len(atoms),batch):
        a=atoms[start:start+batch].long();n=normalized[start:start+batch].float()
        raw=physical_coefficients(n,scales)
        max_conversion_error=max(max_conversion_error,float((raw-n*scales).abs().max()))
        old,oi=nearest_centers(n,old_centers);old=old*scales
        new,ni=nearest_centers(raw,new_centers)
        active=dictionary[a].double()
        for name,decoded,ids,values,centers in [('depth_normalized',old,oi,n,old_centers),('physical_shared',new,ni,raw,new_centers)]:
            error=decoded.double()-raw.double()
            latent=(active*error[...,None]).sum(-2)
            out=accum[name]
            out['coefficient_squared']+=error.reshape(-1,k).square().sum(0)
            out['latent_squared']+=float(latent.square().sum())
            out['count_sites']+=error.numel()//k
            out['outside']+=((values<centers[0])|(values>centers[-1])).reshape(-1,k).sum(0)
            for depth in range(k):out['hist'][depth]+=torch.bincount(ids[...,depth].flatten(),minlength=len(centers))
        if progress and (start%512==0 or start+batch>=len(atoms)):
            progress('evaluating_quantization',images=min(start+batch,len(atoms)),total=len(atoms))
    result={}
    for name,a in accum.items():
        prob=a['hist'].double()/a['count_sites']
        result[name]=dict(physical_coefficient_mse=float(a['coefficient_squared'].sum()/(a['count_sites']*k)),
            physical_coefficient_mse_by_depth=(a['coefficient_squared']/a['count_sites']).tolist(),
            added_latent_mse_vs_continuous=a['latent_squared']/(a['count_sites']*channels),
            outside_center_range_fraction_by_depth=(a['outside']/a['count_sites']).tolist(),
            occupied_bins_by_depth=(a['hist']>0).sum(1).tolist(),
            empirical_token_entropy_nats_by_depth=(-(prob*prob.clamp_min(1e-300).log()).sum(1)).tolist())
    # Same physical-temperature rule as the active training codec.
    probe=normalized[:16].reshape(-1,k)[::16][:1024].float()
    physical=probe*scales
    for name in result:
        entropy=[]
        for d in range(k):
            bins=old_centers*scales[d] if name=='depth_normalized' else new_centers
            probs=(-(physical[:,d,None]-bins).square()/.125).softmax(-1)
            entropy.append(float(-(probs*probs.clamp_min(1e-30).log()).sum(-1).mean()))
        result[name]['soft_target_entropy_nats_by_depth_at_tau_0125']=entropy
    return dict(images=len(atoms),coefficient_pairs=atoms.numel(),metrics=result,
                continuous_coefficient_conversion_max_error=max_conversion_error)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--validation-components',type=Path)
    parser.add_argument('--heldout-images',type=int,default=4096)
    parser.add_argument('--bins',type=int,default=2048)
    parser.add_argument('--iterations',type=int,default=250)
    parser.add_argument('--threads',type=int,default=16)
    args=parser.parse_args()
    torch.set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    output=args.output_dir/'compound-cache-physical.pt'
    if output.exists() or output.resolve()==args.input.resolve():raise ValueError('refusing to overwrite a cache')
    started=time.time()
    def progress(phase,**extra):
        row=dict(phase=phase,elapsed_seconds=time.time()-started,**extra)
        atomic_json(args.output_dir/'status.json',row);print(json.dumps(row),flush=True)
    progress('loading_source')
    source=torch.load(args.input,map_location='cpu',weights_only=True,mmap=True)
    meta=dict(source['meta']);atoms=source['atoms'];normalized=source['coeffs']
    if meta.get('format')!='laser_stochastic_compound_bank_v1' or atoms.shape!=normalized.shape or atoms.ndim!=5:
        raise ValueError('expected a paired stochastic compound bank')
    if not 0<args.heldout_images<len(atoms):raise ValueError('invalid held-out image count')
    source_sha=file_sha256(args.input)
    scales=meta['coeff_scales'];old_centers=torch.tensor(meta['coeff_bin_centers'],dtype=torch.float32)
    if len(old_centers)!=args.bins:raise ValueError('preserve coefficient vocabulary size for this comparison')
    progress('converting_to_physical_coefficients')
    physical=physical_coefficients(normalized,scales)
    progress('sorting_fit_values',fit_images=len(atoms)-args.heldout_images,fit_values=physical[args.heldout_images:].numel())
    centers,fit=fit_scalar_lloyd(physical[args.heldout_images:],args.bins,args.iterations,progress)
    atomic_json(args.output_dir/'fit.json',fit)
    torch.save(centers,args.output_dir/'coefficient-centers.pt')
    state=torch.load(meta['stage1_checkpoint'],map_location='cpu',weights_only=False,mmap=True)['state_dict']
    key='quantizer.dictionary' if 'quantizer.dictionary' in state else 'bottleneck.dictionary'
    dictionary=torch.nn.functional.normalize(state[key].float(),dim=0).T.contiguous();del state
    progress('evaluating_heldout_train')
    heldout=evaluate_quantization(atoms[:args.heldout_images],normalized[:args.heldout_images],
        scales,dictionary,old_centers,centers,progress=progress)
    atomic_json(args.output_dir/'heldout-train.json',heldout)
    validation=None
    if args.validation_components:
        val=torch.load(args.validation_components,map_location='cpu',weights_only=True,mmap=True)
        progress('evaluating_validation')
        validation=evaluate_quantization(val['atoms'],val['coeffs'],scales,dictionary,old_centers,centers,progress=progress)
        atomic_json(args.output_dir/'validation.json',validation)
    units=dict(parent_bank_identity=meta['bank_identity'],source_cache_sha256=source_sha,
        coefficient_units='physical_omp_least_squares',coefficient_normalization='none',
        coeff_scales=[1.]*atoms.shape[-1],coeff_bin_centers_sha256=hashlib.sha256(centers.numpy().tobytes()).hexdigest())
    identity=hashlib.sha256(json.dumps(units,sort_keys=True).encode()).hexdigest()
    meta['source_encoding_precision']=meta.get('encoding_precision')
    meta['source_auto_coeff_scales_percentile']=meta.pop('auto_coeff_scales_percentile',None)
    meta.update(units,bank_identity=identity,coeff_scale=1.,coeff_max=float(physical.abs().max()),
        cache_revision=4,encoding_precision='Existing FP32 stochastic OMP bank; FP32 recovery of physical coefficients; no normalization',
        coeff_vocab_size=args.bins,coeff_bin_centers=centers.tolist(),source_coeff_scales=scales,
        coefficient_quantizer='shared_lloyd_max_physical',coefficient_storage='FP32 physical coefficients; no depth normalization or clipping',
        coefficient_fit_method=fit['objective'],coefficient_fit_images=[args.heldout_images,len(atoms)],
        coefficient_fit_variants='all cached variants at all training locations',
        source_coefficient_quantizer=source['meta'].get('coefficient_quantizer'),
        clip_coefficients=False,requires_new_stage2_training=True,coefficient_bins_changed=True,
        coefficient_preparation_source_sha256=file_sha256(__file__))
    progress('saving_physical_cache')
    temporary=output.with_suffix('.pt.tmp')
    torch.save(dict(atoms=atoms,coeffs=physical,labels=source['labels'],meta=meta),temporary);temporary.replace(output)
    loaded=torch.load(output,map_location='cpu',weights_only=True,mmap=True)
    assert torch.equal(loaded['atoms'],atoms) and torch.equal(loaded['labels'],source['labels'])
    assert torch.equal(loaded['coeffs'],physical) and loaded['meta']['coeff_scales']==[1.]*atoms.shape[-1]
    assert identity!=source['meta']['bank_identity']
    receipt=dict(passed=True,cache=str(output.resolve()),cache_sha256=file_sha256(output),bytes=output.stat().st_size,
        source_cache=str(args.input.resolve()),source_cache_sha256=source_sha,bank_identity=identity,
        images=len(atoms),variants_per_site=atoms.shape[-2],atoms_unchanged=True,labels_unchanged=True,
        source_coeff_scales=scales,coeff_scales=meta['coeff_scales'],coefficient_units=meta['coefficient_units'],
        coefficient_bins=args.bins,coeff_max=meta['coeff_max'],fit=fit,
        heldout=heldout,validation=validation,requires_new_stage2_training=True,
        training_launched=False,elapsed_seconds=time.time()-started)
    atomic_json(args.output_dir/'complete.json',receipt)
    atomic_json(args.output_dir/'codec-overrides.json',dict(token_cache=str(output.resolve()),
        checkpoint=meta['stage1_checkpoint'],coeff_scale=1.,coeff_scales=meta['coeff_scales'],coeff_max=meta['coeff_max'],
        coeff_vocab_size=args.bins,coeff_target_temperature=.125,resume=False,resume_checkpoint=None,init_stage2_checkpoint=None))
    progress('complete',cache=str(output.resolve()),passed=True,training_launched=False)

if __name__=='__main__':main()
