"""Paired FID screen: original GMM samples, coarse only, and learned finer pairs.

All variants use the same frozen decoder, metric backend, class labels and
initial coarse samples. Quantized-coarse and coarse-only controls expose gains
from coefficient clipping or simply dropping finer noise. The existing GMM
supplies the coarse marginal through complete ancestral samples.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'runtime')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--codes',type=Path,required=True)
    p.add_argument('--refiner',type=Path,required=True)
    p.add_argument('--teacher',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--variants',nargs='+',choices=['original','original_quantized_coarse','coarse_only','refined'],
        default=['original','original_quantized_coarse','coarse_only','refined'])
    args=p.parse_args()
    import torch
    import torch.distributed as dist
    from torchvision.utils import save_image
    from src.models.coarse_prefix_pair_refiner import CoarsePrefixPairRefiner
    from src.models.typed_pair_rq import TypedPairCodebook
    from src.orthogonal_sparse_codec import ordered_support_basis
    from src.rqvae_metrics import DistributedOriginalRQVAEMetrics
    from src.training.rqtransformer import LaserAux
    rank,world=int(os.environ.get('RANK',0)),int(os.environ.get('WORLD_SIZE',1))
    if world>1:
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        dist.init_process_group('nccl',device_id=torch.device('cuda',int(os.environ['LOCAL_RANK'])))
    device=torch.device('cuda',torch.cuda.current_device())
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    args.output.mkdir(parents=True,exist_ok=True)
    saved=torch.load(args.refiner,map_location='cpu',mmap=True,weights_only=False)
    book=TypedPairCodebook(saved['codebook']['dictionary'],saved['codebook']['coeff_bins'],depth=4).to(device)
    refiner=CoarsePrefixPairRefiner(256,book.dictionary,book.coeff_bins,
        conditional_coefficients=saved['config'].get('decoder_head')=='conditional').to(device).eval()
    refiner.load_state_dict(saved['state_dict'],strict=True);config=saved['config'];del saved
    decoder=LaserAux(args.teacher,16384,len(book.coeff_bins),16.,coeff_scales=[1.,1.,1.,1.],
        coeff_bin_centers=book.coeff_bins.cpu(),sparsity_level=4).to(device).eval()
    torch.testing.assert_close(decoder.dictionary,book.dictionary,atol=0,rtol=0)
    shards=[torch.load(path,map_location='cpu',mmap=True,weights_only=False) for path in sorted(args.codes.glob('rank-*.pt'))]
    atoms=torch.cat([s['atoms'] for s in shards]);gamma=torch.cat([s['gamma'] for s in shards])
    labels=torch.cat([s['labels'] for s in shards]);scales=shards[0]['coeff_scales'].to(device)
    indices=torch.arange(rank,len(atoms),world)
    names=tuple(args.variants)
    metrics={}
    for name in names:
        inception=next(iter(metrics.values())).inception if metrics else None
        metrics[name]=DistributedOriginalRQVAEMetrics(device,compute_inception_score=True,
            reference_stats_path=args.reference,inception=inception)
    generator=torch.Generator(device=device).manual_seed(261032+rank)
    counts=torch.zeros(len(names)+4,device=device,dtype=torch.float64)
    started=time.monotonic()
    with torch.no_grad():
        for first in range(0,len(indices),args.batch_size):
            ids=indices[first:first+args.batch_size]
            a=atoms[ids].to(device);g=gamma[ids].to(device);y=labels[ids].to(device)
            physical=g*scales
            coarse=physical[...,0]
            b=torch.searchsorted((book.coeff_bins[:-1]+book.coeff_bins[1:])/2,coarse.contiguous())
            tokens=book.empty_tokens(len(ids));tokens[...,0]=a[...,0];tokens[...,1]=b+book.num_atoms
            refined=refiner.sample_finer(tokens,book,y,top_k=0,temperature=1.,generator=generator,amp=True)
            basis,_=ordered_support_basis(book.dictionary.T[a])
            original=(basis*physical[...,None]).sum(-2)
            coarse_latent=book.dictionary.T[a[...,0]]*book.coeff_bins[b,None]
            quantized=original+book.dictionary.T[a[...,0]]*(book.coeff_bins[b]-coarse)[...,None]
            latents=dict(original=original,original_quantized_coarse=quantized,
                         coarse_only=coarse_latent,refined=book.reconstruct(refined))
            previews=[]
            for index,name in enumerate(names):
                latent=latents[name]
                with torch.autocast(device_type=device.type,enabled=False):
                    images=decoder.decoder(decoder.post_quant_conv(latent.permute(0,3,1,2).contiguous())).clamp(-1,1)
                counts[index]+=(images.abs()>.99).sum()
                pixels=((images+1)*.5).clamp(0,1)
                metrics[name].update(pixels,real=False)
                if first==0 and rank==0:previews.append(pixels[:12].cpu())
            counts[len(names)]+=images.numel()
            counts[len(names)+1]+=((coarse<book.coeff_bins[0])|(coarse>book.coeff_bins[-1])).sum()
            counts[len(names)+2]+=coarse.numel()
            counts[len(names)+3]+=(book.coeff_bins[b]-coarse).abs().sum()
            if previews:save_image(torch.stack(previews,1).flatten(0,1),args.output/'paired-preview.png',nrow=len(names))
            print(json.dumps(dict(rank=rank,generated=first+len(ids),seconds=time.monotonic()-started)),flush=True)
        results={}
        for name,metric in metrics.items():
            # Identical class-balanced IS splits for every paired variant.
            torch.manual_seed(261034)
            fid,is_mean,is_std=metric.compute()
            results[name]=dict(fid=fid,inception_score=is_mean,inception_score_std=is_std)
    if world>1:dist.all_reduce(counts)
    result=dict(metrics=results,samples=len(atoms),samples_per_class=len(atoms)//1000,
        official_50000_sample_fid=len(atoms)==50000,reference=str(args.reference),
        backend='original-rqvae',decoder=str(args.teacher),refiner=str(args.refiner),config=config,
        source_prior=shards[0]['checkpoint'],shared_coarse_samples=True,
        coarse_coefficient_outside_bin_fraction=float(counts[len(names)+1]/counts[len(names)+2]),
        coarse_coefficient_quantization_mae=float(counts[len(names)+3]/counts[len(names)+2]),
        saturation_fraction={name:float(counts[i]/counts[len(names)]) for i,name in enumerate(metrics)},
        fine_sampling='exact unfiltered joint draws, temperature 1',seconds=time.monotonic()-started)
    if rank==0:
        (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
    if world>1:dist.destroy_process_group()


if __name__=='__main__':main()
