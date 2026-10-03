"""Exercise real CC3M tokens with the production model, optimizer, and samplers."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
from omegaconf import OmegaConf
from src.training.cc3m_compound import build_model, make_aux, objective, generate
from scripts.tools.build_cc3m_compound_cache import write_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--cache-shard',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    options=OmegaConf.to_container(OmegaConf.load(args.config).options,resolve=True)
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(567)
    device=torch.device('cuda',0)
    shard=torch.load(args.cache_shard,weights_only=True,map_location='cpu')
    scales=torch.tensor(shard['meta']['coeff_abs_max'])/3
    aux=make_aux(options,scales.tolist(),device)
    model=build_model(options).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=options['lr'],betas=(.9,.95),weight_decay=1e-4,fused=True)
    args.output.mkdir(parents=True,exist_ok=True)
    batch=options['batch_size']
    atoms=shard['atoms'][:batch].long().to(device)
    coefficients=(shard['coeffs'][:batch]/scales).to(device)
    text=shard['text_ids'][:batch].long().to(device)
    rows=[]
    for step in range(2):
        started=time.monotonic()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            loss,details=objective(model,aux,atoms,coefficients,text,5.)
            loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        row=dict(step=step,loss=float(loss.detach()),gradient_norm=float(grad),seconds=time.monotonic()-started)
        assert all(torch.isfinite(torch.tensor(v)) for v in row.values())
        rows.append(row)
        print(json.dumps(row),flush=True)
    local_output=Path('/mnt/laser-cc3m/preflight')
    local_output.mkdir(parents=True,exist_ok=True)
    checkpoint=local_output/'roundtrip.pt'
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),options=options),checkpoint)
    restored=torch.load(checkpoint,weights_only=True,map_location='cpu',mmap=True)
    for name,value in restored['model'].items():
        if value.is_floating_point(): assert torch.isfinite(value).all(),name
    assert len(restored['optimizer']['state'])>0
    model.load_state_dict(restored['model'],strict=True)
    optimizer.load_state_dict(restored['optimizer'])
    model.eval()
    pixels=generate(model,aux,text[:4],options)
    assert pixels.shape==(4,3,256,256) and torch.isfinite(pixels).all()
    from torchvision.utils import save_image
    save_image(pixels,args.output/'integration-samples.png',nrow=2)
    from src.rqvae_metrics import OriginalRQVAEInception
    import clip
    from PIL import Image
    inception=OriginalRQVAEInception().to(device).eval()
    with torch.inference_mode():
        features,_=inception(pixels)
        assert features.shape==(4,2048) and torch.isfinite(features).all()
        del inception
        clip_model,preprocess=clip.load('ViT-B/32',device=device)
        images=torch.stack([preprocess(Image.fromarray((x.permute(1,2,0).cpu().numpy()*255).astype('uint8'))) for x in pixels]).to(device)
        scores=torch.nn.functional.cosine_similarity(clip_model.encode_image(images).float(),
            clip_model.encode_text(clip.tokenize(shard['captions'][:4],truncate=True).to(device)).float())
        assert torch.isfinite(scores).all()
    report=dict(passed=True,parameters=sum(p.numel() for p in model.parameters()),
        batch_size=batch,latent_shape=list(atoms.shape[1:]),text_shape=list(text.shape),
        optimizer_steps=rows,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        checkpoint_bytes=checkpoint.stat().st_size,model_optimizer_reload=True,
        generated_images=4,original_inception_loaded=True,clip_vit_b32_loaded=True,
        source_shard=str(args.cache_shard),purpose='Integration only; fresh random weights used for production')
    write_json(args.output/'complete.json',report)
    del restored
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
