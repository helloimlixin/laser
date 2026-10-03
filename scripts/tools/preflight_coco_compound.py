"""Full-size four-GPU integration check; no resulting weights enter training."""
import argparse
from contextlib import nullcontext
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import OmegaConf
from src.training.cc3m_compound import build_model, make_aux, objective, generate, load_cache
from scripts.tools.build_cc3m_compound_cache import text_tokenizer, write_json
from src.training.rqtransformer import gather_rank_rng_states


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    options = OmegaConf.to_container(OmegaConf.load(args.config).options, resolve=True)
    rank, world, local = [int(os.environ[k]) for k in ('RANK','WORLD_SIZE','LOCAL_RANK')]
    torch.cuda.set_device(local); torch.set_num_threads(4); torch.manual_seed(71)
    torch.set_float32_matmul_precision('high')
    dist.init_process_group('nccl')
    device = torch.device('cuda',local)
    data = load_cache(options['token_cache'], options['stage1_sha256'])
    aux = make_aux(options, data['meta']['coeff_scales'], device)
    prior = build_model(options).to(device)
    model = DDP(prior, device_ids=[local], broadcast_buffers=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=options['lr'], betas=(.9,.95), weight_decay=1e-4, fused=True)
    tokenizer = text_tokenizer(.1)
    started = time.monotonic()
    for micro in range(options['accumulation']):
        begin = (micro*world+rank)*options['batch_size']
        rows = torch.arange(begin, begin+options['batch_size']) % len(data['atoms'])
        atoms, coeffs = data['atoms'][rows].long().to(device), data['coeffs'][rows].to(device)
        text = torch.tensor([r.ids for r in tokenizer.encode_batch([data['captions'][i] for i in rows])], device=device)
        with (nullcontext() if micro+1 == options['accumulation'] else model.no_sync()), torch.autocast('cuda',dtype=torch.bfloat16):
            loss, _ = objective(model, aux, atoms, coeffs, text, 5., options['accumulation'])
            loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    probe = prior.cond_classifier.linear.weight.detach().float().sum()
    probes = [torch.zeros_like(probe) for _ in range(world)]
    dist.all_gather(probes, probe)
    assert all(torch.equal(probes[0], x) for x in probes)
    rng = gather_rank_rng_states(device)
    row = torch.tensor([float(loss.detach())*options['accumulation'], float(grad),
        torch.cuda.max_memory_allocated()/2**30, time.monotonic()-started], device=device)
    rows = [torch.zeros_like(row) for _ in range(world)]
    dist.all_gather(rows, row)
    assert all(torch.isfinite(x).all() for x in rows)
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        path = Path('/mnt/laser-coco/preflight-roundtrip.pt')
        torch.save(dict(model=prior.state_dict(), optimizer=optimizer.state_dict()), path)
        state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        prior.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        assert state['optimizer']['state']
        prior.eval()
        pixels = generate(prior, aux, text[:2], options)
        assert pixels.shape == (2,3,256,256) and torch.isfinite(pixels).all()
        from src.rqvae_metrics import OriginalRQVAEInception
        import clip
        from PIL import Image
        from torchvision.utils import save_image
        save_image(pixels, args.output/'integration-samples.png')
        with torch.inference_mode():
            inception = OriginalRQVAEInception().to(device).eval()
            features, _ = inception(pixels)
            assert torch.isfinite(features).all()
            del inception
            clip_model, preprocess = clip.load('ViT-B/32', device=device)
            images = torch.stack([preprocess(Image.fromarray((x.permute(1,2,0).cpu().numpy()*255).astype('uint8'))) for x in pixels]).to(device)
            score = torch.nn.functional.cosine_similarity(clip_model.encode_image(images).float(),
                clip_model.encode_text(clip.tokenize(['a dog','a bus']).to(device)).float())
            assert torch.isfinite(score).all()
        write_json(args.output/'stage2.json', dict(passed=True, world_size=world,
            parameters=sum(p.numel() for p in prior.parameters()), synchronized_parameters=True,
            rng_states=len(rng), total_batch_size=world*options['batch_size']*options['accumulation'],
            ranks=[x.tolist() for x in rows], columns=['loss','gradient_norm','peak_gib','seconds'],
            optimizer_reload=True, conditioned_generation=True, clip_and_inception=True,
            stage1_sha256=options['stage1_sha256'], production_weights_used=False))
        del state
        path.unlink()
    dist.barrier(); dist.destroy_process_group()


if __name__ == '__main__':
    main()
