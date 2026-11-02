"""Controlled fixed-length integer-token rate/distortion study.

Train/heldout fitting images are disjoint subsets of ImageNet training.
Full validation is used only for reporting reconstruction FID and rates.
The source image CNNs stay frozen throughout. No stage-2 production run starts.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from src.models.joint_rate_distortion import JointVectorBook, NeuralJointDecoder, rate_statistics
from src.models.laser_joint_tokenizer import FrozenJointLaserTokenizer, JointAtomCoefficientRQ
from src.training.rq_imagenet_runtime import TarImageNet
from src import rqvae_metrics as metrics
from src.models.transformer_core import TransformerBlock


def write(base, name, data):
    path = base / name; path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def progress(args, phase, **data):
    value = dict(phase=phase, unix=time.time(), **data)
    write(args.base, f'{phase}-progress-rank{args.rank}.json', value)
    print(json.dumps(value), flush=True)


def image_transform(train):
    return transforms.Compose([transforms.Resize(256),
        transforms.RandomCrop(256) if train else transforms.CenterCrop(256),
        *([transforms.RandomHorizontalFlip()] if train else []),
        transforms.ToTensor(), transforms.Normalize([.5]*3, [.5]*3)])


def tokenizer(args):
    return FrozenJointLaserTokenizer(args.base/'inputs/stage1.pt',
        args.base/'inputs/joint65537.pt').to(args.device).eval().requires_grad_(False)


def load_cache(args, split):
    parts = [torch.load(args.base/f'cache/{split}-rank{r}.pt', map_location='cpu', weights_only=True)
             for r in range(8)]
    return {k:torch.cat([p[k] for p in parts]) for k in parts[0]}


@torch.inference_mode()
def cache(args):
    aux = tokenizer(args)
    train = TarImageNet(args.base/'data', 'train', image_transform(True), seed=261001)
    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(261024))
    selections = {'fit':order[:args.fit_images].tolist(),
                  'held':order[args.fit_images:args.fit_images+args.held_images].tolist()}
    if args.rank == 0:
        write(args.base, 'calibration-data.json', dict(
            fit_indices=selections['fit'], heldout_training_indices=selections['held'],
            source_stage1_sha256='dd28db9306d526bdc9fbf8016403e106c4f317fd8f5c04792f0953e53310fdab',
            training_view='seed261001 epoch0 Resize256 RandomCrop256 RandomHorizontalFlip',
            validation_view='Resize256 CenterCrop256', validation_used_for_fitting=False,
            tokens_per_image=256, geometry=[8,8,4]))
    for split, indices in selections.items():
        path = args.base/f'cache/{split}-rank{args.rank}.pt'
        if path.exists():continue
        local = indices[args.rank::args.world]
        loader = DataLoader(Subset(train, local), batch_size=32, num_workers=4,
                            pin_memory=True)
        zs, labels = [], []
        for i, (images, y) in enumerate(loader):
            z = aux.encode(images.to(args.device, non_blocking=True))
            zs.append(z.cpu()); labels.append(y)
            if i % 16 == 0:progress(args, 'cache', split=split, local_images=sum(len(v) for v in zs))
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(latents=torch.cat(zs), labels=torch.cat(labels),
                        indices=torch.tensor(local)), path)
    path = args.base/f'cache/validation-rank{args.rank}.pt'
    if not path.exists():
        inception = metrics.OriginalRQVAEInception().to(args.device).eval().requires_grad_(False)
        data = TarImageNet(args.base/'data', 'val', image_transform(False))
        local = list(range(args.rank, len(data), args.world))
        loader = DataLoader(Subset(data, local), batch_size=32, num_workers=4, pin_memory=True)
        zs, fs, labels = [], [], []
        for i, (images, y) in enumerate(loader):
            images = images.to(args.device, non_blocking=True)
            zs.append(aux.encode(images).cpu())
            features, _ = inception(images.mul(.5).add(.5).clamp(0, 1))
            fs.append(features.cpu()); labels.append(y)
            if i % 25 == 0:progress(args, 'cache', split='validation', local_images=sum(len(v) for v in zs))
        torch.save(dict(latents=torch.cat(zs), features=torch.cat(fs), labels=torch.cat(labels),
                        indices=torch.tensor(local)), path)
    dist.barrier()
    if args.rank == 0:write(args.base, 'cache-complete.json', dict(completed=True,
        fit_images=args.fit_images, heldout_training_images=args.held_images,
        validation_images=50000, encoder_precision='fp32', teacher_tf32=False))


def legacy(args, name):
    s = torch.load(args.base/f'inputs/{name}.pt', map_location='cpu', weights_only=True)
    return JointAtomCoefficientRQ(s['dictionary'], s['atom_ids'], s['coefficients']).to(args.device)


@torch.no_grad()
def initialize_book(args):
    prior = legacy(args, 'joint65537')
    counts = torch.load(args.base/'inputs/training-code-counts.pt', weights_only=True).sum(0)
    # Retain the most frequent old joint IDs, including exactly one null ID.
    ids = torch.cat((torch.zeros(1, dtype=torch.long), counts[1:].topk(args.vocabulary-1).indices + 1))
    vectors = torch.stack([prior.embed_depth(ids.to(args.device), d) for d in range(4)])
    return JointVectorBook(vectors).to(args.device)


def fit(args):
    fit = load_cache(args, 'fit'); held = load_cache(args, 'held')
    x, h = fit['latents'].to(args.device), held['latents'].to(args.device)
    path = args.base/f'books/vq{args.vocabulary}.pt'
    if not path.exists():
        q = initialize_book(args); best, best_error = None, math.inf
        trace = []
        for iteration in range(args.passes+1):
            encoded = q.quantize(x)
            held_error = float((h-q.quantize(h)['quantized']).square().mean())
            row = dict(iteration=iteration, fit_mse=float((x-encoded['quantized']).square().mean()),
                       heldout_training_mse=held_error)
            if held_error < best_error:
                best_error = held_error; best = q.vectors.detach().cpu().clone()
            if iteration < args.passes:row.update(q.refit(x, encoded['codes']))
            trace.append(row); progress(args, 'fit', vocabulary=args.vocabulary, **row)
        path.parent.mkdir(exist_ok=True)
        torch.save(dict(vectors=best, trace=trace, vocabulary=args.vocabulary,
            method='free joint vectors; alternating greedy residual coding and final-sum LS',
            heldout_training_mse=best_error, image_cnns_frozen=True,
            depth_is_implicit=True, tokens_per_image=256), path)
    spec = torch.load(path, map_location='cpu', weights_only=True)
    q = JointVectorBook(spec['vectors']).to(args.device)
    code_path = args.base/f'books/vq{args.vocabulary}-training-codes.pt'
    with torch.no_grad():
        fit_codes = q.quantize(x)['codes']; held_codes = q.quantize(h)['codes']
    torch.save(dict(fit_codes=fit_codes.cpu(), held_codes=held_codes.cpu(),
        fit_labels=fit['labels'], held_labels=held['labels']), code_path)
    neural_path = args.base/f'books/neural{args.vocabulary}.pt'
    if neural_path.exists():return
    torch.manual_seed(261024)
    learner = NeuralJointDecoder().to(args.device)
    optimizer = torch.optim.AdamW(learner.parameters(), lr=3.e-4, weight_decay=1.e-4)
    rng = torch.Generator().manual_seed(261025)
    best, best_error, trace = None, math.inf, []
    @torch.no_grad()
    def assess():
        total = 0.
        for start in range(0, len(h), 64):
            target = h[start:start+64]
            prediction = learner(q.embed(held_codes[start:start+64]))
            total += float((prediction-target).double().square().sum())
        return total / h.numel()
    for step in range(args.neural_steps+1):
        if step in [0, 100, 300, 600, args.neural_steps]:
            error = assess()
            trace.append(dict(step=step, heldout_training_mse=error))
            if error < best_error:
                best_error = error; best = {k:v.detach().cpu().clone() for k,v in learner.state_dict().items()}
            progress(args, 'neural', vocabulary=args.vocabulary, step=step, heldout_training_mse=error)
        if step == args.neural_steps:break
        indices = torch.randint(len(x), (64,), generator=rng).to(args.device)
        optimizer.zero_grad(set_to_none=True)
        prediction = learner(q.embed(fit_codes[indices]))
        loss = (prediction-x[indices]).square().mean()
        loss.backward(); torch.nn.utils.clip_grad_norm_(learner.parameters(), 1.)
        optimizer.step()
    torch.save(dict(state_dict=best, vocabulary=args.vocabulary, trace=trace,
        selected_by='heldout ImageNet training latent MSE; no validation fitting',
        heldout_training_mse=best_error, decoder_parameters=sum(p.numel() for p in learner.parameters()),
        objective='latent MSE', continuous_side_information=False,
        image_cnns_frozen=True, book=str(path)), neural_path)
    progress(args, 'fit_complete', vocabulary=args.vocabulary, neural_heldout_mse=best_error)


@torch.inference_mode()
def evaluate(args):
    aux = tokenizer(args)
    inception = metrics.OriginalRQVAEInception().to(args.device).eval().requires_grad_(False)
    local = torch.load(args.base/f'cache/validation-rank{args.rank}.pt', map_location='cpu', weights_only=True)
    candidates = ['joint65537', 'joint49153']
    if (args.base/'inputs/joint32769.pt').exists():candidates.append('joint32769')
    candidates += [name for v in args.vocabularies for name in (f'vq{v}', f'neural{v}')]
    if args.candidates:candidates=args.candidates
    real = local['features'].to(args.device, dtype=torch.float64)
    validation = TarImageNet(args.base/'data', 'val', image_transform(False))
    image_loader = DataLoader(Subset(validation, local['indices'].tolist()),
        batch_size=args.batch_size, num_workers=4, pin_memory=True, persistent_workers=True)
    result_path = args.base/args.results_file
    results = json.loads(result_path.read_text())['results'] if result_path.exists() else []
    completed = {r['name'] for r in results if r.get('decoder_tf32',False) == args.decoder_tf32}
    for name in candidates:
        if name in completed:continue
        if name.startswith('joint'):
            q = legacy(args, name); decoder = None
        else:
            v = int(name.removeprefix('neural').removeprefix('perceptual').removeprefix('vq'))
            spec = torch.load(args.base/f'books/vq{v}.pt', map_location='cpu', weights_only=True)
            q = JointVectorBook(spec['vectors']).to(args.device); decoder = None
            if name.startswith(('neural','perceptual')):
                decoder = NeuralJointDecoder().to(args.device).eval()
                spec = torch.load(args.base/f'books/{name}.pt', map_location='cpu', weights_only=True)
                decoder.load_state_dict(spec['state_dict'])
        state = metrics.DistributedOriginalRQVAEMetrics(args.device, inception=inception,
                                                       compute_inception_score=False)
        state.real_sum.copy_(real.sum(0)); state.real_cross.addmm_(real.T, real)
        state.real_count.fill_(len(real))
        counts = torch.zeros(4, q.vocab_size, dtype=torch.float64, device=args.device)
        error = torch.zeros(4, dtype=torch.float64, device=args.device)
        tokens = []
        start_time = time.monotonic()
        source_codes = None
        if name.startswith(('neural','perceptual')):
            saved = torch.load(args.base/f'cache/vq{v}-validation-codes-rank{args.rank}.pt',
                               map_location='cpu', weights_only=True)
            torch.testing.assert_close(saved['indices'], local['indices'], rtol=0, atol=0)
            source_codes = saved['codes']
        for batch, (images, labels) in enumerate(image_loader):
            first = batch * args.batch_size
            x = local['latents'][first:first+len(images)].to(args.device)
            if source_codes is None:
                encoded = q.quantize(x, chunk_size=1024)
                codes, z = encoded['codes'], encoded['quantized']
            else:
                codes = source_codes[first:first+len(x)].to(args.device, dtype=torch.long)
                z = q.embed(codes).sum(-2)
            if decoder is not None:z = decoder(q.embed(codes))
            # Only image reconstruction may use TF32. Cached encoder latents,
            # nearest-code decisions, and the reference Inception stay strict FP32.
            previous_tf32 = torch.backends.cudnn.allow_tf32
            try:
                torch.backends.cudnn.allow_tf32 = args.decoder_tf32
                reconstructed = aux.decode(z).mul(.5).add(.5).clamp(0, 1)
            finally:
                torch.backends.cudnn.allow_tf32 = previous_tf32
            state.update(reconstructed, real=False)
            reference = images.to(args.device, non_blocking=True).mul(.5).add(.5).clamp(0, 1)
            error[2].add_((reconstructed-reference).double().square().sum()); error[3].add_(reference.numel())
            for d in range(4):counts[d].add_(torch.bincount(codes[..., d].flatten(), minlength=q.vocab_size))
            error[0].add_((x-z).double().square().sum()); error[1].add_(x.numel())
            tokens.append(codes.cpu().to(torch.int32))
            if first % (args.batch_size*50) == 0:
                progress(args, 'evaluate', candidate=name, local_images=first+len(x), seconds=time.monotonic()-start_time)
        dist.all_reduce(counts); dist.all_reduce(error)
        fid, _, _ = state.compute()
        assert int(state.real_count) == int(state.fake_count) == 50000
        row = dict(name=name, rfid=fid, latent_mse=float(error[0]/error[1]),
            pixel_mse=float(error[2]/error[3]), psnr_db=float(-10*torch.log10(error[2]/error[3])),
            real_images=50000, reconstructed_images=50000, **rate_statistics(counts),
            metric_backend='original-rqvae', reference='same full ImageNet validation 50000',
            neural_decoder=decoder is not None,
            decoder_tf32=args.decoder_tf32, encoder_and_metric_tf32=False,
            neural_decoder_parameters=sum(p.numel() for p in decoder.parameters()) if decoder else 0,
            tokens_per_image_and_image_cnns_fixed=True, continuous_side_information=False,
            identical_codes_to=f'vq{v}' if name.startswith(('neural','perceptual')) else None,
            seconds=time.monotonic()-start_time)
        results.append(row)
        if name == 'joint65537' and args.decoder_tf32:
            strict_reference = 4.424879499474287
            strict_path = args.base/'strict-control.json'
            if strict_path.exists():
                strict_reference = json.loads(strict_path.read_text())['results'][0]['rfid']
            drift = abs(fid - strict_reference)
            if args.rank == 0:
                write(args.base, 'decoder-precision-calibration.json', dict(
                    strict_fp32_reference_rfid=strict_reference, decoder_tf32_rfid=fid,
                    absolute_rfid_difference=drift, maximum_accepted_difference=.01,
                    validation_images=50000, encoder_and_inception_strict_fp32=True,
                    passed=drift < .01))
            if drift >= .01:raise RuntimeError('TF32 decoder failed full-validation precision calibration')
        torch.save(dict(codes=torch.cat(tokens), labels=local['labels'], indices=local['indices']),
                   args.base/f'cache/{name}-validation-codes-rank{args.rank}.pt')
        dist.barrier()
        if args.rank == 0:
            write(args.base, args.results_file, dict(completed=len(results)==len(candidates), results=results))
            print(json.dumps(row), flush=True)
        del q, decoder, state
        torch.cuda.empty_cache()


def probe(args):
    """Matched small causal priors measure prediction cost, not generated FID."""
    name = args.candidate
    path = args.base/f'books/{name}-training-codes.pt'
    if not path.exists():
        fit, held = load_cache(args, 'fit'), load_cache(args, 'held')
        q = legacy(args, name)
        with torch.no_grad():
            data = dict(fit_codes=q.quantize(fit['latents'].to(args.device))['codes'].cpu(),
                held_codes=q.quantize(held['latents'].to(args.device))['codes'].cpu(),
                fit_labels=fit['labels'], held_labels=held['labels'])
        torch.save(data, path); del q, fit, held
    data = torch.load(path, map_location='cpu', weights_only=True)
    if args.prepare_only:return
    vocabulary = int(name.removeprefix('joint').removeprefix('vq'))
    class Prior(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocabulary, 256)
            self.classes = nn.Embedding(1000, 256)
            self.positions = nn.Embedding(256, 256)
            self.blocks = nn.ModuleList([TransformerBlock(256,8,1024,.1,
                attention_dropout=0., ffn_intermediate_dropout=0.) for _ in range(4)])
            self.norm = nn.LayerNorm(256)
            for name, module in self.named_modules():
                if isinstance(module, (nn.Linear,nn.Embedding)):
                    # Output cardinality must not perturb shared-body initialization.
                    module_seed = 261026 + int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], 'little')
                    torch.manual_seed(module_seed)
                    nn.init.normal_(module.weight, std=.02)
                    if getattr(module,'bias',None) is not None:nn.init.zeros_(module.bias)
        def forward(self, codes, labels):
            h = torch.cat((self.classes(labels)[:,None], self.embedding(codes[:,:-1])),1)
            h = h + self.positions(torch.arange(256,device=h.device))
            for block in self.blocks:h = block(h)[0]
            return torch.nn.functional.linear(self.norm(h),self.embedding.weight)
    torch.manual_seed(261026)
    model = Prior().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5.e-4, betas=(.9,.95),weight_decay=.01,fused=True)
    fit = data['fit_codes'].flatten(1).long().to(args.device)
    held = data['held_codes'].flatten(1).long().to(args.device)
    labels = data['fit_labels'].to(args.device); held_labels=data['held_labels'].to(args.device)
    rng = torch.Generator().manual_seed(261027)
    trace=[]; started=time.monotonic()
    @torch.no_grad()
    def assess():
        model.eval(); totals=torch.zeros(4,device=args.device,dtype=torch.float64)
        for first in range(0,len(held),16):
            target=held[first:first+16]
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=model(target,held_labels[first:first+16])
                ce=torch.nn.functional.cross_entropy(logits.flatten(0,1).float(),target.flatten(),reduction='none')
            totals.add_(ce.view(len(target),64,4).double().sum((0,1)))
        model.train()
        return (totals/(len(held)*64*math.log(2))).cpu().tolist()
    for step in range(args.probe_steps+1):
        if step in [0,100,args.probe_steps]:
            bits=assess();row=dict(step=step,heldout_bits_per_token=sum(bits)/4,bits_by_depth=bits,
                heldout_bpp=sum(bits)/4*256/65536,seconds=time.monotonic()-started)
            trace.append(row);progress(args,'prior_probe',candidate=name,**row)
            write(args.base,f'probes/{name}.json',dict(completed=step==args.probe_steps,
                candidate=name,vocabulary=vocabulary,trace=trace,parameters=sum(p.numel() for p in model.parameters()),
                fit_images=len(fit),heldout_training_images=len(held),model_width=256,body_layers=4,
                optimizer_batch=32,uniform_token_nll=True,shared_initialization_seed=261026,
                shared_batch_order_seed=261027,scope='small causal prior probe; not mature stage-2 quality'))
        if step==args.probe_steps:break
        index=torch.randint(len(fit),(32,),generator=rng).to(args.device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model(fit[index],labels[index])
            loss=torch.nn.functional.cross_entropy(logits.flatten(0,1).float(),fit[index].flatten())
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['cache','fit','evaluate','probe'])
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--fit-images', type=int, default=8192)
    p.add_argument('--held-images', type=int, default=1024)
    p.add_argument('--vocabulary', type=int, default=16384)
    p.add_argument('--vocabularies', type=int, nargs='+', default=[8192,16384,32768])
    p.add_argument('--passes', type=int, default=12)
    p.add_argument('--neural-steps', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--decoder-tf32',action='store_true')
    p.add_argument('--candidates',nargs='+')
    p.add_argument('--results-file',default='full-rate-distortion.json')
    p.add_argument('--candidate',default='joint49153')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--probe-steps',type=int,default=300)
    args = p.parse_args()
    args.rank = int(os.environ.get('RANK',0)); args.world = int(os.environ.get('WORLD_SIZE',1))
    torch.set_num_threads(4); torch.cuda.set_device(int(os.environ.get('LOCAL_RANK',0)))
    args.device = torch.device('cuda', torch.cuda.current_device())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    metrics.INCEPTION_PATH = args.base/'upstream/rqvae/metrics/inception.py'
    if args.phase in ('cache','evaluate'):
        dist.init_process_group('nccl', device_id=args.device)
    if args.phase == 'cache':cache(args)
    elif args.phase == 'fit':fit(args)
    elif args.phase == 'evaluate':evaluate(args)
    else:probe(args)
    if dist.is_initialized():dist.destroy_process_group()


if __name__ == '__main__':main()
