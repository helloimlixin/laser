"""Bounded matched pilot: four RQ pair positions, two categorical vocabularies.

Uses a frozen encoder-latent cache. This is a fresh stage-2 experiment, not a
resume of the sign/magnitude GMM optimizer or a full ImageNet quality test.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'runtime')]

import torch

from src.models.typed_pair_rq import TypedPairCodebook
from src.training.typed_pair_targets import best_joint_pairs, typed_pair_targets, typed_pair_cross_entropy


def record(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


@torch.no_grad()
def fit_bins(data, count, device):
    from src.orthogonal_sparse_codec import orthogonal_to_dictionary_coefficients
    dictionary = data['dictionary'].to(device)
    scales = torch.as_tensor(data['coeff_scales'], device=device)
    values = []
    train = data['train']
    for first in range(0, len(train['atoms']), 16):
        atoms = train['atoms'][first:first+16].to(device)
        gamma = train['gamma'][first:first+16].to(device)*scales
        coefficients, _ = orthogonal_to_dictionary_coefficients(dictionary.T[atoms], gamma)
        values.append(coefficients.flatten())
    values = torch.cat(values)
    bins = torch.quantile(values, torch.linspace(0, 1, count, device=device))
    # Scalar Lloyd fitting uses only the adaptation train images. Frozen shared
    # physical bins define exactly the same pair words at all four depths.
    for _ in range(12):
        ids = torch.searchsorted((bins[:-1]+bins[1:])/2, values)
        sums = torch.zeros_like(bins).scatter_add_(0, ids, values)
        counts = torch.zeros_like(bins).scatter_add_(0, ids, torch.ones_like(values))
        bins = torch.where(counts > 0, sums/counts.clamp_min(1), bins)
    if not (bins.diff()>0).all():
        raise ValueError('fitted coefficient bins are not strictly ordered')
    return bins


@torch.no_grad()
def profile(codebook, latent, targets):
    a, b = codebook.fields(targets.tokens)
    residual = latent.clone()
    entropy, atom_entropy, pairs95, full_mass = [], [], [], []
    for depth in range(codebook.depth):
        weights = targets.pair_weights[..., depth, :].flatten(0, -2)
        atoms = targets.candidate_atoms[..., depth, :].flatten(0, -2)
        marginal = weights.new_zeros(len(weights), codebook.num_atoms).scatter_add_(-1, atoms, weights)
        entropy.append(float(-(weights*weights.clamp_min(1e-30).log()).sum(-1).mean()))
        atom_entropy.append(float(-(marginal*marginal.clamp_min(1e-30).log()).sum(-1).mean()))
        pairs95.append(float((weights.sort(-1, descending=True).values.cumsum(-1)<.95).sum(-1).float().mean()+1))
        # Measure retained mass against the FULL Cartesian Boltzmann teacher
        # at 32 evenly spaced sites. Top-K teacher is explicitly an approximation.
        rr = residual.flatten(0, -2)
        rows = torch.linspace(0, len(rr)-1, min(32, len(rr)), device=rr.device).long()
        selected = rr[rows]
        tau = (.003*selected.square().sum(-1, keepdim=True)).clamp_min(1e-8)
        norms = codebook.dictionary.square().sum(0)
        correlation = selected@codebook.dictionary
        partition = selected.new_full((len(rows),), -torch.inf)
        for start in range(0, codebook.num_atoms, 128):
            c = codebook.coeff_bins
            scores = (2*correlation[:, start:start+128, None]*c
                      -norms[None, start:start+128, None]*c.square())/tau[..., None]
            partition = torch.logaddexp(partition, scores.flatten(1).logsumexp(-1))
        _, _, top_scores = best_joint_pairs(selected, codebook.dictionary, codebook.coeff_bins, weights.shape[-1])
        full_mass.append(float(((top_scores/tau).logsumexp(-1)-partition).exp().mean()))
        residual -= codebook.compound_embeddings(a[..., depth], b[..., depth])
    return dict(pair_entropy_nats=entropy, atom_entropy_nats=atom_entropy,
                mean_pairs_for_95_percent_mass=pairs95,
                retained_mass_vs_full_teacher_32_sites_by_depth=full_mass)


@torch.no_grad()
def calibrate(data, codebook, args):
    heldout = data['heldout']
    latent = heldout['latent'][:64].to(codebook.dictionary.device)
    hard = typed_pair_targets(codebook, latent, top_k=args.top_k, stochastic=False)
    soft = typed_pair_targets(codebook, latent, top_k=args.top_k,
        relative_temperature=args.temperature, generator=torch.Generator(device=latent.device).manual_seed(261029))
    from src.orthogonal_sparse_codec import reconstruct_from_orthogonal
    atoms = heldout['atoms'][:len(latent)].to(latent.device)
    gamma = heldout['gamma'][:len(latent)].to(latent.device)*torch.as_tensor(data['coeff_scales'], device=latent.device)
    original = reconstruct_from_orthogonal(codebook.dictionary.T[atoms], gamma)
    original_mse = float((latent-original).square().mean())
    hard_mse = float(hard.residual_energies[..., -1].mean()/latent.shape[-1])
    soft_mse = float(soft.residual_energies[..., -1].mean()/latent.shape[-1])
    result = dict(images=len(latent), original_full_refit_omp_latent_mse=original_mse,
        hard_fixed_pair_latent_mse=hard_mse, stochastic_fixed_pair_latent_mse=soft_mse,
        hard_to_omp_distortion_ratio=hard_mse/original_mse,
        stochastic_to_hard_distortion_ratio=soft_mse/hard_mse,
        hard_residual_mse_by_depth=(hard.residual_energies.mean((0,1,2))/latent.shape[-1]).tolist(),
        stochastic_residual_mse_by_depth=(soft.residual_energies.mean((0,1,2))/latent.shape[-1]).tolist(),
        stochastic_profile=profile(codebook, latent, soft))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--codebook', type=Path)
    parser.add_argument('--calibrate-only', action='store_true')
    parser.add_argument('--teacher', choices=['hard', 'stochastic'], default='stochastic')
    parser.add_argument('--bins', type=int, default=512)
    parser.add_argument('--top-k', type=int, default=1024)
    parser.add_argument('--temperature', type=float, default=.003)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--eval-images', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--key-file', type=Path)
    parser.add_argument('--head', choices=['conditional','independent','physical'], default='conditional')
    parser.add_argument('--initial-checkpoint', type=Path)
    parser.add_argument('--training-seed', type=int)
    parser.add_argument('--run-suffix', default='')
    parser.add_argument('--eval-every', type=int, default=50)
    args = parser.parse_args()
    if args.bins<2 or args.top_k<1 or args.steps<1 or args.batch_size<1 or args.eval_images<1 or args.lr<=0:
        parser.error('positive dimensions, steps, and learning rate are required')
    if args.temperature != .003:
        parser.error('this bounded comparison fixes relative teacher temperature at .003')
    torch.set_num_threads(8)
    torch.manual_seed(261029)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device('cuda')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'result.json').exists():
        raise FileExistsError('completed trial already exists')
    data = torch.load(args.cache, map_location='cpu', mmap=True, weights_only=False)
    assert not set(data['train']['filenames']) & set(data['heldout']['filenames'])
    if args.codebook:
        saved = torch.load(args.codebook, map_location=device, weights_only=True)
        torch.testing.assert_close(saved['dictionary'].cpu(), data['dictionary'], atol=0, rtol=0)
        codebook = TypedPairCodebook(saved['dictionary'], saved['coeff_bins'], depth=4).to(device)
    else:
        codebook = TypedPairCodebook(data['dictionary'], fit_bins(data, args.bins, device), depth=4).to(device)
    if args.calibrate_only:
        torch.save(codebook.state_dict(), args.output/'codebook.pt')
        result = calibrate(data, codebook, args)
        record(args.output/'calibration.json', result)
        print(json.dumps(result), flush=True)
        return

    from src.training.rqtransformer import build_model
    model = build_model(codebook.vocab_size, codebook.num_atoms,
        coeff_vocab_size=codebook.coeff_vocab_size, sparsity_level=4,
        physical_pair_context=True, efficient_pair_backbone=True,
        compound_micro_transformer_layers=2, compound_depth_specific_coeff_heads=True,
        compound_mask_seen_atoms_training=False, coefficient_history_layers=2,
        coefficient_history_normalize_fields=True).to(device)
    if args.initial_checkpoint:
        initial = torch.load(args.initial_checkpoint,map_location='cpu',mmap=True,weights_only=False)
        model.load_state_dict(initial['state_dict'],strict=True)
        del initial
    if args.head != 'conditional':
        from src.models.structured_pair_energy import StructuredPairEnergyRQ
        model = StructuredPairEnergyRQ(model,codebook.dictionary,codebook.coeff_bins,
            physical_interaction=args.head=='physical').to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(.9,.95), weight_decay=1e-4, fused=True)
    config = dict(teacher=args.teacher, train_images=len(data['train']['latent']), eval_images=args.eval_images,
        internal_pair_depth=4, serialized_fields=8, num_atoms=codebook.num_atoms, bins=codebook.coeff_vocab_size,
        joint_pair_outcomes=codebook.num_atoms*codebook.coeff_vocab_size,
        stored_vocabulary_ids=codebook.vocab_size, sampled_prefix_recomputed=True,
        fixed_contributions_no_refit=True, repeated_atoms_allowed=True,
        joint_nll_unweighted=True, top_k_teacher=args.top_k, relative_temperature=args.temperature,
        fresh_optimizer=True, fresh_stage2=True, seed=261029,
        steps=args.steps, batch_size=args.batch_size, lr=args.lr,
        parameters=sum(p.numel() for p in model.parameters()),
        evaluation_split='image-disjoint heldout from ImageNet train; not official validation',
        official_fid_measured=False, full_dataset_training=False)
    config.update(decoder_head=args.head,initial_checkpoint=str(args.initial_checkpoint) if args.initial_checkpoint else None,
                  fresh_stage2=args.initial_checkpoint is None,training_seed=args.training_seed,
                  exact_joint_teacher_ce=args.head!='conditional')
    record(args.output/'config.json', config)
    wb = None
    if args.key_file:
        import wandb
        os.environ['WANDB_API_KEY'] = args.key_file.read_text().strip()
        wb = wandb.init(entity='helloimlixin-rutgers', project='laser',
            id=f'imagenet-two-vocab-pair-{args.teacher}-{args.steps}steps-20261029'+args.run_suffix,
            group='two-vocab-residual-pair-pilot-20261029', config=config, dir=str(args.output))
    train = data['train']
    latent_train, labels_train = train['latent'].to(device), train['labels'].to(device)
    heldout = data['heldout']
    latent_eval = heldout['latent'][:args.eval_images].to(device)
    labels_eval = heldout['labels'][:args.eval_images].to(device)
    fixed_hard = typed_pair_targets(codebook, latent_eval, top_k=args.top_k, stochastic=False)

    @torch.no_grad()
    def evaluate(step):
        from dataclasses import fields
        model.eval()
        totals = torch.zeros(3, 4, device=device)
        per_image = []
        for first in range(0, len(latent_eval), 4):
            last = min(first+4, len(latent_eval))
            target = type(fixed_hard)(**{f.name: getattr(fixed_hard,f.name)[first:last] for f in fields(fixed_hard)})
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output = model(target.tokens, codebook, labels_eval[first:last], amp=True)
                _, atoms, coeffs = typed_pair_cross_entropy(output, target)
            totals += torch.stack((atoms, coeffs, atoms+coeffs))*(last-first)
            if step == args.steps:
                a,b=codebook.fields(target.tokens)
                al=-output['atom_logits'].float().log_softmax(-1).gather(-1,a[...,None]).squeeze(-1)
                cl=-output['coeff_logits'].float().log_softmax(-1).gather(-1,b[...,None]).squeeze(-1)
                per_image.append(torch.stack((al,cl),-1).mean((1,2)).cpu())
        totals /= len(latent_eval)
        if per_image:
            torch.save(torch.cat(per_image),args.output/'per-image-nll.pt')
        metrics = dict(step=step, heldout_atom_nll=totals[0].tolist(), heldout_coeff_nll=totals[1].tolist(),
            heldout_joint_nll=totals[2].tolist(), heldout_mean_joint_nll=float(totals[2].mean()))
        if wb:
            wb.log({f'heldout/{name}/depth{depth+1}': float(totals[i,depth])
                    for i,name in enumerate(['atom_nll','coeff_nll','joint_nll']) for depth in range(4)}, step=step)
        model.train()
        return metrics

    history = [evaluate(0)]
    print(json.dumps(history[-1]), flush=True)
    if args.training_seed is not None:
        torch.manual_seed(args.training_seed)
    order_generator = torch.Generator(device=device).manual_seed(300)
    teacher_generator = torch.Generator(device=device).manual_seed(301)
    permutation = torch.randperm(len(latent_train), device=device, generator=order_generator)
    cursor = 0
    started = time.monotonic()
    for step in range(1, args.steps+1):
        if cursor+args.batch_size>len(permutation):
            permutation = torch.randperm(len(latent_train), device=device, generator=order_generator);cursor=0
        ids = permutation[cursor:cursor+args.batch_size];cursor += args.batch_size
        targets = typed_pair_targets(codebook, latent_train[ids], top_k=args.top_k,
            relative_temperature=args.temperature, stochastic=args.teacher=='stochastic', generator=teacher_generator)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(targets.tokens, codebook, labels_train[ids], amp=True)
            if args.head == 'conditional':
                loss, atom_ce, coeff_ce = typed_pair_cross_entropy(output, targets)
            else:
                from src.models.structured_pair_energy import structured_pair_cross_entropy
                loss, atom_ce, coeff_ce = structured_pair_cross_entropy(output,targets,codebook.coeff_bins)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite joint loss')
        optimizer.step()
        if step%10==0 or step==1:
            metrics = dict(step=step, joint_ce=float(loss.detach()), atom_ce=atom_ce.detach().tolist(),
                coeff_ce=coeff_ce.detach().tolist(), grad_norm=float(grad_norm),
                elapsed_seconds=time.monotonic()-started)
            print(json.dumps(metrics), flush=True)
            if wb:
                wb.log({'train/joint_ce':float(loss.detach()), 'train/grad_norm':float(grad_norm)}, step=step)
        if step%args.eval_every==0 or step==args.steps:
            history.append(evaluate(step)); print(json.dumps(history[-1]), flush=True)
            record(args.output/'metrics.json', history)
    result = dict(config=config, evaluation=history, seconds=time.monotonic()-started,
                  all_steps_finite=True, peak_cuda_bytes=torch.cuda.max_memory_allocated())
    torch.save(dict(state_dict=model.state_dict(), codebook=codebook.state_dict(), config=config,
        optimizer=optimizer.state_dict(), step=args.steps, torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state(), teacher_rng=teacher_generator.get_state(),
        order_rng=order_generator.get_state(), permutation=permutation, cursor=cursor), args.output/'checkpoint.pt')
    record(args.output/'result.json', result)
    if wb:
        wb.summary['all_steps_finite'] = True
        wb.summary['final_heldout_joint_nll'] = history[-1]['heldout_mean_joint_nll']
        wb.finish()
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
