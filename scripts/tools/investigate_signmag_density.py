"""Matched, bounded repair trials against a recovered sign/magnitude pilot.

Uses the run's archived source, exact checkpoint weights and Adam moments.
Cache train/heldout splits must be image disjoint. This is an adaptation trial,
not a continuation of the original full-dataset LR/batch schedule or a FID test.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gradient-scale', type=float, required=True)
    parser.add_argument('--key-file', type=Path)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--coefficient-input-noise', type=float, default=0.)
    parser.add_argument('--soft-atom-targets', action='store_true')
    parser.add_argument('--geometric-atom-head', action='store_true')
    parser.add_argument('--cluster-atom-head', action='store_true')
    parser.add_argument('--innovation-depth-context', action='store_true')
    parser.add_argument('--atom-only-support-context', action='store_true')
    parser.add_argument('--projected-atom-head', action='store_true')
    parser.add_argument('--geometric-head-lr', type=float, default=1e-3)
    args = parser.parse_args()
    if sum((args.cluster_atom_head, args.geometric_atom_head, args.projected_atom_head)) > 1:
        parser.error('choose one dictionary readout')
    if args.atom_only_support_context and (args.innovation_depth_context or args.cluster_atom_head or args.geometric_atom_head or args.projected_atom_head):
        parser.error('atom-only support is an isolated context trial')
    if args.steps <= 0 or args.batch_size <= 0 or args.lr <= 0 or args.coefficient_input_noise < 0:
        parser.error('steps, batch size, and learning rate must be positive')
    sys.path[:0] = [str(args.source), str(args.source/'runtime')]
    import torch
    import torch.nn.functional as F
    from types import SimpleNamespace
    from src.models.sign_magnitude_prior import SignMagnitudeRQTransformer, sign_magnitude_terms
    from src.orthogonal_sparse_codec import ordered_support_basis
    from src.models.orthogonal_gmm_prior import orthogonal_gmm_objective
    from src.training.orthogonal_gmm_runtime import model_config
    from src.training.gmm_gradient_clipping import GMMGradientClipper
    helper = Path(__file__).resolve().parents[2]/'src/training/density_gradient.py'
    spec = importlib.util.spec_from_file_location('density_gradient_policy', helper)
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(261029)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'result.json').exists():
        raise FileExistsError('Trial already completed')
    while not args.cache.exists():
        print('Waiting for encoded, disjoint image splits', flush=True)
        time.sleep(10)
    data = torch.load(args.cache, map_location='cpu', weights_only=False)
    train, heldout = data['train'], data['heldout']
    if args.soft_atom_targets and 'soft_atom_targets' not in data:
        raise ValueError('soft atom training requires an oracle target cache')
    assert not set(train['filenames']) & set(heldout['filenames'])
    saved = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    model = SignMagnitudeRQTransformer(model_config(), 16384, components=16,
                                      scale_floor=.01, micro_transformer_layers=2)
    model.load_state_dict(saved['state_dict'], strict=True)
    model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4,
                                  betas=(.9, .95), fused=True)
    optimizer.load_state_dict(saved['optimizer'])
    for group in optimizer.param_groups:
        group['lr'] = args.lr
    source_step = saved['global_step']
    del saved
    aux = SimpleNamespace(dictionary=data['dictionary'].cuda(),
                          coeff_scales=torch.tensor(data['coeff_scales'], device='cuda'))
    if args.innovation_depth_context:
        context_path = Path(__file__).resolve().parents[2]/'src/training/innovation_depth_context.py'
        context_spec = importlib.util.spec_from_file_location('innovation_depth_context_policy', context_path)
        context_policy = importlib.util.module_from_spec(context_spec)
        context_spec.loader.exec_module(context_policy)
        context_policy.install_innovation_depth_context(model)
    if args.atom_only_support_context:
        context_path = Path(__file__).resolve().parents[2]/'src/training/atom_only_support_context.py'
        context_spec = importlib.util.spec_from_file_location('atom_only_support_policy', context_path)
        context_policy = importlib.util.module_from_spec(context_spec)
        context_spec.loader.exec_module(context_policy)
        context_policy.install_atom_only_support_context(model)
    if args.geometric_atom_head or args.cluster_atom_head or args.projected_atom_head:
        head_path = Path(__file__).resolve().parents[2]/'src/models/dictionary_atom_readout.py'
        head_spec = importlib.util.spec_from_file_location('dictionary_atom_head_policy', head_path)
        head_policy = importlib.util.module_from_spec(head_spec)
        head_spec.loader.exec_module(head_policy)
        install = (head_policy.install_clustered_atom_readout if args.cluster_atom_head else
                   head_policy.install_projected_dictionary_readout if args.projected_atom_head else
                   head_policy.install_dictionary_atom_readout)
        new_parameters = install(model, aux.dictionary)
        optimizer.add_param_group(dict(params=new_parameters, lr=args.geometric_head_lr))
    clip = GMMGradientClipper(model.named_parameters())
    wb = None
    config = dict(source_run='helloimlixin-rutgers/laser/imagenet-signmag-gmm16-pilot-2b300-20261029',
                  source_step=source_step, density_shared_gradient_scale=args.gradient_scale,
                  magnitude_head_gradient_scale=1., sign_gradient_scale=1.,
                  steps=args.steps, batch_size=args.batch_size, lr=args.lr, seed=261029,
                  train_images=len(train['atoms']), heldout_images=len(heldout['atoms']),
                  image_disjoint=True, data_source=data.get('source'),
                  optimizer_moments_restored=True, full_dataset_continuation=False,
                  official_fid_measured=False)
    config['coefficient_input_noise_std'] = args.coefficient_input_noise
    config['soft_atom_targets'] = data.get('soft_atom_targets') if args.soft_atom_targets else None
    config['geometric_atom_head'] = args.geometric_atom_head
    config['cluster_atom_head'] = args.cluster_atom_head
    config['innovation_depth_context'] = args.innovation_depth_context
    config['atom_only_support_context'] = args.atom_only_support_context
    config['projected_atom_head'] = args.projected_atom_head
    config['geometric_head_lr'] = args.geometric_head_lr if (args.geometric_atom_head or args.cluster_atom_head or args.projected_atom_head) else None
    if args.key_file:
        import wandb
        os.environ['WANDB_API_KEY'] = args.key_file.read_text().strip()
        wb = wandb.init(entity='helloimlixin-rutgers', project='laser',
            id=('imagenet-signmag-densitygrad-'+str(args.gradient_scale).replace('.', 'p')+
                ('-noise'+str(args.coefficient_input_noise).replace('.', 'p') if args.coefficient_input_noise else '')+
                ('-softatoms' if args.soft_atom_targets else '')+
                ('-geometry' if args.geometric_atom_head else '')+
                ('-clusters' if args.cluster_atom_head else '')+
                ('-innovctx' if args.innovation_depth_context else '')+
                ('-atomctx' if args.atom_only_support_context else '')+
                ('-projected' if args.projected_atom_head else '')+
                '-4h100-20261029'),
            name=f'ImageNet signmag | density gradient {args.gradient_scale:g}, input noise {args.coefficient_input_noise:g} | repair trial',
            group='signmag-density-gradient-investigation-20261029', config=config,
            dir=str(args.output))

    def batch(split, ids):
        a, g, y = [split[k][ids].cuda() for k in ('atoms', 'gamma', 'labels')]
        return torch.stack((a.float(), g), -1), a, g, y

    @torch.no_grad()
    def evaluate(step):
        model.eval()
        sums = {}
        n = len(heldout['atoms'])
        for start in range(0, n, 8):
            p, a, g, y = batch(heldout, slice(start, start+8))
            output = model(p, aux, y, amp=True)
            _, details = orthogonal_gmm_objective(output, a, g)
            if args.cluster_atom_head:
                groups = model.classifier.groups
                probabilities = output['atom_logits'].float().softmax(-1)
                coarse = probabilities.new_zeros(*probabilities.shape[:-1], 256)
                coarse.scatter_add_(-1, groups.expand_as(probabilities), probabilities)
                targets = groups[a]
                coarse_ce = -coarse.gather(-1, targets.unsqueeze(-1)).squeeze(-1).clamp_min(1e-30).log()
                coarse_accuracy = (coarse.argmax(-1)==targets).float()
                for d in range(a.shape[-1]):
                    details[f'train/atom_coarse_ce_depth{d}'] = coarse_ce[..., d].mean()
                    details[f'train/atom_coarse_accuracy_depth{d}'] = coarse_accuracy[..., d].mean()
            if 'latent' in heldout:
                basis, _ = ordered_support_basis(aux.dictionary.T[a])
                parts = basis * (g*aux.coeff_scales).unsqueeze(-1)
                prefix = torch.cat((torch.zeros_like(parts[..., :1, :]), parts.cumsum(-2)[..., :-1, :]), -2)
                residual = heldout['latent'][start:start+len(a)].cuda().unsqueeze(-2)-prefix
                predicted = output['atom_logits'].argmax(-1)
                predicted_score = (residual*aux.dictionary.T[predicted]).sum(-1).square()
                teacher_score = (residual*aux.dictionary.T[a]).sum(-1).square()
                regret = (1-predicted_score/teacher_score.clamp_min(1e-10)).clamp(0, 1)
                for d in range(a.shape[-1]):
                    details[f'train/atom_correlation_regret_depth{d}'] = regret[..., d].mean()
                if args.soft_atom_targets:
                    ids = heldout['oracle_atom_ids'][start:start+len(a)].cuda().long()
                    weights = heldout['oracle_atom_weights'][start:start+len(a)].cuda()
                    soft_ce = -(output['atom_logits'].float().log_softmax(-1).gather(-1, ids)*weights).sum(-1)
                    for d in range(a.shape[-1]):
                        details[f'train/atom_soft_ce_depth{d}'] = soft_ce[..., d].mean()
            for key, value in details.items():
                sums[key.replace('train/', 'heldout/')] = sums.get(key.replace('train/', 'heldout/'), 0.) + float(value)*len(a)/n
        record = dict(step=step, **sums)
        with (args.output/'metrics.jsonl').open('a') as f:
            f.write(json.dumps(record)+'\n')
        print(json.dumps(record), flush=True)
        if wb:
            wb.log(record)
        return record

    initial = evaluate(0)
    (args.output/'initial.json').write_text(json.dumps(initial, indent=2))
    # Measure competing losses at the shared depth-transformer output, before
    # applying the candidate gradient policy. No model or optimizer update.
    captured = []
    handle = model.head_transformer.register_forward_hook(lambda m, x, out: captured.append(out))
    p, a, g, y = batch(train, slice(0, 8))
    output = model(p, aux, y, amp=True)
    atom_loss = F.cross_entropy(output['atom_logits'].float().flatten(0, -2), a.flatten())
    sign, magnitude, _, _ = sign_magnitude_terms(output, g)
    features = [captured[-1] if args.atom_only_support_context else captured[0], captured[0], captured[0]]
    task_grads = [torch.autograd.grad(loss, feature, retain_graph=True)[0].float()
                  for loss, feature in zip((atom_loss, sign.mean(), magnitude.mean()), features)]
    diagnostics = {}
    for index, name in enumerate(('atom', 'sign', 'magnitude')):
        diagnostics[name+'_gradient_norm'] = float(task_grads[index].norm())
    diagnostics['magnitude_to_atom_gradient_norm_ratio'] = float(task_grads[2].norm()/task_grads[0].norm())
    if not args.atom_only_support_context:
        diagnostics['magnitude_atom_gradient_cosine'] = float(F.cosine_similarity(task_grads[2].flatten(), task_grads[0].flatten(), dim=0))
    else:
        diagnostics['separate_depth_feature_passes'] = True
    (args.output/'gradient-diagnostics.json').write_text(json.dumps(diagnostics, indent=2))
    print(json.dumps(diagnostics), flush=True)
    if wb:
        wb.summary.update({'diagnostic/'+k:v for k,v in diagnostics.items()})
    handle.remove()
    del output, task_grads, captured, atom_loss, sign, magnitude
    policy.install_density_gradient_scale(model, args.gradient_scale)
    permutation = torch.Generator().manual_seed(261029)
    indices = torch.randperm(len(train['atoms']), generator=permutation)
    cursor = 0
    # Identical RNG states across arms after construction/diagnostics.
    torch.manual_seed(261029)
    noise_rng = torch.Generator(device='cuda').manual_seed(261029)
    started = time.time()
    for step in range(1, args.steps+1):
        if cursor + args.batch_size > len(indices):
            indices = torch.randperm(len(indices), generator=permutation)
            cursor = 0
        ids = indices[cursor:cursor+args.batch_size]
        cursor += args.batch_size
        model.train()
        optimizer.zero_grad(set_to_none=True)
        p, a, g, y = batch(train, ids)
        if args.coefficient_input_noise:
            # Only causal history sees noise; supervised targets stay exact.
            # A separate generator preserves the matched dropout RNG stream.
            p[..., 1] += args.coefficient_input_noise * torch.randn(
                g.shape, device=g.device, dtype=g.dtype, generator=noise_rng)
        output = model(p, aux, y, amp=True)
        loss, details = orthogonal_gmm_objective(output, a, g)
        if args.soft_atom_targets:
            ids_soft = train['oracle_atom_ids'][ids].cuda().long()
            weights_soft = train['oracle_atom_weights'][ids].cuda()
            soft_ce = -(output['atom_logits'].float().log_softmax(-1).gather(-1, ids_soft)*weights_soft).sum(-1)
            sign_nll, magnitude_nll, _, _ = sign_magnitude_terms(output, g)
            loss = (soft_ce + sign_nll + magnitude_nll).mean()/2
            details['train/atom_soft_ce'] = soft_ce.mean()
        loss.backward()
        norm, clipping = clip()
        optimizer.step()
        if step % 10 == 0:
            record = dict(step=step, loss=float(loss), grad_norm=float(norm),
                          **{k:float(v) for k,v in details.items()},
                          **{k:float(v) for k,v in clipping.items()})
            print(json.dumps(record), flush=True)
            if wb:
                wb.log(record)
        if step % 25 == 0 or step == args.steps:
            final = evaluate(step)
    path = args.output/'repair.pt'
    torch.save(dict(state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                    source_step=source_step, adaptation_steps=args.steps, config=config), path)
    result = dict(config=config, initial=initial, final=final, diagnostics=diagnostics,
                  seconds=time.time()-started, checkpoint=str(path),
                  wandb_url=wb.url if wb else None)
    (args.output/'result.json').write_text(json.dumps(result, indent=2))
    if wb:
        wb.summary.update({'result/completed':True, 'result/checkpoint':str(path),
                           'result/heldout_atom_nll_delta':final['heldout/atom_nll']-initial['heldout/atom_nll']})
        wb.save(str(args.output/'result.json'), base_path=str(args.output), policy='now')
        wb.save(str(args.output/'gradient-diagnostics.json'), base_path=str(args.output), policy='now')
        wb.finish()


if __name__ == '__main__':
    main()
