#!/usr/bin/env python3
"""Numerically audit the restored Church runtime against upstream and FFHQ."""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import types

ROOT = Path('/mnt/laser-church/dropout-experiment')
REPO = Path('/workspace/Projects/laser')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT / 'runtime'))
import torch
import torch.nn.functional as F
from src.training import rqtransformer as church
from src.models.rqtransformer.transformers import RQTransformer
from src.training.exact_global_batch import ExactGlobalBatchSampler
from tests.test_compound_pair_autoregressive import tiny_config, tiny_aux

torch.set_num_threads(2)
results = {}


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def upstream_equivalence():
    upstream = ROOT / 'runtime/third_party/rq-vae-transformer/rqvae'
    # Import actual released modules without the unrelated stage-1 package init.
    for name, suffix in [('rqvae', ''), ('rqvae.models', 'models'),
                         ('rqvae.models.rqtransformer', 'models/rqtransformer'),
                         ('rqvae.utils', 'utils'), ('rqvae.optimizer', 'optimizer')]:
        package = types.ModuleType(name)
        package.__path__ = [str(upstream / suffix)]
        sys.modules[name] = package
    official = importlib.import_module('rqvae.models.rqtransformer.transformers').RQTransformer
    torch.manual_seed(901)
    config = tiny_config(4)
    config.block_size = [2, 2, 4]
    reference, candidate = official(config).eval(), RQTransformer(config).eval()
    candidate.load_state_dict(reference.state_dict(), strict=True)
    dictionary = torch.randn(7, 4)
    aux = types.SimpleNamespace(get_code_emb_with_depth=lambda ids: (dictionary[ids], None))
    tokens = torch.arange(32).reshape(2, 2, 2, 4) % 7
    a, b = reference(tokens, model_aux=aux), candidate(tokens, model_aux=aux)
    torch.testing.assert_close(a, b, rtol=3e-6, atol=5e-7)
    a.square().mean().backward()
    b.square().mean().backward()
    errors = []
    for (name, pa), (other, pb) in zip(reference.named_parameters(), candidate.named_parameters()):
        assert name == other
        if pa.grad is not None:
            torch.testing.assert_close(pa.grad, pb.grad, rtol=3e-5, atol=5e-7)
            errors.append(float((pa.grad - pb.grad).abs().max()))
    results['released_rqtransformer'] = dict(forward_max_error=float((a-b).detach().abs().max()),
                                            gradient_max_error=max(errors))


def full_pair_checks(model_type, name, depth, masked):
    torch.manual_seed(71)
    config = tiny_config(depth)
    config.block_size = [2, 2, depth]
    kwargs = {'pair_autoregressive': True} if name == 'church' else {}
    model = model_type(config, 7, 5, micro_transformer_layers=2,
                       depth_specific_coeff_heads=True, **kwargs).eval()
    aux = tiny_aux(depth)
    atoms = torch.arange(4*depth).reshape(1, 2, 2, depth) % 7
    targets = atoms*5 + torch.arange(4*depth).reshape_as(atoms) % 5
    with torch.no_grad():
        baseline = model(targets, model_aux=aux)
        checks = 0
        for event in range(targets.numel()):
            for component in ['atom', 'coefficient']:
                changed = targets.clone()
                atom, coefficient = divmod(int(changed.flatten()[event]), 5)
                changed.flatten()[event] = ((atom+1)%7)*5+coefficient if component == 'atom' else atom*5+(coefficient+2)%5
                outputs = model(changed, model_aux=aux)
                for key in ['atom_logits', 'coeff_logits']:
                    a, b = baseline[key].flatten(0, -2), outputs[key].flatten(0, -2)
                    boundary = event + (not (component == 'atom' and key == 'coeff_logits'))
                    torch.testing.assert_close(a[:boundary], b[:boundary], rtol=0, atol=0)
                    for later in range(boundary, targets.numel()):
                        assert not torch.equal(a[later], b[later]), (name, component, event, key, later)
                    checks += 1
        generated = torch.full_like(targets, 2)
        model.init_cache()
        cache_errors = []
        for row in range(2):
            for col in range(2):
                for d in range(depth):
                    hidden = model.cached_head_output(generated, aux, None, (row, col, d), amp=False)
                    atom_logits = model.classifier(hidden)
                    if masked and d:
                        atom_logits.scatter_(1, atoms[:, row, col, :d], -float('inf'))
                    coeff_logits = model.coefficient_logits(hidden, aux.dictionary.t()[atoms[:, row, col, d]], depth_index=d)
                    for key, actual in [('atom_logits', atom_logits), ('coeff_logits', coeff_logits)]:
                        expected = baseline[key][:, row, col, d]
                        torch.testing.assert_close(actual, expected, rtol=3e-6, atol=5e-7)
                        finite = torch.isfinite(expected)
                        cache_errors.append(float((actual[finite]-expected[finite]).abs().max()))
                    generated[:, row, col, d] = targets[:, row, col, d]
        model.init_cache()
    history = torch.randn(2, 12, requires_grad=True)
    vectors = aux.dictionary.t()[torch.tensor([1, 3])].clone().requires_grad_()
    observed = []
    hook = model.coeff_micro_transformer.register_forward_pre_hook(lambda module, args: observed.append(args[0].detach().clone()))
    logits = model.coefficient_logits(history, vectors, depth_index=depth-1)
    hook.remove()
    torch.testing.assert_close(observed[0], torch.stack((history, model.coeff_atom_proj(vectors)), dim=1)+model.coeff_micro_pos)
    logits.square().sum().backward()
    for grad in [history.grad, vectors.grad, model.coeff_atom_proj.weight.grad]:
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0
    results[name] = dict(depth=depth, causal_perturbation_checks=checks,
                         cached_max_error=max(cache_errors), dictionary_vector_and_history_gradients=True,
                         training_masks_seen_atoms=masked)


def batches_and_objective():
    for accumulation in [2, 4]:
        samplers = [ExactGlobalBatchSampler(range(126227), 2048, 4, rank, accumulation) for rank in range(4)]
        batches = [list(s) for s in samplers]
        visited = []
        for step in range(62):
            indices = [i for rank in batches for micro in range(accumulation) for i in rank[step*accumulation+micro]]
            assert len(indices) == (2048 if step < 61 else 1299)
            visited.extend(indices)
        assert sorted(visited) == list(range(126227))
        data = torch.linspace(-2, 3, 133, dtype=torch.float64)
        samplers = [ExactGlobalBatchSampler(data, 80, 4, rank, accumulation) for rank in range(4)]
        batches = [list(s) for s in samplers]
        for step in range(2):
            grads, all_indices = [], []
            for sampler, batch in zip(samplers, batches):
                w = torch.tensor(.7, dtype=torch.float64, requires_grad=True)
                for micro in range(accumulation):
                    cursor = step*accumulation+micro
                    indices = batch[cursor]
                    all_indices.extend(indices)
                    loss = ((w*data[indices]-1)**2).mean()/accumulation
                    (loss*sampler.backward_scale(len(indices), cursor)).backward()
                grads.append(w.grad)
            w = torch.tensor(.7, dtype=torch.float64, requires_grad=True)
            ((w*data[all_indices]-1)**2).mean().backward()
            torch.testing.assert_close(torch.stack(grads).mean(), w.grad, atol=1e-14, rtol=1e-14)
    torch.manual_seed(97)
    a = torch.randn(2, 2, 2, 4, 7, requires_grad=True)
    c = torch.randn(2, 2, 2, 4, 5, requires_grad=True)
    atoms = torch.randint(7, (2, 2, 2, 4))
    q = torch.randn_like(c).softmax(-1)
    actual, _ = church.compound_objective(a, c, None, atoms, q, None, atom_weight=1.5,
                                             geometry_weight=0., accumulation=2)
    expected = (1.5*F.cross_entropy(a.reshape(-1, 7), atoms.flatten()) - (q*c.log_softmax(-1)).sum(-1).mean())/2.5/2
    torch.testing.assert_close(actual, expected)
    ag = torch.autograd.grad(actual, (a, c), retain_graph=True)
    eg = torch.autograd.grad(expected, (a, c))
    for left, right in zip(ag, eg):
        torch.testing.assert_close(left, right)
    results['four_rank_batch_and_loss'] = dict(exact_images=126227, updates_per_epoch=62,
        accumulations=[2, 4], partial_batch=1299, gradient_equivalence=True, weighted_objective_and_gradients=True)


def main():
    upstream_equivalence()
    full_pair_checks(church.CompoundLaserRQTransformer, 'church', 4, True)
    source = ROOT / 'ffhq-source/code/scripts/train_official_rqtransformer_laser_stage2.py'
    archived = load_file('audited_ffhq_trainer', source)
    full_pair_checks(archived.CompoundLaserRQTransformer, 'ffhq', 2, False)
    results['ffhq']['saved_trainer_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    results['ffhq']['backbone_provenance_limit'] = 'Saved trainer verified; imported FFHQ backbone revision was not recorded. Behavioral checks use recovered Church backbone.'
    # The deployed short-attention optimization must preserve the same causality/cache behavior.
    from fast_attention import install
    install()
    full_pair_checks(church.CompoundLaserRQTransformer, 'church', 4, True)
    results['church']['optimized_attention_checked'] = True
    batches_and_objective()
    results['passed'] = True
    (ROOT / 'audit-numerical.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
