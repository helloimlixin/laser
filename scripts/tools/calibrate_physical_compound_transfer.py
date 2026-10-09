"""Measure a real ImageNet checkpoint transfer without using training GPUs."""
import argparse
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--anchor', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(args.base/'support')]
    import torch
    from torchvision.datasets import ImageFolder
    from src.training.rqtransformer import LaserAux, build_model, val_image_transform
    from src.training.physical_pair_crps import physical_pair_objective_components
    from src.models.physical_compound_prior import PhysicalCompoundRQTransformer
    from src.training.k4_checkpoint_io import _checkpoint_upload_source

    torch.set_num_threads(8)
    args.output.mkdir(parents=True, exist_ok=True)
    def record(name, value):
        target = args.output/name
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, indent=2, default=str)+'\n')
        temporary.replace(target)
        print(json.dumps(value, default=str), flush=True)
    started = time.monotonic()
    os.environ['LASER_CHECKPOINT_UPLOAD_CACHE_DIR'] = str(args.base/'checkpoint-upload-cache')
    source = _checkpoint_upload_source(args.anchor.resolve())
    payload = torch.load(source, map_location='cpu', mmap=True, weights_only=False)
    config = payload['config']
    assert (payload['epoch'], payload['global_step']) == (77, 48202)
    assert payload['optimizer']['param_groups'][0]['lr'] == 1e-6
    with torch.device('meta'):
        model = build_model(config['num_atoms']+config['coeff_vocab_size'], config['num_atoms'],
            physical_pair_context=True, sparsity_level=4,
            coeff_vocab_size=config['coeff_vocab_size'], model_preset=config['model_preset'])
    model.load_state_dict(payload['state_dict'], strict=True, assign=True)
    model.eval()
    old_parameters = dict(model.named_parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, betas=(.9,.95), weight_decay=1e-4)
    optimizer.load_state_dict(payload['optimizer'])
    old_states = {name: optimizer.state[p] for name, p in old_parameters.items()}
    assert {int(s['step']) for s in old_states.values()} == {13772}
    aux = LaserAux(args.base/'inputs/resume-stage1-tokenizer.pt', config['num_atoms'],
        config['coeff_vocab_size'], config['coeff_max'], config['coeff_scale'],
        coeff_scales=config['coeff_scales'], soft_target_physical=False,
        clamp_coeffs=False, sparsity_level=4).eval()
    dataset = ImageFolder(Path(config['data'])/'val', transform=val_image_transform())
    torch.manual_seed(261001)
    indices = [dataset.targets.index(label) for label in (19,151,281,517,817,970)]
    scalars, targets, labels = [], [], []
    with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
        for index in indices:
            image, label = dataset[index]
            atoms, coefficients = aux.encode_sparse_components(image[None])
            token, target = aux.sparse_targets(atoms, coefficients, temp=.01125, compact=True)
            scalars.append(token); targets.append(target); labels.append(label)
    weight = torch.tensor(.05)
    def measure(compound):
        scores = []
        with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
            for token, (atoms, probs), label in zip(scalars, targets, labels):
                inputs = (atoms*aux.coeff_vocab_size + token[...,1::2]-aux.num_atoms) if compound else token
                out = model(inputs, aux, torch.tensor([label]))
                loss, ce, crps = physical_pair_objective_components(
                    out['atom_logits'], out['coeff_logits'], atoms, probs, aux.coeff_bins, weight, 1)
                assert torch.isfinite(loss)
                scores.append(dict(label=label, loss=float(loss), cross_entropy=float(ce), crps=float(crps)))
        return dict(images=scores, mean_cross_entropy=sum(s['cross_entropy'] for s in scores)/len(scores))
    baseline = measure(False)
    record('native-teacher-calibration.json', baseline)
    rng = torch.get_rng_state().clone()
    PhysicalCompoundRQTransformer.from_scalar(model)
    assert torch.equal(rng, torch.get_rng_state())
    current = dict(model.named_parameters())
    new = {name:p for name,p in current.items() if name not in old_parameters}
    assert len(new) == 19
    for name, parameter in old_parameters.items():
        assert current[name] is parameter
        assert optimizer.state[parameter] is old_states[name]
        assert torch.equal(parameter, payload['state_dict'][name])
    optimizer.add_param_group(dict(params=list(new.values()), lr=1e-6, initial_lr=config['lr']))
    migration = dict(source=str(source), epoch=77, global_step=48202,
        old_parameter_tensors=len(old_parameters), old_parameter_values=sum(p.numel() for p in old_parameters.values()),
        old_adam_step=13772, old_adam_tensors_preserved=True, parameter_identity_preserved=True,
        source_weights_bitwise_equal=True, new_parameter_tensors=len(new),
        new_parameter_values=sum(p.numel() for p in new.values()),
        new_optimizer_state='fresh only for added history block, normalization and gates',
        scheduler_global_step=payload['scheduler']['last_epoch'], source_lr=1e-6,
        rng_ranks=len(payload['rng_state_by_rank']), migration_did_not_consume_torch_rng=True,
        compound_events=model.events, both_decoders_full_history=True)
    record('checkpoint-transfer.json', migration)
    transferred = measure(True)
    record('compound-teacher-calibration.json', transferred)
    torch.save(dict(tokens=scalars, targets=targets, labels=labels, indices=indices), args.output/'fixed-teacher-batch.pt')
    record('calibration-complete.json', dict(native=baseline['mean_cross_entropy'],
        compound=transferred['mean_cross_entropy'],
        elapsed_seconds=time.monotonic()-started, production_activated=False))


if __name__ == '__main__':
    main()
