"""Generate an inspectable preview without restarting the running trainer."""
import argparse
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--branch', choices=['cross', 'mlp'], default='cross')
    parser.add_argument('--gpu', default='7')
    parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=261001)
    parser.add_argument('--upload-wandb', action='store_true')
    args = parser.parse_args()
    base = Path('/tmp/laser-imagenet-pair-memory-20261007')
    out = Path('/workspace/Projects/laser/outputs/imagenet-pair-memory-trial-20261007')
    assets = Path('/tmp/laser-imagenet-classcond-20261007')
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu,
        TORCH_HOME=str(assets/'torch-cache'),
        TORCHINDUCTOR_CACHE_DIR=str(assets/'inductor-cache'),
        TORCHINDUCTOR_COMPILE_THREADS='2',
        LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(base/'checkpoint-cache'),
        LASER_CHECKPOINT_IMMUTABLE_FILES='1', OMP_NUM_THREADS='2')
    sys.path[:0] = [str(base/'source'), str(base/'source/runtime')]
    import torch
    from src.training import rqtransformer as training
    from src.training import k4_checkpoint_io as io
    from src.training.pair_memory_cross_attention import attach_pair_memory_queries
    from src.data.imagenet_labels import class_names_for_dataset

    status = out/f'{args.branch}-preview-status.json'
    def record(**values):
        values.update(pid=os.getpid(), branch=args.branch, time=time.time())
        temporary = status.with_suffix('.tmp')
        temporary.write_text(json.dumps(values, indent=2)+'\n')
        temporary.replace(status)
        print(json.dumps(values), flush=True)

    torch.cuda.set_per_process_memory_fraction(.22, 0)
    source = (out/args.branch/'train/checkpoints/last.pt').resolve()
    local = io._checkpoint_upload_source(source)
    assert local != source, 'Preview requires a verified local checkpoint serialization'
    pinned = base/f'preview-{args.branch}.pt'
    temporary = pinned.with_suffix('.tmp')
    temporary.unlink(missing_ok=True)
    os.link(local, temporary)
    os.replace(temporary, pinned)
    record(phase='loading', checkpoint=str(source))
    payload = torch.load(pinned, map_location='cpu', mmap=True, weights_only=False)
    step = payload['global_step']
    assert any(name.startswith('pair_memory_queries.') for name in payload['state_dict'])
    with torch.device('meta'):
        model = training.build_model(16384+2048, 16384, compound=True,
            coeff_vocab_size=2048, sparsity_level=4, compound_pair_autoregressive=True,
            model_preset='imagenet-1400m')
        attach_pair_memory_queries(model, width=512, heads=8, memory_layers=1,
                                   query_layers=2, mode=args.branch)
    model.load_state_dict(payload['state_dict'], strict=True, assign=True)
    model = model.requires_grad_(False).eval().to('cuda')
    for block in model.head_transformer.blocks:
        block.attn.short_attention_backend = 'compiled'
    aux = training.LaserAux(assets/'inputs/stage1-tokenizer.pt', 16384, 2048,
        coeff_max=3., coeff_scale=6.4,
        coeff_scales=[8.203365325927734,4.265638828277588,3.0662174224853516,1.8273425102233887],
        attn_resolutions=(8,), soft_target_physical=False, clamp_coeffs=False,
        sparsity_level=4).requires_grad_(False).eval().to('cuda')
    class_names = class_names_for_dataset('imagenet')
    rows = []
    native_save = training.save_class_labeled_grid
    def save(images, chosen, names, target, **kwargs):
        rows.extend(dict(class_id=int(index), class_name=names[int(index)]) for index in chosen.cpu())
        return native_save(images, chosen, names, target, **kwargs)
    training.save_class_labeled_grid = save
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    record(phase='generating', checkpoint_step=step, samples=args.samples, batch_size=args.batch_size)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        target = training.sample_class_grid(model, aux, class_names, out/args.branch/'train',
            step, num_samples=args.samples, sample_batch_size=args.batch_size,
            samples_per_class=8, setting_name='live-preview',
            atom_temperature=1., atom_top_k=256, atom_top_p=.95,
            coeff_temperature=1., coeff_top_k=0, coeff_top_p=.95)
    metadata = dict(checkpoint_step=step, checkpoint=str(source), pinned_local_checkpoint=str(pinned),
        seed=args.seed, samples=args.samples, samples_per_class=8, classes=rows,
        atom_temperature=1., atom_top_k=256, atom_top_p=.95,
        coefficient_temperature=1., coefficient_top_k=0, coefficient_top_p=.95,
        checkpoint_query_mode=args.branch, generated_from_scratch=True,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    target.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    record(phase='saved', checkpoint_step=step, samples=args.samples, grid=str(target))
    if args.upload_wandb:
        import wandb
        os.environ['WANDB_API_KEY'] = Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip()
        api = wandb.Api()
        run = api.run(f'helloimlixin-rutgers/laser/imagenet-rfid421-pair-memory-{args.branch}-8h100-20261007')
        run.upload_file(str(target), root=str(out/args.branch/'train'))
        run.upload_file(str(target.with_suffix('.json')), root=str(out/args.branch/'train'))
        record(phase='complete', checkpoint_step=step, samples=args.samples,
            grid=str(target), wandb_file='samples/'+target.name,
            wandb_url=run.url+'/files')
    return target


if __name__ == '__main__':
    main()
