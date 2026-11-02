"""Compress a frozen sparse-code vocabulary by decoded latent geometry."""
import hashlib
import json
from pathlib import Path
import shutil

import torch

from src.models.joint_rate_distortion import JointVectorBook
from src.models.joint_vocab_merge import load_joint_vectors, load_encoder_vectors, merge_close_vectors, MergedJointVectorBook
from src.training.decompress_joint import write


def run(cfg):
    from src.training.decompress_joint import run as run_experiment
    run_experiment(cfg)


@torch.no_grad()
def source_codes(cfg):
    from src.training.decompress_joint_runtime import device, load_cache, progress
    gpu, base = device(), Path(cfg.local_dir)
    source = JointVectorBook(load_joint_vectors(cfg.source.control_book)).to(gpu)
    with Path(cfg.source.control_book).open('rb') as stream:
        source_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    for split in ['fit', 'held']:
        path = base/f'cache/merge-source-{split}.pt'
        if path.exists():
            cached = torch.load(path, weights_only=True, map_location='cpu')
            if cached.get('source_book_sha256') != source_hash:
                raise ValueError('Cached source sparse codes belong to another encoder vocabulary')
            continue
        data = load_cache(cfg, split)
        x = data['latents'].to(gpu)
        encoded = source.quantize(x, chunk_size=int(cfg.merge.chunk_size))
        counts = torch.stack([torch.bincount(encoded['codes'][..., d].flatten(),
            minlength=source.vocab_size) for d in range(source.depth)])
        temporary = path.with_suffix('.tmp')
        torch.save(dict(codes=encoded['codes'].cpu(), indices=data['indices'], counts=counts.cpu(),
                        source_book_sha256=source_hash), temporary)
        temporary.replace(path)
        progress(cfg, 'source-codes', split=split, images=len(x),
                 source_latent_mse=float((x-encoded['quantized']).square().mean()))


def load_merged_checkpoint(path):
    path = Path(path)
    spec = torch.load(path, map_location='cpu', weights_only=True)
    return MergedJointVectorBook(load_joint_vectors(path), load_encoder_vectors(spec, path.parent), spec['remap'],
        coordinate_sweeps=int(spec.get('coordinate_sweeps', 0)))


def save_merged_checkpoint(cfg, name, vectors, remap, metadata):
    """Depth shards keep every persistent file below the large-file upload limit."""
    base = Path(cfg.local_dir)/'checkpoints'
    out = Path(cfg.output_dir)/'checkpoints'
    base.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True)
    shards = []
    for d, table in enumerate(vectors):
        filename = f'{name}-vectors-depth{d}.pt'
        path = base/filename
        torch.save(table.detach().cpu().clone(), path)
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        shutil.copyfile(path, out/filename)
        shards.append(dict(name=filename, sha256=digest, bytes=path.stat().st_size))
    with Path(cfg.source.control_book).open('rb') as stream:
        encoder_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    payload = dict(kind='laser_joint_merged_vector_rq', format_version=1,
        encoder_decoder_frozen=True, source_stage1_checkpoint_sha256=str(cfg.source.sha256),
        encoder_book=str(cfg.source.control_book), encoder_book_sha256=encoder_hash,
        vector_shards=shards, remap=remap.cpu(),
        shared_codebook_across_depths=False, **metadata)
    path = base/f'{name}-best.pt'
    torch.save(payload, path)
    shutil.copyfile(path, out/path.name)


@torch.no_grad()
def fit(cfg):
    from src.training.decompress_joint_runtime import device, load_cache, progress
    gpu, base = device(), Path(cfg.local_dir)
    if cfg.merge.refit_target != 'source_sparse_reconstruction':
        raise ValueError('Geometric merge refits must preserve the source sparse reconstruction')
    setting = cfg.train.candidates[int(cfg.candidate_index)]
    name, vocabulary = str(setting.name), int(setting.vocabulary)
    fit_data, held_data = load_cache(cfg, 'fit'), load_cache(cfg, 'held')
    fit_codes = torch.load(base/'cache/merge-source-fit.pt', weights_only=True)
    held_codes = torch.load(base/'cache/merge-source-held.pt', weights_only=True)
    torch.testing.assert_close(fit_data['indices'], fit_codes['indices'], rtol=0, atol=0)
    torch.testing.assert_close(held_data['indices'], held_codes['indices'], rtol=0, atol=0)
    source = JointVectorBook(load_joint_vectors(cfg.source.control_book)).to(gpu)
    old_fit, old_held = fit_codes['codes'].to(gpu), held_codes['codes'].to(gpu)
    target = source.embed(old_fit).sum(-2)
    held_target = source.embed(old_held).sum(-2)
    decoder_vectors, maps, merge_trace = [], [], []
    for depth in range(source.depth):
        merged = merge_close_vectors(source.vectors[depth], fit_codes['counts'][depth], vocabulary,
            neighbors=int(cfg.merge.neighbors), pseudocount=float(cfg.merge.usage_pseudocount))
        decoder_vectors.append(merged['vectors']); maps.append(merged['remap'])
        merge_trace.append(merged['trace'])
        progress(cfg, f'fit-{name}', depth=depth, vocabulary=vocabulary, merged=True)
    remap = torch.stack(maps)
    codes = torch.stack([remap[d, old_fit[..., d]] for d in range(source.depth)], -1)
    held = torch.stack([remap[d, old_held[..., d]] for d in range(source.depth)], -1)
    book = JointVectorBook(torch.stack(decoder_vectors)).to(gpu)
    best, best_error, trace = None, float('inf'), []
    x, h = fit_data['latents'].to(gpu), held_data['latents'].to(gpu)
    for iteration in range(int(cfg.merge.refit_passes)+1):
        fitted = book.embed(codes).sum(-2)
        measured = book.embed(held).sum(-2)
        error = float((held_target-measured).square().mean())
        row = dict(iteration=iteration, fit_preservation_mse=float((target-fitted).square().mean()),
            heldout_training_preservation_mse=error,
            heldout_training_latent_mse=float((h-measured).square().mean()), vocabulary=vocabulary)
        if error < best_error:
            best_error, best = error, book.vectors.detach().cpu().clone()
        if iteration < int(cfg.merge.refit_passes):
            row.update(book.refit(target, codes, prior_weight=float(cfg.merge.refit_prior_weight)))
        trace.append(row)
        progress(cfg, f'fit-{name}', **row, completed=iteration == int(cfg.merge.refit_passes))
        write(base/f'{name}-trace.json', trace)
        write(Path(cfg.output_dir)/f'{name}-trace.json', trace)
    save_merged_checkpoint(cfg, name, best, remap, dict(vocabulary=vocabulary, depth=4,
        tokens_per_image=256, encoding='source sparse IDs mapped to geometric clusters',
        refit_target=str(cfg.merge.refit_target), heldout_training_preservation_mse=best_error,
        selection='heldout training preservation MSE; no validation fitting',
        fit_images=len(x), heldout_training_images=len(h), merge_trace=merge_trace, trace=trace,
        continuous_side_information=False, target_policy='hard_deterministic'))
