#!/usr/bin/env python3
"""Apply operational-only changes to the recovered W&B source snapshot."""
import argparse
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    base = parser.parse_args().base
    baseline = base / 'recovery/runtime-baseline'
    baseline.mkdir(parents=True, exist_ok=True)
    for name in ('src/training/rqtransformer.py',
                 'scripts/tools/build_official_imagenet_token_cache.py',
                 'third_party/rq-vae-transformer/rqvae/img_datasets/lsun.py'):
        path = base / 'source' / name
        saved = baseline / Path(name).name
        if not saved.exists():
            shutil.copy2(path, saved)
        text = saved.read_text()

        def replace(old, new):
            nonlocal text
            if text.count(old) != 1:
                raise ValueError(f'{name}: expected exactly one occurrence of {old[:100]!r}')
            text = text.replace(old, new)

        if name.endswith('rqtransformer.py'):
            replace('from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset',
                    'from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset\n'
                    'from church_compound_support import (adapt_resume_payload, install_stop_handler,\n'
                    '                                     should_stop, record_progress, continuation_identity,\n'
                    '                                     record_completion)')
            replace('def main(argv=None):', 'def main(argv=None):\n    install_stop_handler()')
            replace('shuffle=sampler is None, num_workers=8, pin_memory=True,',
                    'shuffle=sampler is None, num_workers=2, pin_memory=True,')
            replace('raw_payload = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)',
                    'raw_payload = torch.load(resume_checkpoint, map_location="cpu", weights_only=False, mmap=True)\n'
                    '            raw_payload = adapt_resume_payload(raw_payload, args)')
            replace('resume="allow" if args.wandb_id else None', 'resume="must" if args.wandb_id else None')
            replace('    use_wandb = rank() == 0 and args.wandb_mode != "disabled"',
                    '    runtime_config["amarel_continuation"] = continuation_identity()\n'
                    '    use_wandb = rank() == 0 and args.wandb_mode != "disabled"')
            replace('"restored exact per-rank checkpoint streams"\n                    if restored_rng',
                    '"restored rank streams with an eight-to-four layout change; global stochastic trajectory changes"\n'
                    '                    if resume_payload.get("amarel_world_size_change")\n'
                    '                    else "restored exact per-rank checkpoint streams"\n                    if restored_rng')
            replace('artifact.add_file(str(token_cache), name=token_cache.name)',
                    'artifact.add_file(str(token_cache), name=token_cache.name, policy="immutable", skip_cache=True)')
            replace('                global_step += 1',
                    '                global_step += 1\n'
                    '                stop_requested = should_stop(device) or (\n'
                    '                    args.max_optimizer_steps > 0\n'
                    '                    and global_step - launch_start_step >= args.max_optimizer_steps\n'
                    '                )')
            replace('                    wb.log(payload)\n                    last_perf_step',
                    '                    wb.log(payload)\n                    record_progress(payload)\n                    last_perf_step')
            replace('                if args.save_step_freq > 0 and global_step % args.save_step_freq == 0:',
                    '                if stop_requested or (args.save_step_freq > 0 and global_step % args.save_step_freq == 0):')
            replace('                if args.sample_grid_every > 0 and global_step % args.sample_grid_every == 0:',
                    '                if not stop_requested and args.sample_grid_every > 0 and global_step % args.sample_grid_every == 0:')
            replace('                if (\n                    args.max_optimizer_steps > 0\n'
                    '                    and global_step - launch_start_step >= args.max_optimizer_steps\n                ):',
                    '                if stop_requested:')
            replace('"as requested; skipped epoch evaluation and checkpointing",',
                    '"after saving and queuing the final recovery checkpoint",')
            replace('    if wb is not None:\n        wb.finish()\n    if dist.is_initialized():\n        dist.destroy_process_group()',
                    '    if wb is not None:\n        wb.finish()\n'
                    '    if rank() == 0:\n        record_completion(global_step)\n'
                    '    if dist.is_initialized():\n        dist.destroy_process_group()')
        elif name.endswith('lsun.py'):
            replace('root = Path(root) / LSUNClass.subpaths[category_name]',
                    'root = str(Path(root) / LSUNClass.subpaths[category_name])')
        else:
            replace('import argparse', 'import argparse\nfrom church_compound_support import sha')
            replace('    shard = args.output.with_suffix(f".rank{rank:02d}.pt")',
                    '    identity = {"tokenizer_sha256": sha(args.checkpoint),\n'
                    '                "items": len(base), "world_size": world, "args": {\n'
                    '                    k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()\n'
                    '                }}\n'
                    '    shard = args.output.with_suffix(f".rank{rank:02d}.pt")')
            replace('    if shard.is_file():\n        print(',
                    '    if shard.is_file():\n'
                    '        previous = torch.load(shard, weights_only=True, mmap=True)\n'
                    '        if previous.get("identity") != identity:\n'
                    '            raise RuntimeError(f"Cache shard provenance mismatch: {shard}")\n'
                    '        del previous\n        print(')
            replace('            "atoms": torch.cat(atoms), "coeffs": torch.cat(coeffs),',
                    '            "identity": identity,\n            "atoms": torch.cat(atoms), "coeffs": torch.cat(coeffs),')
            replace('        order = torch.cat([x["indices"] for x in parts]).argsort()',
                    '        if any(part.get("identity") != identity for part in parts):\n'
                    '            raise RuntimeError("Mixed cache shard provenance")\n'
                    '        all_indices = torch.cat([x["indices"] for x in parts])\n'
                    '        if not torch.equal(all_indices.sort().values, torch.arange(len(base))):\n'
                    '            raise RuntimeError("Cache has missing or duplicate dataset rows")\n'
                    '        order = all_indices.argsort()')
            replace('"transform": "resize256_center_crop256", "items": len(base),',
                    '"tokenizer_sha256": identity["tokenizer_sha256"],\n'
                    '                     "transform": "resize256_center_crop256", "items": len(base),')
        path.write_text(text)
    print('Applied reproducible operational patches; baseline retained at', baseline)


if __name__ == '__main__':
    main()
