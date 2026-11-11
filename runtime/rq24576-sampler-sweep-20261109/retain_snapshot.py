"""Retain inference weights independently of the training checkpoint rotation."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    local = Path('/tmp/laser-rq24576-sampler-sweep-20261109')
    plan = json.loads((local/'plan.json').read_text())
    output, base = Path(plan['output']), Path(plan['base'])
    sys.path.insert(0, str(base))
    import torch
    from runtime.persistent_copy import copy_file
    source = torch.load(plan['checkpoint'], map_location='cpu', mmap=True, weights_only=False)
    assert source['global_step'] == plan['source_step']
    payload = {key:source[key] for key in ['state_dict', 'config', 'recipe', 'global_step', 'joint_codebook_sha256']}
    payload.update(inference_only=True, source_checkpoint_sha256=plan['checkpoint_sha256'],
        stage1_checkpoint_sha256=plan['stage1_checkpoint_sha256'])
    path = local/'frozen-inference-weights.pt'
    temporary = path.with_suffix('.pending')
    torch.save(payload, temporary)
    temporary.replace(path)
    target = output/path.name
    pending = target.with_suffix('.pending')
    copy_file(path, pending)
    size = path.stat().st_size
    block = 32*1024**2
    def verify(first):
        count = min(block, size-first)
        with path.open('rb') as left, pending.open('rb') as right:
            left.seek(first)
            right.seek(first)
            a, b = left.read(count), right.read(count)
        assert len(a) == len(b) == count
        assert hashlib.sha256(a).digest() == hashlib.sha256(b).digest()
        return count
    with ThreadPoolExecutor(max_workers=8) as pool:
        verified = sum(pool.map(verify, range(0, size, block)))
    assert verified == size and pending.stat().st_size == size
    pending.replace(target)
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    proof = dict(path=str(target), bytes=size, sha256=sha, verified_bytes=verified,
        inference_only=True, source_checkpoint_sha256=plan['checkpoint_sha256'],
        source_step=plan['source_step'], source_epoch=plan['source_epoch'],
        independent_of_training_checkpoint_rotation=True, all_model_tensors_preserved=True,
        optimizer_not_needed_for_inference=True)
    temporary = output/'checkpoint-retention.pending'
    temporary.write_text(json.dumps(proof, indent=2)+'\n')
    temporary.replace(output/'checkpoint-retention.json')
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    main()
