"""Stream verified CC3M tar shards once into resumable FP32 compound caches."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from src.training.rqtransformer import LaserAux, val_image_transform

STAGE1_SHA = 'dd28db9306d526bdc9fbf8016403e106c4f317fd8f5c04792f0953e53310fdab'


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def text_tokenizer(dropout=0.0):
    from tokenizers import CharBPETokenizer
    base = ROOT/'third_party/rq-vae-transformer/rqvae/txtimg_datasets/tokenizers/pretrained'
    tok = CharBPETokenizer.from_file(str(base/'bpe-16k-vocab.json'),
        str(base/'bpe-16k-merges.txt'), unk_token='[UNK]', lowercase=True, dropout=dropout)
    tok.add_special_tokens(['[PAD]'])
    tok.enable_padding(length=32, pad_id=tok.token_to_id('[PAD]'))
    tok.enable_truncation(max_length=32)
    assert tok.get_vocab_size() == 16384
    return tok


def shard_records(path):
    """Read adjacent WDS members sequentially, rejecting corrupt or unpaired data."""
    key, record = None, {}
    with tarfile.open(local_shard(path), 'r|*') as archive:
        for member in archive:
            if not member.isfile():
                continue
            stem, suffix = os.path.splitext(member.name)
            if key is not None and stem != key:
                yield key, record
                record = {}
            key = stem
            if suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.txt'}:
                record[suffix.lower()] = archive.extractfile(member).read()
        if key is not None:
            yield key, record


def local_shard(path):
    """Bound shared-mount reads and decode exclusively from verified local disk."""
    root = os.environ.get('CC3M_LOCAL_SHARDS')
    if not root:
        return path
    path = Path(path)
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    target = root/path.name
    receipt = target.with_suffix('.verified.json')
    source = json.loads((path.parent.parent/'verified'/(path.name+'.json')).read_text())
    with (root/(path.name+'.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if receipt.is_file() and target.is_file() and target.stat().st_size == source['size']:
            assert json.loads(receipt.read_text())['sha256'] == source['sha256']
            return target
        with staging_slot(root):
            for attempt in range(5):
                temp = target.with_suffix('.partial')
                try:
                    digest = hashlib.sha256()
                    with path.open('rb') as reader, temp.open('wb') as writer:
                        while chunk := reader.read(4*1024*1024):
                            writer.write(chunk); digest.update(chunk)
                    if digest.hexdigest() != source['sha256'] or temp.stat().st_size != source['size']:
                        raise ValueError(f'CC3M local staging checksum mismatch: {path}')
                    temp.replace(target)
                    write_json(receipt, source)
                    return target
                except OSError:
                    temp.unlink(missing_ok=True)
                    if attempt == 4:
                        raise
                    time.sleep(2*(attempt+1))


@contextmanager
def staging_slot(root):
    locks = [(root/f'.copy-slot-{i}.lock').open('a') for i in range(2)]
    try:
        while True:
            for lock in locks:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    yield
                    return
                except BlockingIOError:
                    continue
            time.sleep(.1)
    finally:
        for lock in locks:
            lock.close()


class ShardBatches(IterableDataset):
    def __init__(self, shards, batch_size):
        self.shards, self.batch_size = list(shards), batch_size

    def __iter__(self):
        worker = get_worker_info()
        shards = self.shards if worker is None else self.shards[worker.id::worker.num_workers]
        transform = val_image_transform()
        for path in shards:
            images, captions, keys = [], [], []
            for key, record in shard_records(path):
                raw = next((record[e] for e in ('.jpg', '.jpeg', '.png', '.webp') if e in record), None)
                if raw is None or '.txt' not in record:
                    raise ValueError(f'Unpaired sample: {path}:{key}')
                caption = record['.txt'].decode('utf-8').strip()
                if not caption:
                    raise ValueError(f'Empty caption: {path}:{key}')
                with Image.open(io.BytesIO(raw)) as image:
                    images.append(transform(image.convert('RGB')))
                captions.append(caption)
                keys.append(key)
                if len(images) == self.batch_size:
                    yield dict(shard=Path(path).stem, images=torch.stack(images), captions=captions, keys=keys, done=False)
                    images, captions, keys = [], [], []
            if images:
                yield dict(shard=Path(path).stem, images=torch.stack(images), captions=captions, keys=keys, done=False)
            yield dict(shard=Path(path).stem, done=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--num-workers', type=int, default=3)
    p.add_argument('--max-shards', type=int, default=0)
    args = p.parse_args()
    rank, world, local = [int(os.environ.get(k, d)) for k, d in [('RANK',0),('WORLD_SIZE',1),('LOCAL_RANK',0)]]
    torch.set_num_threads(4)
    torch.cuda.set_device(local)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32 = False
    with args.checkpoint.open('rb') as f:
        assert hashlib.file_digest(f, 'sha256').hexdigest() == STAGE1_SHA
    complete = json.loads((args.data/'COMPLETE.json').read_text())
    shards = sorted((args.data/'shards').glob('cc3m-*.tar'))
    assert len(shards) == complete['verified_tar_shards'] == 592
    if args.max_shards:
        shards = shards[:args.max_shards]
    args.output.mkdir(parents=True, exist_ok=True)
    pending = []
    for path in shards[rank::world]:
        target = args.output/(path.stem+'.pt')
        receipt = target.with_suffix('.json')
        if target.is_file() and receipt.is_file():
            saved = json.loads(receipt.read_text())
            assert saved['stage1_sha256'] == STAGE1_SHA and saved['source_sha256'] == json.loads((args.data/'verified'/(path.name+'.json')).read_text())['sha256']
        else:
            pending.append(path)
    aux = LaserAux(args.checkpoint, 16384, 2048, 1e9, 1.0,
        attn_resolutions=(8,), sparsity_level=4, clamp_coeffs=False).cuda().eval()
    tokenizer = text_tokenizer()
    loader = DataLoader(ShardBatches(pending, args.batch_size), batch_size=None,
        num_workers=args.num_workers, pin_memory=True, prefetch_factor=2 if args.num_workers else None)
    buffers, seen, start = {}, 0, time.monotonic()
    with torch.inference_mode():
        for batch in loader:
            name = batch['shard']
            if batch['done']:
                parts = buffers.pop(name)
                payload = {k: torch.cat(parts[k]) for k in ('atoms', 'coeffs', 'text_ids')}
                payload.update(captions=parts['captions'], keys=parts['keys'])
                assert payload['atoms'].shape[1:] == (8,8,4)
                assert len(set(payload['keys'])) == len(payload['keys'])
                assert torch.isfinite(payload['coeffs']).all()
                source = args.data/'shards'/(name+'.tar')
                payload['meta'] = dict(format='laser_cc3m_compound_physical_v1',
                    shape=[8,8,4], stage1_sha256=STAGE1_SHA,
                    source_sha256=json.loads((args.data/'verified'/(source.name+'.json')).read_text())['sha256'],
                    items=len(payload['keys']), coefficient_storage='fp32', encoder_precision='fp32',
                    clip_coefficients=False, transform='resize256_center_crop256',
                    text_tokenizer='bpe16k_huggingface', text_length=32,
                    coeff_abs_max=payload['coeffs'].abs().reshape(-1,4).amax(0).tolist())
                target = args.output/(name+'.pt')
                temp = target.with_suffix('.tmp.pt')
                torch.save(payload, temp)
                temp.replace(target)
                write_json(target.with_suffix('.json'), payload['meta'])
                print(json.dumps(dict(rank=rank, shard=name, items=payload['meta']['items'],
                    total_items=seen, images_per_second=seen/(time.monotonic()-start))), flush=True)
                continue
            atoms, coeffs = aux.encode_sparse_components(batch['images'].cuda(non_blocking=True))
            part = buffers.setdefault(name, dict(atoms=[], coeffs=[], text_ids=[], captions=[], keys=[]))
            part['atoms'].append(atoms.cpu().short())
            part['coeffs'].append(coeffs.cpu().float())
            part['text_ids'].append(torch.tensor([r.ids for r in tokenizer.encode_batch(batch['captions'])], dtype=torch.int16))
            part['captions'].extend(batch['captions'])
            part['keys'].extend(batch['keys'])
            seen += len(batch['captions'])
    assert not buffers
    write_json(args.output/f'rank{rank}-complete.json', dict(rank=rank, items=seen, world=world))


if __name__ == '__main__':
    main()
