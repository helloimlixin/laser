"""Verified local tar access and reproducible image/caption augmentation."""
from collections import OrderedDict
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from omegaconf import OmegaConf


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def text_tokenizer(dropout=0.):
    from tokenizers import CharBPETokenizer
    from src import rq_upstream
    base = rq_upstream.ROOT/'rqvae/txtimg_datasets/tokenizers/pretrained'
    tok = CharBPETokenizer.from_file(str(base/'bpe-16k-vocab.json'),
        str(base/'bpe-16k-merges.txt'), unk_token='[UNK]', lowercase=True, dropout=dropout)
    tok.add_special_tokens(['[PAD]'])
    tok.enable_padding(length=32, pad_id=tok.token_to_id('[PAD]'))
    tok.enable_truncation(max_length=32)
    assert tok.get_vocab_size() == 16384
    return tok


def stage_shard(spec, source, local):
    """Copy once under a process lock and verify the original download digest."""
    local = Path(local)
    local.mkdir(parents=True, exist_ok=True)
    target = local/spec['name']
    receipt = target.with_suffix('.verified.json')
    with (local/(spec['name']+'.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (receipt.is_file() and target.is_file() and target.stat().st_size == spec['bytes']
                and json.loads(receipt.read_text())['sha256'] == spec['sha256']):
            return target
        # The shared filesystem is most reliable with bounded concurrent reads.
        slots = [(local/f'.copy-{i}.lock').open('a') for i in range(4)]
        selected = None
        try:
            while selected is None:
                for slot in slots:
                    try:
                        fcntl.flock(slot, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        selected = slot
                        break
                    except BlockingIOError:
                        pass
                if selected is None:
                    time.sleep(.1)
            temp = target.with_suffix(f'.{os.getpid()}.partial')
            for attempt in range(4):
                try:
                    digest = hashlib.sha256()
                    # A mounted-file read can otherwise hang while holding
                    # a copy slot and block every training rank. Bound the
                    # copy in a child process, then hash the local bytes.
                    try:
                        subprocess.run(['cp', '--reflink=never', '--',
                            str(Path(source)/spec['name']), str(temp)],
                            check=True, timeout=120, capture_output=True)
                    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
                        raise OSError(f'CC3M shard copy failed: {spec["name"]}: {error}') from error
                    with temp.open('rb') as reader:
                        while chunk := reader.read(8*1024*1024):
                            digest.update(chunk)
                    if digest.hexdigest() != spec['sha256'] or temp.stat().st_size != spec['bytes']:
                        raise ValueError(f'CC3M shard checksum mismatch: {spec["name"]}')
                    temp.replace(target)
                    write_json(receipt, dict(sha256=spec['sha256'], bytes=spec['bytes']))
                    return target
                except OSError:
                    temp.unlink(missing_ok=True)
                    if attempt == 3:
                        raise
                    time.sleep(attempt+1)
        finally:
            for slot in slots:
                slot.close()


def shard_index(spec, source, local):
    path = stage_shard(spec, source, local)
    index = path.with_suffix('.index.json')
    with path.with_suffix('.index.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if index.is_file():
            value = json.loads(index.read_text())
            assert value['sha256'] == spec['sha256'] and len(value['records']) == spec['items']
            return path, value['records']
        records, record, previous = [], {}, None

        def append():
            if not record:
                return
            if 'image' not in record or not record.get('caption'):
                raise ValueError(f'Unpaired image/caption in {path}:{previous}')
            records.append([*record['image'], record['caption']])

        with tarfile.open(path, 'r:') as archive:
            for member in archive:
                if not member.isfile():
                    continue
                stem, suffix = os.path.splitext(member.name)
                if previous is not None and stem != previous:
                    append()
                    record = {}
                previous = stem
                if suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'):
                    record['image'] = (member.offset_data, member.size)
                elif suffix.lower() == '.txt':
                    record['caption'] = archive.extractfile(member).read().decode('utf-8').strip()
            append()
        assert len(records) == spec['items'], (spec['name'], len(records), spec['items'])
        write_json(index, dict(sha256=spec['sha256'], records=records))
        return path, records


def epoch_order(specs, seed, epoch, group_shards=16):
    """A bijection over the split, mixing images within shuffled shard groups."""
    counts = np.array([s['items'] for s in specs], dtype=np.int64)
    starts = np.r_[0, counts.cumsum()[:-1]]
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    shards = rng.permutation(len(specs))
    groups = []
    for first in range(0, len(shards), group_shards):
        rows = np.concatenate([np.arange(starts[i], starts[i]+counts[i])
                               for i in shards[first:first+group_shards]])
        rng.shuffle(rows)
        groups.append(rows)
    return np.concatenate(groups)


class CC3MJointDataset(Dataset):
    def __init__(self, specs, source, local, *, train, seed, epoch=0):
        from rqvae.txtimg_datasets.transforms import create_transforms
        self.specs, self.source, self.local = specs, source, local
        self.ends = np.cumsum([s['items'] for s in specs])
        self.seed, self.epoch, self.train = seed, epoch, train
        self.transform = create_transforms(OmegaConf.create(dict(
            transforms='dalle-vqvae', image_resolution=256)), 'train' if train else 'val')
        self.records = {}
        self.handles = OrderedDict()
        self.tokenizer = None

    def __len__(self):
        return int(self.ends[-1])

    def __getitem__(self, index):
        index = int(index)
        shard = int(np.searchsorted(self.ends, index, side='right'))
        row = index-(int(self.ends[shard-1]) if shard else 0)
        if shard not in self.records:
            self.records[shard] = shard_index(self.specs[shard], self.source, self.local)
        path, records = self.records[shard]
        offset, length, caption = records[row]
        if shard not in self.handles:
            self.handles[shard] = os.open(path, os.O_RDONLY)
            if len(self.handles) > 32:
                _, fd = self.handles.popitem(last=False)
                os.close(fd)
        else:
            self.handles.move_to_end(shard)
        raw = os.pread(self.handles[shard], length, offset)
        assert len(raw) == length
        with Image.open(io.BytesIO(raw)) as image, torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(
                (self.seed + self.epoch*len(self) + index) % 2**63)
            pixels = self.transform(image.convert('RGB'))
        if self.tokenizer is None:
            from src.data.seeded_bpe import SeededBPE
            self.tokenizer = SeededBPE(dropout=.1 if self.train else 0., tokenizer=text_tokenizer())
        text = self.tokenizer.encode(caption, (self.seed << 64)+(self.epoch << 32)+index).ids
        return pixels, torch.tensor(text, dtype=torch.long), caption
