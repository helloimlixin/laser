"""Full-utterance data and transcript-disjoint splits for the LASER TTS pilot."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import random
import re

import torch
from torch.utils.data import Dataset, Sampler

CODEC_HELDOUT_SPEAKERS = {'p360', 'p361', 'p362', 'p363', 'p364', 'p374', 'p376', 's5'}


def text_key(text):
    return ' '.join(re.findall(r"[a-z0-9']+", text.lower()))


def text_split(text, seed=20260912):
    value = int(hashlib.sha256(f'{seed}:{text_key(text)}'.encode()).hexdigest()[:8], 16) / 2**32
    return 'validation' if value < .03 else 'test' if value < .06 else 'train'


def phonemize_texts(texts):
    from phonemizer import phonemize
    from phonemizer.separator import Separator
    return phonemize(texts, language='en-gb', backend='espeak', strip=True,
                     preserve_punctuation=True, with_stress=True,
                     separator=Separator(phone=' ', word=' | ', syllable=''), njobs=1)


class TTSDataset(Dataset):
    def __init__(self, cache, split, limit=0):
        self.records = [r for r in cache['records'] if r['split'] == split]
        if limit:
            self.records = self.records[:limit]
        self.phone_to_id = cache['phone_to_id']
        self.speaker_to_id = cache['speaker_to_id']
        self.lengths = [len(r['codes']) + 1 for r in self.records]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        phones = [self.phone_to_id.get(p, 1) for p in r['phonemes'].split()] + [2]
        return {'codes': r['codes'].long(), 'phones': torch.tensor(phones),
                'speaker': self.speaker_to_id[r['speaker']], 'record': r}


def collate_tts(items):
    frames = max(len(x['codes']) for x in items)
    text = max(len(x['phones']) for x in items)
    codes = torch.zeros(len(items), frames, 4, dtype=torch.long)
    phones = torch.zeros(len(items), text, dtype=torch.long)
    for i, item in enumerate(items):
        codes[i, :len(item['codes'])] = item['codes']
        phones[i, :len(item['phones'])] = item['phones']
    return {'codes': codes, 'phones': phones,
            'lengths': torch.tensor([len(x['codes']) for x in items]),
            'text_lengths': torch.tensor([len(x['phones']) for x in items]),
            'speakers': torch.tensor([x['speaker'] for x in items])}


class FrameBatchSampler(Sampler):
    """Length buckets with a fixed padded-frame budget, no discarded examples."""
    def __init__(self, lengths, frame_budget=8192, max_batch=16, shuffle=True, seed=1234):
        self.lengths, self.frame_budget = list(lengths), frame_budget
        self.max_batch, self.shuffle, self.seed = max_batch, shuffle, seed
        self.epoch = 0

    def batches(self):
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(indices)
        ordered = []
        for start in range(0, len(indices), 256):
            ordered.extend(sorted(indices[start:start + 256], key=lambda i: self.lengths[i]))
        batches, batch, longest = [], [], 0
        for index in ordered:
            proposed = max(longest, self.lengths[index])
            if batch and (proposed * (len(batch) + 1) > self.frame_budget or len(batch) >= self.max_batch):
                batches.append(batch); batch, longest = [], 0
            batch.append(index); longest = max(longest, self.lengths[index])
        if batch:
            batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self):
        return iter(self.batches())

    def __len__(self):
        return len(self.batches())
