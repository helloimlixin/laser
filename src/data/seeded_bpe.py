"""Released CharBPE vocabulary with reproducible merge dropout.

Hugging Face's BPE dropout uses an unexposed Rust RNG. Here the same merge
dropout procedure uses a local Python RNG seeded by epoch and image index, so
an optimizer-boundary restart reproduces every future caption token sequence.
"""
import heapq
import json
import random
import re
from types import SimpleNamespace

from scripts.tools.build_cc3m_compound_cache import text_tokenizer


class SeededBPE:
    def __init__(self, dropout=0.1):
        self.tokenizer = text_tokenizer(0.)
        config = json.loads(self.tokenizer.to_str())
        self.vocab = config['model']['vocab']
        self.unknown = self.vocab[config['model']['unk_token']]
        self.suffix = config['model'].get('end_of_word_suffix') or ''
        merges = config['model']['merges']
        self.ranks = {tuple(pair.split() if isinstance(pair, str) else pair): i
                      for i, pair in enumerate(merges)}
        self.pad = self.tokenizer.token_to_id('[PAD]')
        self.dropout = dropout
        self.special = {item['content']: item['id'] for item in config['added_tokens']}
        self.special_pattern = re.compile('(' + '|'.join(re.escape(t) for t in self.special) + ')')

    def word(self, word, rng):
        pieces = list(word)
        if not pieces:
            return []
        pieces[-1] += self.suffix
        # Vocabulary misses become UNK before merging, as in the Rust model.
        pieces = [p if p in self.vocab else '[UNK]' for p in pieces]
        length = len(pieces)
        previous = [i-1 for i in range(length)]
        following = [i+1 for i in range(length)]
        following[-1] = -1
        heap = []

        def push(i):
            if i >= 0 and following[i] >= 0:
                j = following[i]
                pair = (pieces[i], pieces[j])
                if pair in self.ranks:
                    heapq.heappush(heap, (self.ranks[pair], i, j, pair))

        for i in range(length-1):
            push(i)
        skipped = []
        while heap:
            item = heapq.heappop(heap)
            _, i, j, pair = item
            if self.dropout and rng.random() < self.dropout:
                skipped.append(item)
                continue
            for pending in skipped:
                heapq.heappush(heap, pending)
            skipped.clear()
            if following[i] != j or pieces[i] is None or (pieces[i], pieces[j]) != pair:
                continue
            pieces[i] += pieces[j]
            pieces[j] = None
            following[i] = following[j]
            if following[j] >= 0:
                previous[following[j]] = i
            push(previous[i])
            push(i)
        return [self.vocab.get(p, self.unknown) for p in pieces if p is not None]

    def encode(self, text, seed):
        # Retain the released Unicode normalization and Bert pre-tokenization.
        rng = random.Random(seed)
        ids = []
        for part in self.special_pattern.split(text):
            if part in self.special:
                ids.append(self.special[part])
            else:
                normalized = self.tokenizer.normalizer.normalize_str(part)
                words = self.tokenizer.pre_tokenizer.pre_tokenize_str(normalized)
                ids.extend(token for word, _ in words for token in self.word(word, rng))
        ids = ids[:32]
        return SimpleNamespace(ids=ids + [self.pad] * (32-len(ids)))

    def encode_batch(self, captions, *, indices, epoch, seed):
        return [self.encode(caption, (seed << 64) + (epoch << 32) + int(index))
                for caption, index in zip(captions, indices)]
