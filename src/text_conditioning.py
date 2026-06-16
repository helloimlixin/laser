"""Small character-level text conditioning helpers for stage-2 priors."""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Sequence

import torch


TEXT_PAD_ID = 0
TEXT_UNK_ID = 1
DEFAULT_TEXT_VOCAB_CHARS = "".join(chr(i) for i in range(32, 127))


def normalize_text(text: object) -> str:
    raw = "" if text is None else str(text)
    raw = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii")
    raw = raw.lower()
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw


def text_vocab_size(vocab_chars: str | Sequence[str] | None = None) -> int:
    chars = DEFAULT_TEXT_VOCAB_CHARS if vocab_chars is None else "".join(vocab_chars)
    return 2 + len(chars)


def encode_texts(
    texts: Iterable[object],
    *,
    max_length: int,
    vocab_chars: str | Sequence[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    max_length = max(1, int(max_length))
    chars = DEFAULT_TEXT_VOCAB_CHARS if vocab_chars is None else "".join(vocab_chars)
    stoi = {ch: idx + 2 for idx, ch in enumerate(chars)}

    normalized = [normalize_text(text) for text in texts]
    tokens = torch.full((len(normalized), max_length), TEXT_PAD_ID, dtype=torch.long)
    mask = torch.zeros((len(normalized), max_length), dtype=torch.bool)
    for row, text in enumerate(normalized):
        for col, ch in enumerate(text[:max_length]):
            tokens[row, col] = int(stoi.get(ch, TEXT_UNK_ID))
            mask[row, col] = True
    return tokens, mask, normalized


def cycle_prompts(prompts: Sequence[object], n: int) -> list[str]:
    if not prompts:
        prompts = [""]
    n = max(0, int(n))
    return [normalize_text(prompts[idx % len(prompts)]) for idx in range(n)]


def encode_prompts_from_cache(
    prompts: Sequence[object],
    *,
    cache: dict,
    n: int,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    meta = dict(cache.get("meta", {}) or {})
    max_length = int(meta.get("text_max_length", 160) or 160)
    vocab_chars = str(meta.get("text_vocab_chars", DEFAULT_TEXT_VOCAB_CHARS) or DEFAULT_TEXT_VOCAB_CHARS)
    selected = cycle_prompts(list(prompts), n)
    tokens, mask, normalized = encode_texts(selected, max_length=max_length, vocab_chars=vocab_chars)
    if device is not None:
        tokens = tokens.to(device=device)
        mask = mask.to(device=device)
    return tokens, mask, normalized
