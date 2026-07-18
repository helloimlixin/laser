"""Text conditioning helpers for stage-2 priors."""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import torch


TEXT_PAD_ID = 0
TEXT_UNK_ID = 1
DEFAULT_TEXT_VOCAB_CHARS = "".join(chr(i) for i in range(32, 127))
RQ_BPE16K_TOKENIZER = "rq_bpe16k"
CHAR_TOKENIZER = "char"
RQ_BPE16K_VOCAB_SIZE = 16384
RQ_BPE16K_PAD_ID = 0
RQ_BPE16K_UNK_ID = 1
RQ_BPE16K_ASSET_DIR = Path(__file__).resolve().parents[1] / "third_party" / "rqvae_tokenizers"


def normalize_text(text: object) -> str:
    raw = "" if text is None else str(text)
    raw = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii")
    raw = raw.lower()
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw


def text_vocab_size(vocab_chars: str | Sequence[str] | None = None) -> int:
    chars = DEFAULT_TEXT_VOCAB_CHARS if vocab_chars is None else "".join(vocab_chars)
    return 2 + len(chars)


def _normalize_tokenizer_name(name: str | None) -> str:
    text = str(name or CHAR_TOKENIZER).strip().lower()
    aliases = {
        "": CHAR_TOKENIZER,
        "ascii": CHAR_TOKENIZER,
        "character": CHAR_TOKENIZER,
        "characters": CHAR_TOKENIZER,
        "bpe16k": RQ_BPE16K_TOKENIZER,
        "bpe16k_huggingface": RQ_BPE16K_TOKENIZER,
        "rqvae_bpe16k": RQ_BPE16K_TOKENIZER,
    }
    return aliases.get(text, text)


@lru_cache(maxsize=1)
def _rq_bpe16k_tokenizer():
    try:
        from tokenizers import CharBPETokenizer
    except ImportError as exc:
        raise ImportError(
            "tokenizers is required for tokenizer='rq_bpe16k'. Install it with `pip install tokenizers`."
        ) from exc
    vocab = RQ_BPE16K_ASSET_DIR / "bpe-16k-vocab.json"
    merges = RQ_BPE16K_ASSET_DIR / "bpe-16k-merges.txt"
    if not vocab.exists() or not merges.exists():
        raise FileNotFoundError(f"Missing RQ BPE16k tokenizer assets in {RQ_BPE16K_ASSET_DIR}")
    tok = CharBPETokenizer.from_file(
        str(vocab),
        str(merges),
        unk_token="[UNK]",
        lowercase=True,
    )
    tok.add_special_tokens(["[PAD]"])
    return tok


def encode_texts(
    texts: Iterable[object],
    *,
    max_length: int,
    vocab_chars: str | Sequence[str] | None = None,
    tokenizer: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    max_length = max(1, int(max_length))
    tokenizer_name = _normalize_tokenizer_name(tokenizer)
    if tokenizer_name == RQ_BPE16K_TOKENIZER:
        tok = _rq_bpe16k_tokenizer()
        tok.enable_padding(length=max_length, pad_id=RQ_BPE16K_PAD_ID, pad_token="[PAD]")
        tok.enable_truncation(max_length=max_length)
        normalized = [normalize_text(text) for text in texts]
        tokens = torch.full((len(normalized), max_length), RQ_BPE16K_PAD_ID, dtype=torch.long)
        mask = torch.zeros((len(normalized), max_length), dtype=torch.bool)
        for row, text in enumerate(normalized):
            ids = tok.encode(text).ids[:max_length]
            if ids:
                row_ids = torch.as_tensor(ids, dtype=torch.long)
                tokens[row, : int(row_ids.numel())] = row_ids
                mask[row, : int(row_ids.numel())] = row_ids.ne(RQ_BPE16K_PAD_ID)
        return tokens, mask, normalized
    if tokenizer_name != CHAR_TOKENIZER:
        raise ValueError(f"Unsupported text tokenizer: {tokenizer!r}")

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
    tokenizer = str(meta.get("text_tokenizer", CHAR_TOKENIZER) or CHAR_TOKENIZER)
    vocab_chars = str(meta.get("text_vocab_chars", DEFAULT_TEXT_VOCAB_CHARS) or DEFAULT_TEXT_VOCAB_CHARS)
    selected = cycle_prompts(list(prompts), n)
    tokens, mask, normalized = encode_texts(
        selected,
        max_length=max_length,
        vocab_chars=vocab_chars,
        tokenizer=tokenizer,
    )
    if device is not None:
        tokens = tokens.to(device=device)
        mask = mask.to(device=device)
    return tokens, mask, normalized
