#!/usr/bin/env python3
"""Retokenize CC3M caption conditioning in an existing LASER token cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tarfile

import torch
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.text_conditioning import (
    CHAR_TOKENIZER,
    DEFAULT_TEXT_VOCAB_CHARS,
    RQ_BPE16K_PAD_ID,
    RQ_BPE16K_TOKENIZER,
    RQ_BPE16K_VOCAB_SIZE,
    TEXT_PAD_ID,
    encode_texts,
    text_vocab_size,
)


def _parse_source_path(raw: object) -> tuple[str, str]:
    text = str(raw)
    if "::" not in text:
        raise ValueError(f"Expected CC3M source path formatted as shard.tar::member, got {text!r}")
    shard, member = text.split("::", 1)
    shard_path = str(Path(shard).expanduser().resolve())
    member = member.lstrip("/")
    if not shard_path or not member:
        raise ValueError(f"Invalid CC3M source path: {text!r}")
    return shard_path, member


def _member_candidates(image_member: str) -> list[str]:
    path = Path(image_member)
    base = str(path.with_suffix(""))
    return [f"{base}.txt", f"{base}.json"]


def _read_caption(tf: tarfile.TarFile, image_member: str) -> str:
    for candidate in _member_candidates(image_member):
        try:
            member = tf.getmember(candidate)
        except KeyError:
            continue
        extracted = tf.extractfile(member)
        if extracted is None:
            continue
        raw = extracted.read().decode("utf-8", errors="replace")
        if candidate.endswith(".json"):
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                return ""
            return str(payload.get("caption", "") or "").strip()
        return raw.strip()
    return ""


def _read_captions_from_source_paths(source_paths: list[object]) -> list[str]:
    parsed = [_parse_source_path(path) for path in source_paths]
    captions = [""] * len(parsed)
    by_shard: dict[str, list[tuple[int, str]]] = {}
    for idx, (shard, member) in enumerate(parsed):
        by_shard.setdefault(shard, []).append((idx, member))

    for shard, rows in tqdm(by_shard.items(), desc="Reading CC3M captions", unit="shard", dynamic_ncols=True):
        with tarfile.open(shard, "r:*") as tf:
            for idx, member in rows:
                captions[idx] = _read_caption(tf, member)
    return captions


def _normalize_tokenizer_name(name: str) -> str:
    text = str(name or CHAR_TOKENIZER).strip().lower()
    if text in {"bpe16k", "bpe16k_huggingface", "rqvae_bpe16k"}:
        return RQ_BPE16K_TOKENIZER
    return text


def _attach_text(cache: dict, captions: list[str], *, max_length: int, tokenizer: str) -> None:
    total_items = int(cache["tokens_flat"].size(0))
    if len(captions) != total_items:
        raise ValueError(f"Expected {total_items} captions, found {len(captions)}")

    tokenizer = _normalize_tokenizer_name(tokenizer)
    tokens, mask, normalized = encode_texts(
        captions,
        max_length=int(max_length),
        vocab_chars=DEFAULT_TEXT_VOCAB_CHARS,
        tokenizer=tokenizer,
    )
    if tokenizer == RQ_BPE16K_TOKENIZER:
        vocab_size = RQ_BPE16K_VOCAB_SIZE
        pad_id = RQ_BPE16K_PAD_ID
        extra_meta = {
            "text_tokenizer": RQ_BPE16K_TOKENIZER,
            "text_vocab_name": "bpe16k_huggingface",
        }
        cache.get("meta", {}).pop("text_vocab_chars", None)
    elif tokenizer == CHAR_TOKENIZER:
        vocab_size = text_vocab_size(DEFAULT_TEXT_VOCAB_CHARS)
        pad_id = TEXT_PAD_ID
        extra_meta = {
            "text_tokenizer": CHAR_TOKENIZER,
            "text_vocab_chars": DEFAULT_TEXT_VOCAB_CHARS,
        }
        cache.get("meta", {}).pop("text_vocab_name", None)
    else:
        raise ValueError(f"Unsupported tokenizer: {tokenizer!r}")

    preview_count = min(len(normalized), 1024)
    meta = cache.setdefault("meta", {})
    cache["text_tokens"] = tokens.contiguous()
    cache["text_mask"] = mask.contiguous()
    cache["text"] = normalized[:preview_count]
    meta.update(
        {
            "has_text_conditioning": True,
            "text_conditioning": "caption",
            "text_vocab_size": int(vocab_size),
            "text_max_length": int(max_length),
            "text_pad_id": int(pad_id),
            "text_num_captions": int(total_items),
            "text_preview_count": int(preview_count),
            **extra_meta,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Existing token cache with source_paths.")
    parser.add_argument("--output", type=Path, required=True, help="Retokenized token cache output path.")
    parser.add_argument("--text_max_length", type=int, default=32)
    parser.add_argument(
        "--text_tokenizer",
        type=str,
        default=RQ_BPE16K_TOKENIZER,
        choices=[CHAR_TOKENIZER, RQ_BPE16K_TOKENIZER, "bpe16k", "bpe16k_huggingface", "rqvae_bpe16k"],
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input cache not found: {input_path}")
    if output_path.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {output_path}")

    cache = torch.load(input_path, map_location="cpu", weights_only=False)
    source_paths = cache.get("source_paths")
    if not source_paths:
        raise ValueError("Input cache does not contain source_paths.")
    total_items = int(cache["tokens_flat"].size(0))
    if len(source_paths) < total_items:
        raise ValueError(f"Expected at least {total_items} source paths, found {len(source_paths)}")

    captions = _read_captions_from_source_paths(list(source_paths[:total_items]))
    _attach_text(
        cache,
        captions,
        max_length=int(args.text_max_length),
        tokenizer=str(args.text_tokenizer),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output_path)
    print(f"Input cache:  {input_path}")
    print(f"Output cache: {output_path}")
    print(f"Text tokens:  {tuple(cache['text_tokens'].shape)}")
    print(f"Tokenizer:    {cache['meta'].get('text_tokenizer')}")
    print(f"Text length:  {cache['meta'].get('text_max_length')}")


if __name__ == "__main__":
    main()
