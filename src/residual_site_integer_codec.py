"""Factorized complete-site integers with an explicit, bounded bit budget.

Packing changes storage, not the number of decisions a predictor must model.
Residual decoding receives only the integer and fixed codebooks. A caller can
project its dense sum into a complete sparse code using a frozen dictionary.
"""
from __future__ import annotations

import operator

import torch

from src.coefficient_pattern_codec import assign_coefficient_patterns, fit_coefficient_patterns


def _widths(widths):
    widths = tuple(operator.index(w) for w in widths)
    if not widths or min(widths) < 1 or sum(widths) > 63:
        raise ValueError('Fields must use between 1 and 63 total bits')
    return widths


def _integers(values):
    if values.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError('Expected an integer tensor')
    if values.numel() and int(values.min()) < 0:
        raise ValueError('Integer fields must be nonnegative')


def pack_fields(fields, widths):
    """Pack nonnegative fields into one signed-int64-compatible integer per site."""
    widths = _widths(widths)
    _integers(fields)
    if fields.ndim < 1 or fields.shape[-1] != len(widths):
        raise ValueError('Last dimension must match the number of fields')
    packed = torch.zeros(fields.shape[:-1], dtype=torch.long, device=fields.device)
    shift = 0
    for column, width in enumerate(widths):
        value = fields[..., column].long()
        if value.numel() and int(value.max()) > (1 << width) - 1:
            raise ValueError('Field exceeds its allocated bit width')
        packed.bitwise_or_(value << shift)
        shift += width
    return packed


def unpack_fields(ids, widths):
    widths = _widths(widths)
    _integers(ids)
    if ids.numel() and int(ids.max()) > (1 << sum(widths)) - 1:
        raise ValueError('Site integer exceeds the allocated bit budget')
    values, shift = [], 0
    for width in widths:
        values.append((ids.long() >> shift) & ((1 << width) - 1))
        shift += width
    return torch.stack(values, -1)


def _book_widths(codebooks):
    if codebooks.ndim != 3 or min(codebooks.shape) < 1 or not torch.isfinite(codebooks).all():
        raise ValueError('Expected finite codebooks [stages, vocabulary, channels]')
    size = codebooks.shape[1]
    if size < 2 or size & (size - 1):
        raise ValueError('Residual vocabulary must be a power of two')
    return _widths([size.bit_length() - 1] * len(codebooks))


@torch.no_grad()
def encode_residual_fields(vectors, codebooks, *, chunk_size=4096):
    _book_widths(codebooks)
    if vectors.shape[-1] != codebooks.shape[-1]:
        raise ValueError('Latent channels and codebooks must agree')
    residual = vectors.reshape(-1, vectors.shape[-1]).clone()
    fields = []
    for book in codebooks:
        ids, _ = assign_coefficient_patterns(residual, book, chunk_size=chunk_size)
        fields.append(ids.reshape(vectors.shape[:-1]))
        residual -= book[ids]
    return torch.stack(fields, -1)


def decode_residual_ids(ids, codebooks):
    """Recover a dense latent sum using the integer alone, without source support."""
    fields = unpack_fields(ids, _book_widths(codebooks))
    result = torch.zeros((*ids.shape, codebooks.shape[-1]), device=codebooks.device, dtype=codebooks.dtype)
    for stage, book in enumerate(codebooks):
        result = result + book[fields[..., stage]]
    return result


@torch.no_grad()
def fit_residual_codebooks(vectors, *, stages=7, vocabulary=512, iterations=8,
                          seed=9711, chunk_size=4096, progress=None):
    if vectors.ndim != 2 or not 0 < vocabulary - 1 <= len(vectors):
        raise ValueError('Expected enough training vectors for the codebook')
    if vocabulary < 2 or vocabulary & (vocabulary - 1):
        raise ValueError('Vocabulary must be a power of two')
    _widths([vocabulary.bit_length() - 1] * stages)
    residual, books = vectors.clone(), []
    for stage in range(stages):
        callback = None if progress is None else lambda iteration, error, empty: progress(stage + 1, iteration, error, empty)
        centers = fit_coefficient_patterns(residual, num_patterns=vocabulary - 1,
            iterations=iterations, seed=seed + stage, chunk_size=chunk_size, progress=callback)
        # A zero residual is always available; greedy dense distortion cannot
        # increase when another stage is added, including on unseen vectors.
        book = torch.cat((vectors.new_zeros(1, vectors.shape[1]), centers))
        ids, _ = assign_coefficient_patterns(residual, book, chunk_size=chunk_size)
        residual -= book[ids]
        books.append(book)
    return torch.stack(books)
