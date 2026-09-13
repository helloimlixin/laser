"""Lossless sparse-token transport with training-fitted Huffman tables.

The dictionary and tables are checkpoint metadata. Payloads include frame count,
valid bit count, and CRC. A repeat symbol refers only to the preceding frame at
the same depth; the first frame never repeats. No evaluation frequencies are fit.
"""
import heapq
import struct
import zlib

import numpy as np


class SparseHuffman:
    def __init__(self, counts):
        counts = np.asarray(counts, dtype=np.float64)
        if counts.ndim != 2 or counts.shape[1] < 3 or not np.isfinite(counts).all() or (counts < 0).any():
            raise ValueError('Expected finite, nonnegative depth-by-symbol counts')
        self.depth, symbols = counts.shape
        self.vocab_size = symbols - 1  # last symbol is temporal repeat
        self.books, self.decoders = [], []
        for row in counts:
            heap = [(float(n) + .25, i, i) for i, n in enumerate(row)]
            heapq.heapify(heap); serial = symbols
            while len(heap) > 1:
                a, _, left = heapq.heappop(heap); b, _, right = heapq.heappop(heap)
                heapq.heappush(heap, (a+b, serial, (left, right))); serial += 1
            lengths = [0] * symbols
            stack = [(heap[0][2], 0)]
            while stack:
                node, length = stack.pop()
                if isinstance(node, int): lengths[node] = max(1, length)
                else: stack.extend([(node[0], length+1), (node[1], length+1)])
            book = [None] * symbols; code = previous_length = 0
            for length, symbol in sorted((n, i) for i, n in enumerate(lengths)):
                code <<= length - previous_length
                book[symbol] = (code, length)
                code += 1; previous_length = length
            self.books.append(book)
            self.decoders.append({(n, c): i for i, (c, n) in enumerate(book)})

    def encode(self, tokens):
        tokens = np.asarray(tokens)
        if tokens.ndim != 2 or tokens.shape[1] != self.depth or not np.issubdtype(tokens.dtype, np.integer):
            raise ValueError('Expected integer frame-by-depth tokens')
        if ((tokens < 0) | (tokens >= self.vocab_size)).any():
            raise ValueError('Token outside vocabulary')
        previous = np.full(self.depth, -1, dtype=np.int64)
        output = bytearray(); buffer = available = total = 0
        for row in tokens:
            for d, token in enumerate(row):
                symbol = self.vocab_size if token == previous[d] else int(token)
                previous[d] = token
                code, length = self.books[d][symbol]
                buffer = (buffer << length) | code; available += length; total += length
                while available >= 8:
                    available -= 8; output.append((buffer >> available) & 255)
                buffer &= (1 << available) - 1
        if available: output.append(buffer << (8 - available))
        body = bytes(output)
        header = struct.pack('<II', len(tokens), total)
        return header + struct.pack('<I', zlib.crc32(header + body)) + body

    def decode(self, payload):
        if len(payload) < 12: raise ValueError('Truncated Huffman header')
        frames, valid_bits, crc = struct.unpack('<III', payload[:12]); body = payload[12:]
        if len(body) != (valid_bits+7)//8 or zlib.crc32(payload[:8]+body) != crc:
            raise ValueError('Invalid Huffman length or checksum')
        if valid_bits < frames*self.depth:
            raise ValueError('Not enough bits for declared frame count')
        if valid_bits % 8 and body[-1] & ((1 << (8-valid_bits % 8))-1):
            raise ValueError('Nonzero final padding')
        result = np.empty((frames, self.depth), dtype=np.int64)
        position = 0
        for f in range(frames):
            for d in range(self.depth):
                code = length = 0; symbol = None
                while symbol is None:
                    if position >= valid_bits: raise ValueError('Truncated Huffman symbol')
                    bit = (body[position//8] >> (7-position % 8)) & 1
                    position += 1; length += 1; code = (code << 1) | bit
                    symbol = self.decoders[d].get((length, code))
                if symbol == self.vocab_size:
                    if f == 0: raise ValueError('Repeat without a previous frame')
                    symbol = result[f-1, d]
                result[f, d] = symbol
        if position != valid_bits: raise ValueError('Unexpected trailing Huffman symbols')
        return result
