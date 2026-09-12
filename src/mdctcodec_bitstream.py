"""Fixed-width LASER audio payload: two (13-bit atom, 7-bit coefficient) pairs.

Each 150 Hz frame occupies exactly five bytes. Model weights, the shared
coefficient bound, and the original waveform length are stream metadata, outside
this raw payload. Signed coefficient integers use codes 0..126 for -63..63.
"""
import numpy as np


def pack_frames(support, coefficients) -> bytes:
    support, coefficients = np.asarray(support), np.asarray(coefficients)
    if support.shape != coefficients.shape or support.ndim < 1 or support.shape[-1] != 2:
        raise ValueError('Expected equally shaped integer arrays with two atoms per frame')
    for array, lower, upper in ((support, 0, 8191), (coefficients, -63, 63)):
        if not np.issubdtype(array.dtype, np.integer) or np.any((array < lower) | (array > upper)):
            raise ValueError(f'Integer tokens must be within [{lower}, {upper}]')
    atoms = support.astype(np.uint64).reshape(-1, 2)
    values = (coefficients.astype(np.int64) + 63).astype(np.uint64).reshape(-1, 2)
    words = (atoms[:, 0] << 27) | (values[:, 0] << 20) | (atoms[:, 1] << 7) | values[:, 1]
    shifts = np.array([32, 24, 16, 8, 0], dtype=np.uint64)
    return ((words[:, None] >> shifts) & 255).astype(np.uint8).tobytes()


def unpack_frames(payload: bytes):
    if len(payload) % 5:
        raise ValueError('Truncated LASER payload: expected five bytes per frame')
    octets = np.frombuffer(payload, dtype=np.uint8).astype(np.uint64).reshape(-1, 5)
    shifts = np.array([32, 24, 16, 8, 0], dtype=np.uint64)
    words = np.bitwise_or.reduce(octets << shifts, axis=1)
    atoms = np.stack([(words >> 27) & 8191, (words >> 7) & 8191], axis=-1).astype(np.int64)
    values = np.stack([(words >> 20) & 127, words & 127], axis=-1).astype(np.int64)
    if np.any(values == 127):
        raise ValueError('Reserved coefficient code 127')
    return atoms, values - 63
