"""A hard 6000-bit/s packet budget at 48 kHz, including a 16-byte header.

LASER preserves four distinct atoms and their nine coefficient symbols. Sorting
atom/coefficient pairs is lossless because their latent contributions are summed.
Combinatorial ranking packs the support set and coefficients into 57 bits/frame.
RVQ uses four 10-bit integers (40 bits/frame). Neither format needs entropy fitting.
"""
from functools import lru_cache
import math
import struct
import zlib

import numpy as np

SAMPLE_RATE = 48000
RATE_BPS = 6000
HEADER_BYTES = 16
MIN_SAMPLES = 1536  # 32 ms: enough for a header and at least one LASER frame
SUPPORT_COUNT = math.comb(4096, 4)
COEFFICIENT_PATTERNS = 9**4
LASER_WORDS = SUPPORT_COUNT * COEFFICIENT_PATTERNS
FRAME_BITS = {'laser':(LASER_WORDS-1).bit_length(), 'rvq':40}
MAGIC = {'laser':b'L6K4', 'rvq':b'R6K4'}
assert FRAME_BITS['laser'] == 57


def max_packet_bytes(samples):
    if not isinstance(samples,(int,np.integer)) or not MIN_SAMPLES <= samples <= 0xffffffff:
        raise ValueError('Hard6k supports 1536..2^32-1 samples of mono 48 kHz audio')
    return int(samples)//64  # floor(samples * 6000 / 48000 / 8), exact integers


def frame_budget(samples, arm):
    return ((max_packet_bytes(samples)-HEADER_BYTES)*8)//FRAME_BITS[arm]


def samples_for_frames(frames,arm):
    """Deterministic generated-audio duration, including header and byte padding."""
    if not isinstance(frames,(int,np.integer)) or frames<1:raise ValueError('At least one coded frame is required')
    samples=max(MIN_SAMPLES,64*(HEADER_BYTES+(int(frames)*FRAME_BITS[arm]+7)//8))
    if frame_budget(samples,arm)!=frames:raise ValueError('Frame count cannot be represented')
    return samples


def pack_tts_codes(codes,arm,samples=None):
    """LASER uses four alternating atom/bin pairs; RVQ uses four book IDs."""
    codes=np.asarray(codes)
    fields=8 if arm=='laser' else 4
    if codes.ndim!=2 or codes.shape[1]!=fields:raise ValueError(f'Expected {fields} integer fields')
    samples=samples_for_frames(len(codes),arm) if samples is None else int(samples)
    return pack_packet(arm,samples,codes[:,0::2],codes[:,1::2]) if arm=='laser' else pack_packet(arm,samples,codes)


@lru_cache(maxsize=1)
def combinations():
    return np.array([[math.comb(a,depth) for a in range(4097)] for depth in range(1,5)],dtype=np.uint64)


def sparse_words(support, bins):
    support,bins=np.asarray(support),np.asarray(bins)
    if support.ndim!=2 or support.shape[1]!=4 or bins.shape!=support.shape:
        raise ValueError('Need matching frames-by-four atom and coefficient matrices')
    if not all(np.issubdtype(x.dtype,np.integer) for x in [support,bins]):
        raise ValueError('Integer atom and coefficient IDs required')
    if ((support<0)|(support>=4096)).any() or ((bins<0)|(bins>=9)).any():
        raise ValueError('Sparse ID outside vocabulary')
    order=np.argsort(support,axis=1,kind='stable')
    atoms=np.take_along_axis(support,order,axis=1)
    coeffs=np.take_along_axis(bins,order,axis=1).astype(np.uint64)
    if (np.diff(atoms,axis=1)==0).any(): raise ValueError('OMP support must have four distinct atoms')
    table=combinations()
    rank=sum(table[d,atoms[:,d]] for d in range(4))
    radix=(coeffs*np.array([729,81,9,1],dtype=np.uint64)).sum(1)
    return rank*COEFFICIENT_PATTERNS+radix


def sparse_from_words(words):
    words=np.asarray(words,dtype=np.uint64)
    if (words>=LASER_WORDS).any(): raise ValueError('Reserved combinatorial code')
    remainder=words//COEFFICIENT_PATTERNS
    radix=words%COEFFICIENT_PATTERNS
    table=combinations(); support=np.empty((len(words),4),dtype=np.int64)
    for d in reversed(range(4)):
        atom=np.searchsorted(table[d],remainder,side='right')-1
        support[:,d]=atom
        remainder=remainder-table[d,atom]
    if remainder.any() or (np.diff(support,axis=1)<=0).any(): raise ValueError('Invalid support rank')
    bins=np.stack([(radix//power)%9 for power in [729,81,9,1]],axis=1).astype(np.int64)
    return support,bins


def pack_packet(arm, samples, codes, coefficient_bins=None):
    codes=np.asarray(codes)
    if codes.ndim!=2 or codes.shape[1]!=4 or not np.issubdtype(codes.dtype,np.integer):
        raise ValueError('Need frames-by-four integer codes')
    frames=len(codes); maximum=frame_budget(samples,arm)
    if not 1<=frames<=maximum: raise ValueError('Frame count exceeds the hard 6kbps budget')
    if arm=='laser': words=sparse_words(codes,coefficient_bins)
    else:
        if ((codes<0)|(codes>=1024)).any(): raise ValueError('RVQ ID outside vocabulary')
        words=(codes.astype(np.uint64)<<np.array([30,20,10,0],dtype=np.uint64)).sum(1)
    width=FRAME_BITS[arm]
    shifts=np.arange(width-1,-1,-1,dtype=np.uint64)
    body=np.packbits(((words[:,None]>>shifts)&1).astype(np.uint8).reshape(-1),bitorder='big').tobytes()
    header=struct.pack('<4sII',MAGIC[arm],int(samples),frames)
    packet=header+struct.pack('<I',zlib.crc32(header+body))+body
    if len(packet)>max_packet_bytes(samples): raise RuntimeError('Encoder exceeded hard byte budget')
    return packet


def unpack_packet(packet, expected_arm=None):
    if len(packet)<HEADER_BYTES: raise ValueError('Truncated packet header')
    magic,samples,frames,crc=struct.unpack('<4sIII',packet[:HEADER_BYTES])
    arm=next((a for a,m in MAGIC.items() if m==magic),None)
    if arm is None or (expected_arm is not None and arm!=expected_arm): raise ValueError('Wrong codec format')
    maximum=frame_budget(samples,arm);width=FRAME_BITS[arm];bits_needed=frames*width
    if not 1<=frames<=maximum or len(packet)>max_packet_bytes(samples): raise ValueError('Packet violates hard rate budget')
    body=packet[HEADER_BYTES:]
    if len(body)!=(bits_needed+7)//8 or zlib.crc32(packet[:12]+body)!=crc:
        raise ValueError('Invalid packet length or checksum')
    bits=np.unpackbits(np.frombuffer(body,np.uint8),bitorder='big')
    if bits[bits_needed:].any(): raise ValueError('Nonzero final padding')
    shifts=np.arange(width-1,-1,-1,dtype=np.uint64)
    words=(bits[:bits_needed].reshape(frames,width).astype(np.uint64)<<shifts).sum(1)
    if arm=='laser': codes,coefficients=sparse_from_words(words)
    else:
        codes=((words[:,None]>>np.array([30,20,10,0],dtype=np.uint64))&1023).astype(np.int64)
        coefficients=None
    return {'arm':arm,'samples':samples,'frames':frames,'codes':codes,'coefficient_bins':coefficients}
