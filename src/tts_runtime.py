"""Synthesis and ASR diagnostics for the frozen MDCTCodec-LASER TTS path."""
from __future__ import annotations
import re

import numpy as np
import torch
import torchaudio

from src.mdctcodec_bitstream import pack_frames, unpack_frames
from src.models.laser import LASER
from src.tts_data import phonemize_texts, text_key
from archive.scripts.benchmark_mdctcodec_trained_rvq import payload_roundtrip


def encode_prompt(text, metadata, device):
    phones = phonemize_texts([text])[0].split()
    if not phones:
        raise ValueError('Text must contain a pronounceable prompt')
    if len(phones) > 400:
        raise ValueError('This pilot supports at most 400 phoneme tokens per utterance')
    ids = [metadata['phone_to_id'].get(p, 1) for p in phones] + [2]
    return torch.tensor([ids], device=device), phones


class CodecDecoder:
    def __init__(self, checkpoint, device='cuda:1'):
        self.device = torch.device(device)
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        hp=saved['hyper_parameters'];self.hard6k=hp.get('hard_rate_cap_bps')==6000
        if self.hard6k:
            from src.mdctcodec_hard6k import HardRateLASER,HardRateRVQ
            cls=HardRateRVQ if hp['bottleneck_type']=='mdctcodec_rvq' else HardRateLASER
            self.model=cls(**hp);self.model.load_state_dict(saved['state_dict'],strict=True)
        else:
            self.model=LASER.load_from_checkpoint(str(checkpoint),map_location='cpu')
        self.model=self.model.to(device).eval()
        self.model.requires_grad_(False)
        self.codec = 'rvq' if self.model.bottleneck_type == 'mdctcodec_rvq' else 'laser'
        if self.hard6k:
            self.bound=None
        elif self.codec == 'rvq':
            assert self.model.bottleneck.num_embeddings == 1024 and self.model.bottleneck.code_depth == 4
            self.bound = None
        else:
            self.bound = self.model.bottleneck.coefficient_quantization_max
            assert self.model.bottleneck.num_embeddings == 8192 and self.model.bottleneck.sparsity_level == 2
            assert self.model.bottleneck.coefficient_quantization_bits == 7

    @torch.inference_mode()
    def decode(self, codes,samples=None):
        if len(codes) == 0:
            return np.zeros(1, dtype=np.float32), b''
        fields = codes.cpu().long().numpy()
        if self.hard6k:
            from src.audio_hard6k_bitstream import pack_tts_codes
            payload=pack_tts_codes(fields,self.codec,samples)
            with torch.autocast(self.device.type,enabled=False):decoded=self.model.decode_packet(payload)
            return decoded[0,0].cpu().numpy(),payload
        if fields.ndim != 2 or fields.shape[1] != 4:
            raise ValueError('Expected four integer fields per frame')
        if self.codec == 'rvq':
            payload, parsed = payload_roundtrip(fields.T[None])
            latent = self.model.bottleneck.quantizer.from_codes(torch.from_numpy(parsed).to(self.device))[0]
            decoded = self.model.decoder(latent)
        else:
            payload = pack_frames(fields[:, 0::2], fields[:, 1::2] - 63)
            atoms, integers = unpack_frames(payload)
            support = torch.from_numpy(atoms).to(self.device)[None, None]
            values = torch.from_numpy(integers).to(self.device)[None, None].float() * (self.bound / 63)
            decoded = self.model.decode_from_atoms_and_coeffs(support, values)
        waveform = decoded[0, 0].float().cpu().numpy()
        if not np.isfinite(waveform).all():
            raise RuntimeError('Decoder produced nonfinite audio')
        return waveform.clip(-1, 1), payload


class ASREvaluator:
    def __init__(self, device='cuda:1'):
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.model = bundle.get_model().to(device).eval()
        self.labels, self.device = bundle.get_labels(), torch.device(device)

    @torch.inference_mode()
    def transcribe(self, audio):
        if len(audio) < 4800:
            return ''
        waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32)).to(self.device)[None]
        waveform = torchaudio.functional.resample(waveform, 48000, 16000)
        emissions, _ = self.model(waveform)
        ids = emissions[0].argmax(-1).unique_consecutive().cpu().tolist()
        return ''.join(self.labels[i] for i in ids if i != 0).replace('|', ' ').strip()


def word_error(reference, hypothesis):
    reference, hypothesis = text_key(reference).split(), text_key(hypothesis).split()
    previous = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, 1):
        current = [i]
        for j, hyp in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ref != hyp)))
        previous = current
    return previous[-1], len(reference)
