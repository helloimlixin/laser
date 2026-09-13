"""Portable encoder/decoder for the validation-selected compact audio candidate."""
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from src.audio_scaled_quantizer import nearest_centers, pack_integer_frames, unpack_integer_frames
from src.models.laser import LASER
from src.scaled_atom_rq import ScaledAtomRQ
from src.tts_pairing import file_sha
from archive.scripts.benchmark_mdctcodec_vctk import align_mdct


class CompactAudioCodec:
    """Two joint sparse IDs plus one small residual-vector ID per audio frame.

    EOS belongs to a future prior, not to this codec's three integer fields.
    The frozen backbone is loaded separately so its provenance stays explicit.
    """
    def __init__(self, codec_checkpoint, quantizer_checkpoint, device='cuda:0'):
        self.device = torch.device(device)
        spec = torch.load(quantizer_checkpoint, map_location='cpu', weights_only=True)
        if file_sha(codec_checkpoint) != spec['codec_sha256']:
            raise ValueError('Frozen audio backbone differs from the calibrated checkpoint')
        self.spec = spec
        self.model = LASER.load_from_checkpoint(str(codec_checkpoint), map_location='cpu').to(device).eval()
        self.model.requires_grad_(False)
        dictionary = F.normalize(self.model.bottleneck.effective_dictionary().detach().float(), dim=0, eps=1e-8)
        torch.testing.assert_close(dictionary.cpu(), spec['dictionary'], rtol=0, atol=0)
        self.quantizer = ScaledAtomRQ(dictionary, spec['levels'].to(device), depth=2)
        self.refinement = spec['refinement'].to(device)
        self.vocab_sizes = spec['vocab_sizes']
        if self.vocab_sizes != [self.quantizer.vocab_size]*2 + [len(self.refinement)]:
            raise ValueError('Integer vocabulary differs from the saved quantizer')

    @torch.inference_mode()
    def decode(self, codes, samples=None):
        if torch.is_tensor(codes):
            if codes.dtype not in (torch.int16, torch.int32, torch.int64, torch.uint8):
                raise ValueError('Expected integer codes')
            fields = codes.detach().cpu().numpy()
        else:
            fields = np.asarray(codes)
        payload = pack_integer_frames(fields, self.vocab_sizes)
        return self.decode_payload(payload, len(fields), samples), payload

    @torch.inference_mode()
    def decode_payload(self, payload, frames, samples=None):
        codes = torch.from_numpy(unpack_integer_frames(payload, self.vocab_sizes, frames)).to(self.device)
        if not frames:
            if samples not in (None, 0): raise ValueError('Nonempty audio requested from an empty payload')
            return np.empty(0, dtype=np.float32)
        latent = self.quantizer.embed(codes[:, :2]).sum(-2) + self.refinement[codes[:, 2]]
        wave = self.model.decode(latent.T[None, :, None])[0, 0].float().cpu().numpy()
        if samples is not None:
            if not 0 < samples <= len(wave): raise ValueError('Original sample count exceeds the decoded waveform')
            wave = wave[:samples]
        if not np.isfinite(wave).all(): raise RuntimeError('Nonfinite reconstruction')
        return wave.clip(-1, 1)

    @torch.inference_mode()
    def encode(self, waveform):
        wave = torch.as_tensor(waveform, dtype=torch.float32, device=self.device)
        if wave.ndim != 1 or not len(wave) or not torch.isfinite(wave).all():
            raise ValueError('Expected a finite nonempty mono48k waveform')
        aligned, samples = align_mdct(wave[None, None])
        latent = self.model._to_bottleneck_input(self.model.pre_bottleneck(self.model.encoder(aligned)))
        z = latent[0, :, 0].T.contiguous(); chunks = []
        for part in z.split(2048):
            result = self.quantizer.quantize(part)
            correction = nearest_centers(part-result['quantized'], self.refinement)
            chunks.append(torch.cat((result['codes'], correction[:, None]), dim=1))
        codes = torch.cat(chunks)
        payload = pack_integer_frames(codes.cpu().numpy(), self.vocab_sizes)
        return codes, payload, {'samples': samples, 'frames': len(codes), 'sample_rate': 48000}
