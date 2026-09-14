"""Phoneme-conditioned, frame-autoregressive LASER/RVQ codec language model.

The temporal transformer predicts one coded frame at a time. A small recurrent
depth decoder is retained for old LASER checkpoints. New paired priors use a
causal depth transformer over alternating atom/coefficient fields for LASER
(four for K2, eight for K4), or four residual codebook IDs for RVQ. Head zero also predicts
EOS, which is not part of either codec payload.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class TTSConfig:
    phone_vocab: int
    speakers: int
    width: int = 512
    heads: int = 8
    text_layers: int = 3
    audio_layers: int = 8
    dropout: float = .1
    codec: str = 'laser'
    depth_layers: int = 0  # Zero preserves the original GRU checkpoints.
    laser_atoms: int = 8192
    laser_sparsity: int = 2
    coefficient_levels: int = 127
    depth_positions: int = 4
    hard_rate_cap_bps: int = 0


def position_encoding(length, width, device, offset=0):
    positions = torch.arange(offset, offset + length, device=device).float()[:, None]
    frequencies = torch.exp(torch.arange(0, width, 2, device=device).float() * (-math.log(10000.0) / width))
    angles = positions * frequencies
    return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2)[None]


class Attention(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.heads, self.head_width, self.dropout = heads, width // heads, dropout
        self.q, self.k, self.v, self.out = [nn.Linear(width, width) for _ in range(4)]

    def split(self, x):
        return x.view(x.shape[0], x.shape[1], self.heads, self.head_width).transpose(1, 2)

    def kv(self, memory):
        return self.split(self.k(memory)), self.split(self.v(memory))

    def forward(self, x, memory=None, mask=None, causal=False, cache=None, static_kv=None, return_weights=False):
        q = self.split(self.q(x))
        k, v = self.kv(x if memory is None else memory) if static_kv is None else static_kv
        if cache is not None and cache.get('k') is not None:
            k, v = torch.cat((cache['k'], k), dim=2), torch.cat((cache['v'], v), dim=2)
        if cache is not None:
            cache.update(k=k, v=v)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
            is_causal=causal and cache is None, dropout_p=self.dropout if self.training else 0.0)
        y = self.out(y.transpose(1, 2).contiguous().flatten(2))
        weights = None
        if return_weights:
            # One alignment head; other heads can attend to wider text context.
            scores = q[:, 0].float() @ k[:, 0].float().transpose(-2, -1) / math.sqrt(self.head_width)
            if mask is not None:
                scores = scores.masked_fill(~mask[:, 0], -1e4)
            weights = scores.softmax(-1)
        return y, weights


class DecoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(cfg.width) for _ in range(3)])
        self.self_attention = Attention(cfg.width, cfg.heads, cfg.dropout)
        self.cross_attention = Attention(cfg.width, cfg.heads, cfg.dropout)
        self.ff = nn.Sequential(nn.Linear(cfg.width, cfg.width * 4), nn.GELU(),
                                nn.Dropout(cfg.dropout), nn.Linear(cfg.width * 4, cfg.width))
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, memory, text_mask, cache=None, static_kv=None, alignment=False):
        y, _ = self.self_attention(self.norms[0](x), causal=True, cache=cache)
        x = x + self.drop(y)
        y, weights = self.cross_attention(self.norms[1](x), memory=memory,
            mask=text_mask[:, None, None, :], static_kv=static_kv, return_weights=alignment)
        x = x + self.drop(y)
        return x + self.drop(self.ff(self.norms[2](x))), weights


class DepthBlock(nn.Module):
    """Causal attention over the fields of one frame; never over future fields."""
    def __init__(self, cfg):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(cfg.width), nn.LayerNorm(cfg.width)
        self.attention = Attention(cfg.width, cfg.heads, cfg.dropout)
        self.ff = nn.Sequential(nn.Linear(cfg.width, 4 * cfg.width), nn.GELU(),
                                nn.Dropout(cfg.dropout), nn.Linear(4 * cfg.width, cfg.width))
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cache=None):
        y, _ = self.attention(self.norm1(x), causal=True, cache=cache)
        x = x + self.drop(y)
        return x + self.drop(self.ff(self.norm2(x)))


class LaserTTS(nn.Module):
    EOS = 8192
    vocab_sizes = (8193, 127, 8192, 127)

    def __init__(self, cfg: TTSConfig):
        super().__init__()
        if cfg.width % cfg.heads or cfg.width % 2:
            raise ValueError('Width must be even and divisible by attention heads')
        if cfg.codec not in ('laser', 'rvq') or cfg.depth_layers < 0:
            raise ValueError('Expected laser/rvq codec and nonnegative depth_layers')
        self.cfg = cfg
        self.EOS = cfg.laser_atoms if cfg.codec == 'laser' else 1024
        vocab=[cfg.laser_atoms,cfg.coefficient_levels]*cfg.laser_sparsity if cfg.codec=='laser' else [1024]*4
        vocab[0]+=1;self.vocab_sizes=tuple(vocab);self.field_count=len(vocab)
        if cfg.depth_positions<self.field_count:raise ValueError('Insufficient depth positions')
        if cfg.hard_rate_cap_bps not in (0,6000):raise ValueError('Unsupported hard rate')
        if cfg.hard_rate_cap_bps and cfg.codec=='laser' and (cfg.laser_atoms,cfg.laser_sparsity,cfg.coefficient_levels)!=(4096,4,9):
            raise ValueError('Hard6k LASER requires K4, 4096 atoms and nine coefficient symbols')
        self.phones = nn.Embedding(cfg.phone_vocab, cfg.width, padding_idx=0)
        self.speakers = nn.Embedding(cfg.speakers, cfg.width)
        layer = nn.TransformerEncoderLayer(cfg.width, cfg.heads, 4 * cfg.width, cfg.dropout,
                                           activation='gelu', batch_first=True, norm_first=True)
        self.text_encoder = nn.TransformerEncoder(layer, cfg.text_layers, norm=nn.LayerNorm(cfg.width), enable_nested_tensor=False)
        self.fields = nn.ModuleList([nn.Embedding(size, cfg.width) for size in self.vocab_sizes])
        self.bos = nn.Parameter(torch.zeros(1, 1, cfg.width))
        self.frame_norm = nn.LayerNorm(cfg.width)
        self.blocks = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.audio_layers)])
        self.out_norm = nn.LayerNorm(cfg.width)
        if cfg.depth_layers:
            self.depth_blocks = nn.ModuleList([DepthBlock(cfg) for _ in range(cfg.depth_layers)])
            self.depth_position = nn.Parameter(torch.randn(1, cfg.depth_positions, cfg.width) * .02)
        else:
            self.depth_cell = nn.GRUCell(cfg.width, cfg.width)
        self.depth_norm = nn.LayerNorm(cfg.width)
        self.heads = nn.ModuleList([nn.Linear(cfg.width, size) for size in self.vocab_sizes])
        nn.init.normal_(self.bos, std=.02)
        with torch.no_grad():
            self.heads[0].bias[self.EOS] = -4.0

    def text_memory(self, phones, speakers):
        mask = phones.ne(0)
        if not torch.all(mask.any(1)):
            raise ValueError('Every input must contain phonemes or the text EOS token')
        x = self.phones(phones) + position_encoding(phones.shape[1], self.cfg.width, phones.device)
        x = x + self.speakers(speakers)[:, None]
        return self.text_encoder(x, src_key_padding_mask=~mask), mask

    def frame_embedding(self, codes):
        return self.frame_norm(sum(self.fields[d](codes[..., d]) for d in range(self.field_count)) / math.sqrt(self.field_count))

    def temporal(self, codes, memory, text_mask, speakers, alignment=False):
        x = torch.cat((self.bos.expand(len(codes), -1, -1), self.frame_embedding(codes)), dim=1)
        x = x + position_encoding(x.shape[1], self.cfg.width, x.device).to(x.dtype)
        x = x + self.speakers(speakers)[:, None]
        weights = None
        for i, block in enumerate(self.blocks):
            x, weights = block(x, memory, text_mask, alignment=alignment and i == len(self.blocks) - 1)
        return self.out_norm(x), weights

    def depth_logits(self, contexts, teacher_codes):
        # Context at t contains audio frames strictly before t. Ground-truth
        # fields only condition later fields within that same target frame.
        state = contexts.flatten(0, 1)
        teacher = teacher_codes.flatten(0, 1)
        if self.cfg.depth_layers:
            prefix = torch.stack([self.fields[d](teacher[:, d]) for d in range(self.field_count-1)], dim=1).cumsum(1)
            x = torch.cat((state[:, None], prefix), dim=1) + self.depth_position[:,:self.field_count]
            for block in self.depth_blocks:
                x = block(x)
            states = self.depth_norm(x)
        result = []
        for d, head in enumerate(self.heads):
            logits = head(states[:, d] if self.cfg.depth_layers else self.depth_norm(state))
            if d>0 and d%2==0 and self.cfg.codec == 'laser':
                logits = logits.scatter(1, teacher[:, :d:2].clamp_max(self.cfg.laser_atoms-1), -1e4)
            result.append(logits.view(*contexts.shape[:2], -1))
            if d < self.field_count-1 and not self.cfg.depth_layers:
                state = self.depth_cell(self.fields[d](teacher[:, d]), state)
        return result

    def depth_next_logits(self, context, fields, caches):
        """One cached depth-transformer step, equivalent to teacher forcing."""
        d = len(fields)
        x = context[:, None] if not fields else sum(
            self.fields[i](value) for i, value in enumerate(fields))[:, None]
        x = x + self.depth_position[:, d:d+1]
        for block, cache in zip(self.depth_blocks, caches):
            x = block(x, cache=cache)
        return self.heads[d](self.depth_norm(x[:, 0]))

    def forward(self, batch, guide_weight=0.0, return_logits=False):
        codes, lengths = batch['codes'], batch['lengths']
        memory, text_mask = self.text_memory(batch['phones'], batch['speakers'])
        context, attention = self.temporal(codes, memory, text_mask, batch['speakers'], alignment=guide_weight > 0)
        teacher = F.pad(codes, (0, 0, 0, 1))
        logits = self.depth_logits(context, teacher)
        positions = torch.arange(context.shape[1], device=codes.device)[None]
        valid = positions < lengths[:, None]
        nlls, losses, accuracy = [], [], []
        for d, prediction in enumerate(logits):
            target = teacher[..., d].clone().masked_fill(~valid, -100)
            if d == 0:
                target.scatter_(1, lengths[:, None], self.EOS)
            per_token = F.cross_entropy(prediction.flatten(0, 1).float(), target.flatten(), reduction='none', ignore_index=-100).view_as(target)
            mask = target.ne(-100)
            nlls.append(per_token.sum() / mask.sum().clamp_min(1))
            weights = torch.where(target == self.EOS, 5.0, 1.0) if d == 0 else torch.ones_like(per_token)
            losses.append((per_token * weights).sum() / (mask * weights).sum().clamp_min(1))
            accuracy.append(((prediction.argmax(-1) == target) & mask).sum() / mask.sum().clamp_min(1))
        guided = context.new_zeros(())
        if attention is not None:
            t = positions.float() / lengths[:, None].clamp_min(1)
            s = torch.arange(memory.shape[1], device=codes.device)[None].float() / batch['text_lengths'][:, None].clamp_min(1)
            penalty = 1 - torch.exp(-(t[:, :, None] - s[:, None, :]).square() / (2 * .2**2))
            guided = ((attention * penalty).sum(-1) * valid).sum() / valid.sum().clamp_min(1)
        result = {'loss': torch.stack(losses).mean() + guide_weight * guided,
                  'nll': torch.stack(nlls).mean(),
                  'token_accuracy': torch.stack(accuracy).mean(), 'guided_attention': guided}
        result.update({f'field_{d}_nll': value for d, value in enumerate(nlls)})
        if self.cfg.codec == 'laser':
            result.update(atom_nll=torch.stack(nlls[0::2]).mean(), coefficient_nll=torch.stack(nlls[1::2]).mean())
        if return_logits:
            result['logits'] = logits
        return result

    @torch.inference_mode()
    def generate(self, phones, speaker, max_frames=1500, min_frames=30, temperature=.8, top_k=50,
                 max_seconds=None,min_seconds=None):
        if phones.shape[0] != 1:
            raise ValueError('Generation currently supports one utterance at a time')
        if self.cfg.hard_rate_cap_bps:
            from src.audio_hard6k_bitstream import frame_budget,samples_for_frames
            if max_seconds is not None:max_frames=frame_budget(round(max_seconds*48000),self.cfg.codec)
            if min_seconds is not None:min_frames=frame_budget(round(min_seconds*48000),self.cfg.codec)
        memory, mask = self.text_memory(phones, speaker)
        cache = [{} for _ in self.blocks]
        cross_kv = [block.cross_attention.kv(memory) for block in self.blocks]
        frames, stopped = [], False
        x = self.bos
        for position in range(max_frames):
            y = x + position_encoding(1, self.cfg.width, phones.device, position).to(x.dtype)
            y = y + self.speakers(speaker)[:, None]
            for block, state, static in zip(self.blocks, cache, cross_kv):
                y, _ = block(y, memory, mask, cache=state, static_kv=static)
            state = self.out_norm(y[:, 0])
            fields = []
            depth_caches = [{} for _ in range(self.cfg.depth_layers)]
            depth_fields = []
            for d, head in enumerate(self.heads):
                scores = (self.depth_next_logits(state, depth_fields, depth_caches)
                          if self.cfg.depth_layers else head(self.depth_norm(state))).float()
                if d == 0 and position < min_frames:
                    scores[:, self.EOS] = -1e4
                if d>0 and d%2==0 and self.cfg.codec == 'laser':
                    scores[:, fields[0:d:2]] = -1e4
                if temperature <= 0:
                    value = int(scores.argmax(-1))
                else:
                    scores = scores / temperature
                    if top_k:
                        threshold = scores.topk(min(top_k, scores.shape[-1]), dim=-1).values[:, -1:]
                        scores = scores.masked_fill(scores < threshold, -1e4)
                    value = int(torch.multinomial(scores.softmax(-1), 1))
                if d == 0 and value == self.EOS:
                    stopped = True; break
                fields.append(value)
                depth_fields.append(torch.tensor([value], device=phones.device))
                if d < self.field_count-1 and not self.cfg.depth_layers:
                    state = self.depth_cell(self.fields[d](torch.tensor([value], device=phones.device)), state)
            if stopped:
                break
            frame = torch.tensor(fields, device=phones.device)[None, None]
            frames.append(frame[0, 0])
            x = self.frame_embedding(frame)
        tokens = torch.stack(frames) if frames else torch.empty(0, self.field_count, dtype=torch.long, device=phones.device)
        seconds=(samples_for_frames(len(frames),self.cfg.codec)/48000 if frames else 0.) if self.cfg.hard_rate_cap_bps else len(frames)/150
        return tokens, {'eos_reached': stopped, 'frames': len(frames), 'seconds': seconds}
