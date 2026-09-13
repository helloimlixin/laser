"""K4 audio: discrete coefficient training and real rate checks.

4096 shared atoms match the total vectors in RVQ's four 1024-entry codebooks.
The superseded 1024-atom checkpoints remain loadable with their saved sizes.
"""
import inspect
import json
from pathlib import Path
import time

import lightning as pl
import numpy as np
import torch

from src.audio_k4_entropy import SparseHuffman
from src.mdctcodec_matched import MatchedModel, measure, align_mdct
from src.models.dictionary_learner import DictionaryLearning


class K4Dictionary(DictionaryLearning):
    """Keep OMP and alternating dictionary learning; use eight signed levels + zero."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.sparsity_level != 4 or self.num_embeddings not in (1024,4096):
            raise ValueError('This experiment requires K=4 and 4096 learned atoms (1024 allowed for historical restore)')
        self.joint_vocabulary = 1 + self.num_embeddings*8
        self.register_buffer('coefficient_levels', torch.tensor([-.8, -.4, -.2, -.08, 0., .08, .2, .4, .8]))
        self._pending_magnitudes = None

    def coefficient_bins(self, values):
        levels = self.coefficient_levels
        return torch.bucketize(values.float().contiguous(), (levels[:-1]+levels[1:])/2)

    def _quantize_coefficients(self, values, *, record_stats=True):
        values = values.float()
        if not torch.isfinite(values).all(): raise ValueError('Nonfinite OMP coefficients')
        if record_stats:
            absolute = values.detach().abs().flatten()
            self._last_coefficient_quantization_fraction.fill_(1.)
            self._last_coefficient_saturation_fraction.zero_()  # nearest centers, no hard range clipping
            self._last_coefficient_abs_p99.copy_(torch.quantile(absolute, .99))
            self._last_coefficient_abs_p999.copy_(torch.quantile(absolute, .999))
            self._last_coefficient_abs_max.copy_(absolute.max())
            if self.training: self._pending_magnitudes = absolute
        return self.coefficient_levels[self.coefficient_bins(values)]

    def joint_tokens(self, support, values):
        bins = self.coefficient_bins(values)
        nonzero_bin = bins - (bins > 4).long()
        return torch.where(bins == 4, 0, 1 + support.long()*8 + nonzero_bin)

    def from_joint_tokens(self, tokens):
        if (tokens < 0).any() or (tokens >= self.joint_vocabulary).any(): raise ValueError('Invalid sparse token')
        packed = (tokens.long()-1).clamp_min(0)
        support, bins = packed // 8, packed % 8
        bins = bins + (bins >= 4).long()
        values = self.coefficient_levels[bins] * (tokens != 0)
        return support, values

    @torch.no_grad()
    def update_levels_after_batch(self):
        values = self._pending_magnitudes; self._pending_magnitudes = None
        if values is None: raise RuntimeError('Missing training-only coefficient observations')
        positive = torch.cat((values.new_zeros(1), self.coefficient_levels[5:]))
        ids = torch.bucketize(values, (positive[:-1]+positive[1:])/2)
        count = torch.bincount(ids, minlength=5)
        sums = torch.bincount(ids, weights=values, minlength=5)
        targets = torch.where(count[1:] > 0, sums[1:]/count[1:].clamp_min(1), positive[1:])
        # Training-only online scalar Lloyd update. Both optimizer passes use
        # identical levels; validation/inference never change these buffers.
        updated = positive[1:].lerp(targets, .1)
        self.coefficient_levels.copy_(torch.cat((-updated.flip(0), values.new_zeros(1), updated)))
        if not (self.coefficient_levels[1:] > self.coefficient_levels[:-1]).all():
            raise RuntimeError('Coefficient centers collapsed')


class K4AudioModel(MatchedModel):
    def __init__(self, **kwargs):
        kwargs.pop('k4_transport', None)
        if kwargs.get('num_embeddings') not in (1024,4096) or kwargs.get('sparsity_level') != 4:
            raise ValueError('Refusing a K2 or mismatched-dictionary audio run')
        if kwargs.get('bottleneck_type') != 'dictionary': raise ValueError('K4 LASER model expected')
        super().__init__(**kwargs)
        parameters = inspect.signature(DictionaryLearning.__init__).parameters
        bottleneck = K4Dictionary(**{k:v for k,v in kwargs.items() if k in parameters})
        missing, extra = bottleneck.load_state_dict(self.bottleneck.state_dict(), strict=False)
        assert missing == ['coefficient_levels'] and not extra
        self.bottleneck = bottleneck
        self.register_buffer('entropy_counts', torch.zeros(4, bottleneck.joint_vocabulary+1))
        self.hparams['k4_transport'] = f'{bottleneck.num_embeddings} learned atoms, K4, eight signed coefficient levels + zero; training-fitted temporal-repeat Huffman'
        self._huffman = None

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self._huffman = SparseHuffman(self.entropy_counts.detach().cpu().numpy())
        self._raw_payload_bits = 0

    def validation_step(self, batch, batch_idx):
        x, _, metadata = batch
        y, payload, tokens = reconstruct_k4(self, x, return_tokens=True)
        reference = x[0,0].float().cpu().numpy(); decoded = y[0,0].float().cpu().numpy()
        self._metric_jobs.append(self._metric_pool.submit(measure, (Path(metadata['path'][0]).name, reference, decoded)))
        self._validation_payload_bits += len(payload)*8
        self._validation_samples += len(reference)
        self._raw_payload_bits += tokens.shape[0]*4*(self.bottleneck.joint_vocabulary-1).bit_length()

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        seconds = self._validation_samples/48000
        self.log('val/raw_joint_payload_kbps', self._raw_payload_bits/seconds/1000)
        self.log('val/entropy_payload_kbps_including_header', self._validation_payload_bits/seconds/1000)
        self.log('val/meets_6kbps_mean_payload_target', float(self._validation_payload_bits/seconds <= 6000))
        self.log('val/learned_atoms', float(self.bottleneck.num_embeddings))
        self.log('val/sparsity_level', 4.)


@torch.inference_mode()
def reconstruct_k4(model, x, *, return_tokens=False):
    x, length = align_mdct(x)
    with torch.autocast(device_type=x.device.type, enabled=False):
        latent, _, codes = model.encode(x.float())
        if codes.support.shape[-1] != 4: raise RuntimeError('Wrong sparse depth')
        tokens = model.bottleneck.joint_tokens(codes.support, codes.values).reshape(-1,4)
        transport = model._huffman or SparseHuffman(model.entropy_counts.detach().cpu().numpy())
        payload = transport.encode(tokens.cpu().numpy())
        parsed = transport.decode(payload)
        if not np.array_equal(parsed, tokens.cpu().numpy()): raise RuntimeError('Bitstream mismatch')
        support, values = model.bottleneck.from_joint_tokens(torch.from_numpy(parsed).to(x.device).reshape_as(codes.support))
        y = model.decode_from_atoms_and_coeffs(support, values)[..., :length]
    if y.shape != (1,1,length) or not torch.isfinite(y).all(): raise ValueError('Invalid K4 waveform')
    result = (y.clamp(-1,1), payload)
    return (*result, parsed) if return_tokens else result


class K4TrainingStatistics(pl.Callback):
    def __init__(self, gpu_hour_ceiling=12., output=None, reference_order=None):
        self.elapsed = 0.
        self.started = None
        self.ceiling = float(gpu_hour_ceiling)*3600
        self.output = Path(output) if output else None
        self.reference_order = ({r['epoch']:r for r in
            (json.loads(line) for line in Path(reference_order).read_text().splitlines())}
            if reference_order else None)

    def state_dict(self):
        return {'elapsed_seconds':self.elapsed + (time.monotonic()-self.started if self.started is not None else 0.)}

    def load_state_dict(self,state):
        self.elapsed=float(state['elapsed_seconds']);self.started=None

    def on_fit_start(self,trainer,model):
        self.started=time.monotonic()

    def on_train_epoch_end(self,trainer,model):
        if self.reference_order is None or trainer.limit_train_batches != 1.0: return
        rows=[json.loads(line) for line in (self.output/'data_order.jsonl').read_text().splitlines()]
        current=next(r for r in reversed(rows) if r['epoch']==trainer.current_epoch)
        reference=self.reference_order[trainer.current_epoch]
        if getattr(model,'continuation_stopped',False) and current['batches'] < reference['batches']:
            # A user/budget stop intentionally ends a partial epoch. Preserve
            # the checkpoint without mislabeling this as a data-order failure.
            # Complete preceding epochs have already passed the exact audit.
            return
        if current != reference:
            raise RuntimeError('K4 LASER data/crop audit differs from the matched RVQ control')

    @torch.no_grad()
    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        if not model.training: raise RuntimeError('No validation-dependent calibration')
        q = model.bottleneck
        codes = q._last_sparse_codes_for_visualization
        expected_atoms=int(model.hparams['num_embeddings'])
        if codes.support.shape[-1] != 4 or q.dictionary.shape != (32,expected_atoms):
            raise RuntimeError('The requested K4 dictionary architecture changed')
        tokens = q.joint_tokens(codes.support, codes.values).squeeze(1)
        symbols = tokens.clone()
        symbols[:,1:] = torch.where(tokens[:,1:] == tokens[:,:-1], q.joint_vocabulary, tokens[:,1:])
        observed = torch.stack([torch.bincount(symbols[:,:,d].flatten(), minlength=q.joint_vocabulary+1) for d in range(4)]).float()
        model.entropy_counts.mul_(.999).add_(observed)
        q.update_levels_after_batch()
        model._huffman = None
        elapsed=self.state_dict()['elapsed_seconds']
        if elapsed >= self.ceiling:
            model.continuation_stopped=True
            trainer.should_stop=True
        if int(model._manual_train_step) % 20 == 0:
            model.logger.log_metrics({'train/learned_atoms':expected_atoms, 'train/sparsity_level':4,
                'train/assigned_gpu_hours':elapsed/3600,
                'train/coefficient_level_max':float(q.coefficient_levels[-1]),
                'train/nonzero_coefficients_per_frame':float((codes.values != 0).float().sum(-1).mean())}, step=trainer.global_step)
