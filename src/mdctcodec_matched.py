"""Shared data order and serialized validation for matched 6 kbps training."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path

import lightning as pl
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from src.models.laser import LASER
from src.mdctcodec_bitstream import pack_frames, unpack_frames
from archive.scripts.benchmark_mdctcodec_trained_rvq import payload_roundtrip
from archive.scripts.benchmark_mdctcodec_vctk import align_mdct, measure


def tensor_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str((value.shape, value.dtype)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class PairedAudioDataset(Dataset):
    def __init__(self, paths, seed=1234, crop_samples=7960):
        self.paths, self.seed, self.crop_samples = list(paths), seed, crop_samples

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, key):
        epoch, index = key if isinstance(key, tuple) else (0, key)
        path = self.paths[index]
        info = sf.info(path)
        if info.samplerate != 48000 or info.channels != 1:
            raise ValueError(f'Expected mono 48 kHz: {path}')
        offset = 0
        if self.crop_samples:
            # Independent of model RNG, worker scheduling and prefetch timing.
            seed = int.from_bytes(hashlib.sha256(
                f'{self.seed}:{epoch}:{Path(path).name}'.encode()).digest()[:8], 'little')
            rng = np.random.default_rng(seed)
            offset = int(rng.integers(0, max(1, info.frames-self.crop_samples+1)))
            audio, _ = sf.read(path, start=offset, frames=self.crop_samples, dtype='float32')
            audio = np.pad(audio, (0, max(0, self.crop_samples-len(audio))))
        else:
            audio, _ = sf.read(path, dtype='float32')
        if not np.isfinite(audio).all():
            raise ValueError(f'Nonfinite audio: {path}')
        metadata = {'path': path, 'speaker_id': Path(path).parent.name,
                    'crop_mode': int(bool(self.crop_samples)), 'source_num_samples':info.frames,
                    'crop_offset': offset, 'audio_format': 'waveform',
                    'spec_min': 0.0, 'spec_max': 1.0, 'spec_shape': torch.tensor([0,len(audio)])}
        return torch.from_numpy(audio)[None], 0, metadata


class PairedSampler(Sampler):
    def __init__(self, dataset, epoch_getter, seed=1234):
        self.dataset, self.epoch_getter, self.seed = dataset, epoch_getter, seed

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        epoch = int(self.epoch_getter())
        order = torch.randperm(len(self.dataset), generator=torch.Generator().manual_seed(self.seed+epoch))
        return iter((epoch, i) for i in order.tolist())


class MatchedData(pl.LightningDataModule):
    def __init__(self, manifest, workers=8, batch_size=48, validation_limit=0):
        super().__init__()
        self.manifest, self.workers, self.batch_size = manifest, workers, batch_size
        self.config = {'dataset':'vctk','sample_rate':48000,'audio_representation':'waveform',
                       'mean':(0.0,),'std':(1.0,)}
        self.train_dataset = PairedAudioDataset(manifest['train'], manifest['seed'])
        paths = manifest['validation'][:validation_limit] if validation_limit else manifest['validation']
        self.val_dataset = PairedAudioDataset(paths, crop_samples=0)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, drop_last=True,
                          sampler=PairedSampler(self.train_dataset, lambda:self.trainer.current_epoch,
                                                self.manifest['seed']),
                          num_workers=self.workers, pin_memory=True,
                          persistent_workers=self.workers>0,
                          generator=torch.Generator().manual_seed(99123))

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, num_workers=min(4,self.workers),
                          pin_memory=True, generator=torch.Generator().manual_seed(99124))


@torch.no_grad()
def reconstruct_serialized(model, x):
    """Return waveform decoded strictly from five-byte-per-frame payloads."""
    x, length = align_mdct(x)
    with torch.autocast(device_type=x.device.type, enabled=False):
        _, _, codes = model.encode(x.float())
        if model.bottleneck_type == 'mdctcodec_rvq':
            ids = codes.support.squeeze(1).transpose(1,2)
            payload, parsed = payload_roundtrip(ids.cpu().numpy())
            latent = model.bottleneck.quantizer.from_codes(torch.from_numpy(parsed).to(x.device))[0]
            y = model.decoder(latent)
        else:
            bound = float(model.bottleneck.coefficient_quantization_max)
            integers = codes.values.float().div(bound/63).round().long()
            payload = pack_frames(codes.support.cpu().numpy(), integers.cpu().numpy())
            atoms, values = unpack_frames(payload)
            atoms = torch.from_numpy(atoms).to(x.device).reshape_as(codes.support)
            values = torch.from_numpy(values).to(x.device).reshape_as(codes.values).float()*(bound/63)
            y = model.decode_from_atoms_and_coeffs(atoms, values)
    y = y[..., :length]
    if y.shape != (1,1,length) or not torch.isfinite(y).all():
        raise ValueError('Invalid serialized reconstruction')
    return y.clamp(-1,1), payload


class MatchedModel(LASER):
    """Keep the shared training step; validate full recordings in FP32."""
    output_path = None
    metric_workers = 8

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
        if self.bottleneck_type == 'dictionary':
            bound=checkpoint.get('hyper_parameters',{}).get('coefficient_quantization_max')
            if bound is not None:
                self.bottleneck.coefficient_quantization_max=float(bound)
                self.hparams['coefficient_quantization_max']=float(bound)

    def on_validation_epoch_start(self):
        self._metric_pool = ThreadPoolExecutor(max_workers=self.metric_workers)
        self._metric_jobs = []
        self._validation_payload_bits = self._validation_samples = 0

    def validation_step(self, batch, batch_idx):
        x, _, metadata = batch
        y, payload = reconstruct_serialized(self, x)
        reference, decoded = x[0,0].float().cpu().numpy(), y[0,0].float().cpu().numpy()
        self._metric_jobs.append(self._metric_pool.submit(measure,
            (Path(metadata['path'][0]).name, reference, decoded)))
        self._validation_payload_bits += len(payload)*8
        self._validation_samples += len(reference)

    def on_validation_epoch_end(self):
        try:
            rows = [job.result() for job in self._metric_jobs]
        finally:
            self._metric_pool.shutdown(wait=True)
        score = float(np.mean([r['visqol_audio48k'] for r in rows]))
        rate = self._validation_payload_bits/(self._validation_samples/48000)/1000
        self.log('val/audio_visqol_audio48k', score, on_epoch=True, batch_size=len(rows))
        self.log('val/stoi', float(np.mean([r['stoi'] for r in rows])), on_epoch=True, batch_size=len(rows))
        self.log('val/payload_kbps', rate, on_epoch=True, batch_size=len(rows))
        step = int(self._manual_train_step)
        if self.output_path:
            path = Path(self.output_path)/'validation'; path.mkdir(exist_ok=True)
            (path/f'step-{step:07d}.json').write_text(json.dumps(
                {'generator_updates':step, 'epoch':self.current_epoch, 'visqol':score,
                 'payload_kbps':rate, 'rows':rows}, indent=2))
        print('MATCHED_VALIDATION',json.dumps({'step':step,'epoch':self.current_epoch,
              'n':len(rows),'visqol':score,'payload_kbps':rate}),flush=True)


class PairedAudit(pl.Callback):
    def __init__(self, output, budget):
        self.output, self.budget = Path(output), budget
        self.records=[]
        self.batches=0
        self.digest=hashlib.sha256()

    def state_dict(self):
        return {'records':self.records,'batches':self.batches}

    def load_state_dict(self, state):
        self.records=list(state['records'])
        self.batches=int(state['batches'])
        self.digest=hashlib.sha256()
        for record in self.records:self.digest.update(record.encode())

    def on_train_epoch_start(self, trainer, model):
        self.digest = hashlib.sha256()
        self.batches = 0
        self.records=[]

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        metadata = batch[2]
        for path, offset in zip(metadata['path'], metadata['crop_offset'].tolist()):
            record=f'{Path(path).name}:{offset}\n'
            self.records.append(record)
            self.digest.update(record.encode())
        self.batches += 1
        updates = int(model._manual_train_step)
        if updates == 1:
            required=['train/audio_mdct_loss','train/audio_mel_loss','train/audio_feature_matching_loss']
            for key in required:
                value=trainer.callback_metrics.get(key)
                if value is None or not torch.isfinite(value) or float(value)<=0:
                    raise RuntimeError(f'Missing/nonpositive audio objective in preflight: {key}={value}')
        if updates > self.budget or trainer.global_step != updates*2:
            raise RuntimeError('Expected exactly one generator and discriminator update per batch')
        if updates % 100 == 0 or updates < 3:
            metrics = {'generator_updates': updates, 'completed_epochs': trainer.current_epoch}
            model.logger.log_metrics(metrics, step=trainer.global_step)
            print('MATCHED_PROGRESS',json.dumps({**metrics,'batch_idx':batch_idx,
                'loss':float(outputs['loss'] if isinstance(outputs,dict) else outputs)}),flush=True)

    def on_train_epoch_end(self, trainer, model):
        if self.batches==0:return  # Legacy preflight checkpoint resumes after validation.
        row = {'epoch':trainer.current_epoch, 'generator_updates':int(model._manual_train_step),
               'batches':self.batches, 'data_order_sha256':self.digest.hexdigest()}
        path=self.output/'data_order.jsonl'
        if path.exists():
            previous=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            same=[r for r in previous if r['epoch']==row['epoch']]
            if same:
                if same[-1]!=row:raise RuntimeError('Resume changed the audited epoch stream')
                return
        with path.open('a') as stream:
            stream.write(json.dumps(row)+'\n')

    def on_exception(self, trainer, model, exception):
        (self.output/'failure.json').write_text(json.dumps({'type':type(exception).__name__,
            'message':str(exception), 'generator_updates':int(model._manual_train_step)},indent=2))
        if not isinstance(exception,KeyboardInterrupt):
            try:
                trainer.save_checkpoint(str(self.output/'checkpoints/failure.ckpt'))
                model.logger.experiment.summary['status']='failed'
            except Exception as error:
                (self.output/'failure_checkpoint_error.txt').write_text(str(error))


class TrainingCoefficientRange(pl.Callback):
    """Track a GLOBAL model range using the last training batch's raw OMP values.

    Both D and G use the same bound during a step. The observer runs afterwards,
    so validation, inference and test cannot calibrate their own input range.
    Bound changes are saved in model hyperparameters, and observer/guard state in
    callback state. No per-frame or per-recording side information is introduced.
    """
    def __init__(self, output, *, margin=1.1, release=0.001, window=100,
                 guard_start=1000, maximum_mean_clipping=0.05, **_):
        self.output=Path(output)
        self.margin=float(margin)
        self.release=float(release)
        self.window=int(window)
        self.guard_start=int(guard_start)
        self.maximum_mean_clipping=float(maximum_mean_clipping)
        if self.margin<1 or not 0<self.release<=1 or self.window<1:
            raise ValueError('Invalid coefficient range policy')
        self.last_bound=None
        self.clipping=[]
        self.last_step=0

    def state_dict(self):
        return {'last_bound':self.last_bound,'clipping':self.clipping,'last_step':self.last_step,
                'policy':self.policy()}

    def policy(self):
        return {'margin':self.margin,'release':self.release,'window':self.window,
                'guard_start':self.guard_start,'maximum_mean_clipping':self.maximum_mean_clipping}

    def load_state_dict(self,state):
        if state['policy']!=self.policy():raise ValueError('Resume changed the range policy')
        self.last_bound=state['last_bound']
        self.clipping=list(state['clipping'])
        self.last_step=int(state['last_step'])

    def observe(self, bound, p999, saturation, step):
        if not all(math.isfinite(x) for x in [bound,p999,saturation]) or bound<=0 or p999<0:
            raise FloatingPointError('Invalid training coefficient range observation')
        if not 0<=saturation<=1:raise ValueError('Invalid clipping fraction')
        target=max(1e-6,self.margin*p999)
        # Grow immediately to follow early encoder scale drift. Shrink slowly to
        # avoid range oscillations between quiet and loud training minibatches.
        updated=target if target>bound else bound+self.release*(target-bound)
        self.last_bound=float(np.float32(updated))
        if not math.isfinite(self.last_bound) or self.last_bound<=0:
            raise FloatingPointError('Training coefficient range cannot be represented in FP32')
        self.clipping=(self.clipping+[float(saturation)])[-self.window:]
        self.last_step=int(step)
        return self.last_bound

    def on_train_batch_end(self,trainer,model,outputs,batch,batch_idx):
        if model.bottleneck_type!='dictionary' or not model.training:
            raise RuntimeError('Range observer must run only on LASER training batches')
        bottleneck=model.bottleneck
        used=float(bottleneck.coefficient_quantization_max)
        step=int(model._manual_train_step)
        p999=float(bottleneck._last_coefficient_abs_p999)
        saturation=float(bottleneck._last_coefficient_saturation_fraction)
        updated=self.observe(used,p999,saturation,step)
        bottleneck.coefficient_quantization_max=updated
        model.hparams['coefficient_quantization_max']=updated
        mean=float(np.mean(self.clipping))
        row={'generator_updates':step,'bound_used':used,'bound_next':updated,
             'training_abs_p999':p999,'clipping_fraction':saturation,'clipping_window_mean':mean}
        if step%20==0 or step<3:
            model.logger.log_metrics({f'train/range_{k}':v for k,v in row.items()
                                      if k!='generator_updates'},step=trainer.global_step)
        if step%100==0 or step<3:
            with (self.output/'coefficient_range.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            print('COEFFICIENT_RANGE',json.dumps(row),flush=True)
        if step>=self.guard_start and len(self.clipping)==self.window and mean>self.maximum_mean_clipping:
            raise RuntimeError(f'Range guard: mean clipping {mean:.3%} exceeds '
                               f'{self.maximum_mean_clipping:.1%}; stop the paired experiment')
