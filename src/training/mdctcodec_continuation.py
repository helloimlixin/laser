"""Validation media and graceful stopping for the longer matched codec runs."""
from pathlib import Path
import signal

import lightning as pl
import soundfile as sf
import torch
import wandb

from src.audio_research_media import select_validation_examples, render_audio_comparison
from src.mdctcodec_matched import reconstruct_serialized


class GracefulBudget(pl.Callback):
    def __init__(self, output):
        self.output = Path(output)
        self.requested = False

    def on_train_start(self, trainer, model):
        def stop(*_):
            self.requested = True
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        if self.requested or (self.output.parent / 'STOP').exists():
            model.continuation_stopped = True
            trainer.should_stop = True


class AudioContinuationMedia(pl.Callback):
    def __init__(self, output, manifest, arm):
        self.output, self.arm = Path(output), arm
        self.paths = select_validation_examples(manifest)

    @torch.inference_mode()
    def on_validation_end(self, trainer, model):
        epoch = trainer.current_epoch + 1
        if trainer.sanity_checking or epoch % 5:
            return
        step = int(model._manual_train_step)
        target = self.output / 'audio_media' / f'epoch-{epoch:03d}-step-{step:07d}'
        target.mkdir(parents=True, exist_ok=True)
        media = {'audio_media/completed_epochs': epoch, 'audio_media/generator_updates': step}
        with torch.random.fork_rng(devices=[model.device.index or 0]):
            for index, path in enumerate(self.paths):
                reference, sr = sf.read(path, dtype='float32')
                assert sr == 48000
                x = torch.from_numpy(reference)[None, None].to(model.device)
                with torch.autocast('cuda', enabled=False):
                    decoded, payload = reconstruct_serialized(model, x)
                audio = decoded[0, 0].float().cpu().numpy()
                stem = Path(path).stem
                wav = target / f'{stem}_{self.arm}.wav'
                sf.write(wav, audio, sr, subtype='FLOAT')
                wav.with_suffix('.bin').write_bytes(payload)
                reference_path = target / f'{stem}_reference.wav'
                sf.write(reference_path, reference, sr, subtype='FLOAT')
                figures, _ = render_audio_comparison({'reference': reference, self.arm: audio},
                    target / stem, f'{stem} | {self.arm.upper()} | {step:,} updates | 6 kbps')
                media[f'audio/{index}/reference'] = wandb.Audio(str(reference_path))
                media[f'audio/{index}/reconstruction'] = wandb.Audio(str(wav))
                for kind, image in figures.items():
                    media[f'audio/{index}/{kind}'] = wandb.Image(image)
        trainer.logger.log_metrics(media, step=trainer.global_step)
        run = trainer.logger.experiment
        artifact = wandb.Artifact(f'mdctcodec-continuation-audio-{run.id}', type='audio-evaluation',
            metadata={'completed_epochs': epoch, 'generator_updates': step,
                      'split': 'fixed_validation', 'arm': self.arm})
        artifact.add_dir(str(target))
        run.log_artifact(artifact, aliases=['latest', f'epoch-{epoch:03d}']).wait()
