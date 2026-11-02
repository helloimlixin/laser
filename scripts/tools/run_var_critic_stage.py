"""Run a recovered VAR stage from its source snapshot with durable backups."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--durable', required=True)
    parser.add_argument('--gate')
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.source).resolve()))
    from omegaconf import OmegaConf
    import torch
    import torch.distributed as dist
    from src.training import var_laser
    from src.original_rq_training import atomic_json, file_sha256

    cfg = OmegaConf.load(args.config)
    durable = Path(args.durable)
    durable.mkdir(parents=True, exist_ok=True)
    out = Path(cfg.output_dir)
    stage2 = cfg.stage == 'stage2'
    checkpoint_name = 'prior-last.pt' if stage2 else 'tokenizer-last.pt'
    if not (out / 'tokenizer-last.pt').is_file():
        raise FileNotFoundError('Refusing to initialize a fresh tokenizer during recovery')
    if stage2:
        if not args.gate:
            raise ValueError('Stage 2 requires the joint epoch-10 gate')
        gate = json.loads(Path(args.gate).read_text())
        if not gate.get('ready') or set(gate.get('runs', {})) != {'laser', 'vq'}:
            raise ValueError('Both epoch-10 tokenizers must finish before stage 2')
        for arm, receipt in gate['runs'].items():
            progress = receipt['progress']
            if progress['epoch'] != 10 or progress['batch'] != 0:
                raise ValueError(f'{arm} has not completed epoch 10')
        source_checkpoint = out / 'tokenizer-last.pt'
        expected = gate['runs'][cfg.model.bottleneck]['checkpoint_sha256']
        if file_sha256(source_checkpoint) != expected:
            raise ValueError('Stage-2 source tokenizer differs from the epoch-10 gate')
        original_builder = var_laser.build_scratch_prior
        if cfg.model.bottleneck == 'laser':
            if cfg.model.coefficient_bins != 512:
                raise ValueError('Sparse stage 2 requires 512 bins')
            from src.models.compound_multiscale_var import CompoundLaserVAR
            def build(tokenizer, kind, seed, **options):
                if kind != 'laser':
                    return original_builder(tokenizer, kind, seed, **options)
                tokenizer.quantize.coefficient_range_decay = None
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed)
                    return CompoundLaserVAR(tokenizer, **options)
            var_laser.build_scratch_prior = build
    original_save = var_laser.save_checkpoint
    original_init = var_laser.Experiment.__init__
    experiment = None
    published = set()
    codec = dict(token_format='atom_coeff_offset' if cfg.model.bottleneck == 'laser' and stage2 else 'tokenizer',
                 atoms=int(cfg.model.atoms), coefficient_bins=int(cfg.model.coefficient_bins),
                 atom_id_interval=[0, int(cfg.model.atoms)],
                 coefficient_offset=int(cfg.model.atoms))

    def initialize(self, config):
        nonlocal experiment
        original_init(self, config)
        experiment = self
        if self.rank == 0:
            self.run.config.update(OmegaConf.to_container(config, resolve=True), allow_val_change=True)
            recovery = json.loads((durable / 'recovery.json').read_text())
            self.run.summary.update({
                'resume/recovered_epoch': recovery['progress']['epoch'],
                'resume/recovered_step': recovery['progress']['step'],
                'resume/recovered_artifact': recovery['artifact'],
                'resume/target_tokenizer_epochs': 10,
                'resume/world_size': self.world,
                'resume/full_optimizer_and_critic_state': True,
                'resume/all_rank_rng_preserved': not stage2,
                'resume/stage2_joint_gate': stage2,
            })

    def save(path, model, optimizer, progress, extras=None):
        payload_extras = dict(extras or {})
        if stage2:
            payload_extras['token_contract'] = codec
        original_save(path, model, optimizer, progress, payload_extras)
        backup_error = [None]
        if dist.get_rank() == 0:
            try:
                temporary = durable / (path.name + '.mirror.tmp')
                shutil.copyfile(path, temporary)
                if temporary.stat().st_size != path.stat().st_size:
                    raise OSError('Incomplete durable checkpoint mirror')
                # The workspace's remote filesystem rejects fsync with EIO even
                # after a successful close. Verify bytes before atomic replacement.
                checkpoint_sha = file_sha256(path)
                if file_sha256(temporary) != checkpoint_sha:
                    raise OSError('Durable checkpoint checksum differs from local state')
                temporary.replace(durable / path.name)
                atomic_json(durable / 'checkpoint.json', dict(
                    progress=dict(progress), bytes=path.stat().st_size,
                    world_size=dist.get_world_size(), token_contract=codec,
                    checkpoint_sha256=checkpoint_sha, saved_unix=time.time()))
                epoch_end = progress['batch'] == 0
                # Back up mid-epoch as well as epoch boundaries, limiting rollback
                # when a machine disappears before the next epoch completes.
                identity = (progress['epoch'], progress['step'])
                if (epoch_end or progress['step'] % 1000 == 0 or experiment.stop) and identity not in published:
                    import wandb
                    arm = cfg.model.bottleneck
                    collection = (f'imagenet256-var-{arm}-stage2-20261026' if stage2 else
                                  f'imagenet256-var-{arm}-tokenizer-relaunch10-20261025')
                    artifact = wandb.Artifact(collection, type='model', metadata=dict(
                        progress=dict(progress), source_run=cfg.wandb.id,
                        stage=cfg.stage, world_size=dist.get_world_size(), full_state=True,
                        token_contract=codec))
                    artifact.add_file(str(path), name='checkpoint.pt', policy='immutable')
                    artifact.add_file(args.config, name='resolved-config.yaml', policy='immutable')
                    artifact.add_file(str(durable / 'recovery.json'), name='recovery.json', policy='immutable')
                    aliases = ['latest', f'step-{progress["step"]:09d}']
                    if epoch_end:
                        aliases.append(f'epoch-{progress["epoch"]:03d}')
                    experiment.run.log_artifact(artifact, aliases=aliases).wait()
                    published.add(identity)
                    atomic_json(durable / 'remote-backup.json', dict(
                        progress=dict(progress), artifact=artifact.qualified_name,
                        uploaded_unix=time.time()))
            except Exception as error:
                backup_error[0] = f"{type(error).__name__}: {error}"
        dist.broadcast_object_list(backup_error, src=0)
        if backup_error[0]:
            raise RuntimeError("Checkpoint publication failed: " + backup_error[0])

    var_laser.Experiment.__init__ = initialize
    var_laser.save_checkpoint = save
    if stage2 and (out / checkpoint_name).exists():
        checkpoint = torch.load(out / checkpoint_name, map_location='cpu', weights_only=False, mmap=True)
        if checkpoint.get('token_contract') != codec:
            raise ValueError('Prior checkpoint uses a different token formulation')
        del checkpoint
    var_laser.run(cfg)
    if int(os.environ.get('RANK', 0)) == 0:
        checkpoint = torch.load(out / checkpoint_name, map_location='cpu', weights_only=False, mmap=True)
        progress = checkpoint['progress']
        target = cfg.prior.epochs if stage2 else 10
        if progress['epoch'] == target and progress['batch'] == 0:
            atomic_json(durable / 'complete.json', dict(
                progress=progress, checkpoint=str(durable / checkpoint_name),
                checkpoint_sha256=file_sha256(out / checkpoint_name),
                completed_unix=time.time(), stage=cfg.stage))


if __name__ == '__main__':
    main()
