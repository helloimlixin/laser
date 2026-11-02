"""Prepare the recovered source, stage recipes, and immutable recovery receipts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

from omegaconf import OmegaConf


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recovered', required=True)
    parser.add_argument('--base', required=True)
    parser.add_argument('--local', required=True)
    args = parser.parse_args()
    recovered, base, local = [Path(p).resolve() for p in (args.recovered, args.base, args.local)]
    repo = Path(__file__).resolve().parents[2]
    base.mkdir(parents=True, exist_ok=True)
    source = recovered / 'runtime'
    # Preserve the recovered critic, objective, sampler, logging, and checkpoint
    # implementations. Add only even-bin support and the new stage-2 prior.
    q_path = source / 'src/models/multiscale_laser_var.py'
    text = q_path.read_text().replace(
        'coefficient_bins < 3 or coefficient_bins % 2 != 1 or coefficient_max <= 0',
        'coefficient_bins < 3 or coefficient_max <= 0').replace(
        'Use an odd coefficient vocabulary and positive range',
        'Use at least three coefficient bins and a positive range')
    q_path.write_text(text)
    shutil.copyfile(repo / 'src/models/compound_multiscale_var.py', source / 'src/models/compound_multiscale_var.py')
    shutil.copyfile(repo / 'scripts/tools/run_var_critic_stage.py', source / 'run_var_critic_stage.py')
    driver = source / 'src/training/var_laser.py'
    text = driver.read_text()
    if 'if evaluator.get(\'official_adm\', True):' not in text:
        start = text.index('                evaluator = self.cfg.evaluation\n', text.index('def generate('))
        end = text.index('\n        dist.barrier()\n\n    def train_prior', start)
        block = text[start:end]
        lines = block.splitlines()
        text = (text[:start] + lines[0] + '\n' +
                '                if evaluator.get(\'official_adm\', True):\n' +
                '\n'.join('    ' + line if line else line for line in lines[1:]) + text[end:])
        driver.write_text(text)
    bundle = base / 'source.tar.gz'
    with tarfile.open(bundle, 'w:gz') as archive:
        for path in sorted(source.rglob('*')):
            relative = path.relative_to(source)
            if path.is_file() and not any(p in ('.git', '__pycache__', '.pytest_cache') for p in relative.parts):
                archive.add(path, arcname=str(relative), recursive=False)
    manifest = {str(p.relative_to(source)): digest(p) for p in source.rglob('*.py')
                if '.git' not in p.parts and '__pycache__' not in p.parts}
    (base / 'source-manifest.json').write_text(json.dumps(dict(
        original_source_sha256=digest(recovered / 'source/source.tar.gz'),
        training_source_sha256=digest(bundle), files=manifest), indent=2) + '\n')
    for stage in ('stage1', 'stage2'):
        for arm in ('laser', 'vq'):
            durable, out = base / stage / arm, local / stage / arm
            durable.mkdir(parents=True, exist_ok=True)
            out.mkdir(parents=True, exist_ok=True)
            cfg = OmegaConf.load(recovered / arm / 'resolved-config.yaml')
            cfg.output_dir = str(out)
            cfg.stage = stage
            cfg.pipeline = False
            cfg.resume = True
            cfg.tokenizer.epochs = 10
            cfg.tokenizer.batch_size = 32
            cfg.tokenizer.accumulation = 2
            cfg.data.root = '/tmp/laser-imagenet-var/data'
            cfg.data.manifests = '/tmp/laser-imagenet-var/manifests'
            cfg.evaluation.reconstruction_on_resume = False
            cfg.logging.source_bundle = str(bundle)
            cfg.logging.source_alias = 'resume-stage2-20261026'
            cfg.logging.source_artifact_name = 'imagenet256-var-source-resume-stage2-20261026'
            cfg.logging.source_reason = ('Resume original critic-repair tokenizers through epoch 10; '
                'joint gate before class-conditional stage 2; LASER uses atom/offset-coefficient '
                'compound tokens with 512 bins and the original VAR-d16 spatial transformer.')
            if stage == 'stage1':
                receipt = dict(artifact=f'helloimlixin-rutgers/laser/imagenet256-var-{arm}-tokenizer-relaunch10-20261025:v4',
                               progress=dict(epoch=7, batch=0, step=70063),
                               checkpoint_sha256=digest(recovered / arm / 'checkpoint.pt'),
                               all_rank_rng_preserved=True, global_batch=128, world_size=2)
                (durable / 'recovery.json').write_text(json.dumps(receipt, indent=2) + '\n')
                for target in (out / 'tokenizer-last.pt', durable / 'tokenizer-last.pt'):
                    if not target.exists():
                        shutil.copyfile(recovered / arm / 'checkpoint.pt', target)
                cfg.wandb.notes += (' Resume 2026-10-26 from complete epoch-7 backup (step 70063), '
                    'preserving both optimizers, critic, and both rank RNG states; '
                    'two RTX PRO 6000 Blackwell GPUs, unchanged effective batch 128. '
                    'Stage 2 waits for BOTH epoch-10 completion markers.')
            else:
                cfg.prior.batch_size = 32
                cfg.prior.accumulation = 6
                cfg.prior.epochs = 200
                cfg.data.workers = 6
                cfg.prior.token_format = 'atom_coeff_offset' if arm == 'laser' else 'vq'
                cfg.evaluation.official_adm = False
                cfg.logging.checkpoint_every_steps = 100
                if arm == 'laser':
                    cfg.model.coefficient_bins = 512
                cfg.wandb.id = (f'imagenet256-var-{arm}-' + ('compound512-' if arm == 'laser' else '') +
                                'd16-stage2-20261026')
                cfg.wandb.name = f'ImageNet256 - {arm.upper()} VAR-d16 stage 2' + (' compound512' if arm == 'laser' else '')
                cfg.wandb.group = 'imagenet256-var-epoch10-classcond-stage2-20261026'
                cfg.wandb.notes = ('Class-conditional ImageNet256 generation from frozen epoch-10 critic-repair '
                    'tokenizer; original FoundationVision VAR-d16 spatial body (16 layers, 1024 width, '
                    '16 heads, 680 spatial positions), class conditioning and CFG dropout; '
                    'AdamW 3e-4, betas .9/.95, weight decay .05, 4 warmup epochs, lin0 floor .1, '
                    '200 epochs, global batch 768 on four GPUs. PyTorch FID uses all 50k validation '
                    'images; exact 50k uint8 sample archives are exported every 20 epochs; '
                    'ADM metrics are not enabled. ' +
                    ('Sparse tokens [atom1,4096+bin1,atom2,4096+bin2], vocabulary 4608; '
                     '512 signed asinh coefficient bins with frozen ranges, requantized from OMP; '
                     'two-layer 256-wide causal within-site token transformer with shared head '
                     'and type/no-duplicate masks.' if arm == 'laser' else 'Original VQ VAR categorical prediction.'))
            OmegaConf.save(cfg, durable / 'launch-config.yaml', resolve=True)
    (base / 'supervisor-status.json').write_text(json.dumps(dict(
        phase='prepared', source=str(source), local=str(local),
        stage1_gpus=dict(laser='0,1', vq='2,3'),
        stage2_gpus=dict(laser='0,1,2,3', vq='4,5,6,7'),
        stage2_joint_epoch10_gate=True), indent=2) + '\n')
    print(json.dumps(dict(base=str(base), source=str(source), local=str(local)), indent=2))


if __name__ == '__main__':
    main()
