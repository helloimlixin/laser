"""Repartition the verified epoch-four checkpoint, preserving both Adam states."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile

import torch

HERE = Path(__file__).resolve().parent
BASE = Path('/tmp/laser-s1-best-b200')
FROZEN = BASE / 'frozen/runtime'
RUN = 'imnet-s1-laser-best2934-continue10-1b200-20261104'
DURABLE = HERE.parents[1] / 'outputs' / RUN
SOURCE_RUN = 'helloimlixin-rutgers/laser/imnet-s1-laser-s4-k2-j4096-680-scratch10-2h100-20261102'
SOURCE_HASH = '8d113f210de88a2f3a4920f5c76496200bdc9bd7fbaad67ae4c5f3df69a9eeb6'


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + '\n')


def patch_runtime():
    original = BASE / 'original-train_matched_stage1.py'
    if not original.exists():
        shutil.copyfile(FROZEN / 'train_matched_stage1.py', original)
    text = original.read_text()
    text = text.replace('import dist as var_dist',
                        'import dist as var_dist\nfrom continuation import learning_rate')
    old = "for group in self.optimizer.param_groups:group['lr']=cfg['lr']*ratio"
    new = """for group in self.optimizer.param_groups:group['lr']=learning_rate(cfg,self.progress['step'])
                for group in self.d_optimizer.param_groups:group['lr']=learning_rate(cfg,self.progress['step'])"""
    assert old in text
    text = text.replace(old, new)
    text = text.replace("if self.progress['step']<=10 or self.progress['step']%10==0:",
                        "if self.smoke or self.progress['step']<=10 or self.progress['step']%10==0:")
    old = '                if flag.item():return\n'
    new = """                if flag.item():return
                if not self.smoke and (self.progress['step']==40536 or (self.progress['step']-40036)%2500==0):
                    self.save(publish=True);self.evaluate()
"""
    assert old in text
    text = text.replace(old, new)
    (FROZEN / 'train_matched_stage1.py').write_text(text)
    shutil.copyfile(HERE / 'continuation.py', FROZEN / 'continuation.py')


def prepare(microbatch, smoke=False):
    torch.set_num_threads(4)
    original = json.loads((BASE / 'checkpoint-artifact/config.json').read_text())
    checkpoint = BASE / 'checkpoint-artifact/checkpoint.pt'
    assert sha(checkpoint) == SOURCE_HASH
    state = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    assert state['world_size'] == 4 and state['progress'] == dict(epoch=4, batch=0, step=40036)
    old_contract = {k:v for k,v in original.items() if k not in {
        'output','wandb','workers','checkpoint_every','preview_every','source_bundle'}}
    assert hashlib.sha256(json.dumps(old_contract, sort_keys=True).encode()).hexdigest() == state['contract_sha256']
    output = BASE / (f'benchmark-mb{microbatch}' if smoke else 'run-output')
    output.mkdir(parents=True, exist_ok=True)
    cfg = dict(original)
    cfg.pop('migration', None)
    cfg.update(output=str(output), microbatch=microbatch, accumulation=128//microbatch,
               workers=8, eval_batch=64, checkpoint_every=1000, preview_every=500,
               train_archive=str(BASE/'data/train.tar'), val_archive=str(BASE/'data/val.tar'),
               training_archive_receipt=str(BASE/'data/train.verified.json'),
               smoke_cache=str(BASE/'data/smoke-cache.pt'),
               source_bundle=str(BASE/'continuation-source.tar.gz'),
               wandb=dict(id=RUN, name='ImageNet stage1 joint4096 | best rFID2.934 | cosine5e-5 | B200'))
    for key, name in [('train_index','train-index.npz'),('val_index','validation-index.npz'),
                      ('val_labels','val-labels.npy'),('fid_reference','real-feature-moments.npz')]:
        cfg[key] = str(BASE / 'source-v2/data' / name)
    cfg['continuation'] = dict(
        durable=str(DURABLE), source_run=SOURCE_RUN, source_artifact=SOURCE_RUN+'-tokenizer:v47',
        source_checkpoint_sha256=SOURCE_HASH, source_rfid=2.9342634986184635,
        source_epoch=4, source_step=40036, target_epoch=10, effective_batch=128,
        evaluation_images=50000, evaluate_first_at=40536, evaluate_every_updates=2500,
        topology_bitwise_replay=False, model_and_optimizer_states_preserved=True,
        learning_rate=dict(kind='cosine', maximum=5e-5, minimum=1e-5,
                           start_step=40036, end_step=100090, applies_to='generator_and_discriminator'))
    cfg['runtime_sha256'] = {name:sha(FROZEN/name) for name in original['runtime_sha256']}
    cfg['runtime_sha256']['continuation.py'] = sha(FROZEN/'continuation.py')
    contract = {k:v for k,v in cfg.items() if k not in {
        'output','wandb','workers','checkpoint_every','preview_every','source_bundle'}}
    contract_sha = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    migrated = dict(state, world_size=1, contract_sha256=contract_sha, rng=[state['rng'][0]],
                    topology_migration=cfg['continuation'])
    destination = output/'tokenizer-last.pt'
    torch.save(migrated, destination)
    roundtrip = torch.load(destination, map_location='cpu', weights_only=False, mmap=True)
    sys.path.insert(0, str(FROZEN))
    from migrate_stage1_world_size import assert_equal, global_indices
    for key in ['model','optimizer','discriminator','discriminator_optimizer']:
        assert_equal(state[key], roundtrip[key])
    assert_equal(state['rng'][0], roundtrip['rng'][0])
    before = global_indices(1281167,4,32,1,0,4,0)
    after = global_indices(1281167,1,microbatch,128//microbatch,0,4,0)
    assert before == after
    ages = sorted({int(v['step']) for v in roundtrip['optimizer']['state'].values() if 'step' in v})
    d_ages = sorted({int(v['step']) for v in roundtrip['discriminator_optimizer']['state'].values() if 'step' in v})
    assert ages == [40036] and d_ages == [30036]
    receipt = dict(passed=True, source_checkpoint_sha256=SOURCE_HASH,
        checkpoint_sha256=sha(destination), progress=state['progress'], full_state=True,
        contract_sha256=contract_sha, world_size=1, microbatch=microbatch,
        accumulation=128//microbatch, global_batch=128,
        optimizer_age=ages, discriminator_optimizer_age=d_ages,
        model_and_both_optimizers_bit_exact=True, original_rank_zero_rng_preserved=True,
        next_three_global_batches_identical=True, topology_is_not_bitwise_trajectory_replay=True)
    write(output/'config.json', cfg)
    write(output/'checkpoint.json', receipt)
    write(output/'migration-receipt.json', receipt)
    write(output/'initialization.json', dict(
        initialization='full_state_resume_from_best_rfid', source_artifact=SOURCE_RUN+'-tokenizer:v47',
        source_checkpoint_sha256=SOURCE_HASH, source_rfid=2.9342634986184635,
        optimizer_age=ages, discriminator_optimizer_age=d_ages, source_step=40036))
    if not smoke:
        DURABLE.mkdir(parents=True, exist_ok=True)
        for name in ['config.json','migration-receipt.json','initialization.json']:
            shutil.copyfile(output/name, DURABLE/name)
        write(output/'best-reconstruction.json',dict(
            epoch=4, step=40036, rfid=2.9342634986184635, count=50000,
            checkpoint=dict(artifact=SOURCE_RUN+'-tokenizer:v47',checkpoint_sha256=SOURCE_HASH)))
        shutil.copyfile(destination, output/'tokenizer-best-rfid.pt')
        shutil.copyfile(destination, DURABLE/'tokenizer-best-rfid.pt')
        with tarfile.open(BASE/'continuation-source.tar.gz','w:gz') as archive:
            for path in FROZEN.rglob('*'):
                if path.is_file() and '__pycache__' not in path.parts:
                    archive.add(path,arcname='runtime/'+str(path.relative_to(FROZEN)))
        shutil.copyfile(BASE/'continuation-source.tar.gz', DURABLE/'continuation-source.tar.gz')
    print(json.dumps(receipt),flush=True)
    return cfg


if __name__ == '__main__':
    patch_runtime()
    prepare(int(sys.argv[1]), smoke='--smoke' in sys.argv)
