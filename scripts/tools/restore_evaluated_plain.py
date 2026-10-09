"""Restore externally measured best metadata without changing training state."""
import json
import os
from pathlib import Path

_restored = None


def restore_plain_best(payload, path, output, record):
    global _restored
    if (os.environ.get('LASER_BRANCH') != 'baseline'
            or os.environ.get('LASER_PHASE') != 'train'
            or os.environ.get('LASER_WANDB_RESUME') != 'must'
            or not isinstance(payload, dict) or payload.get('global_step') != 3798
            or not isinstance(path, (str, Path))):
        return payload
    metadata_path = Path(output) / 'production-best-metadata.json'
    if not metadata_path.is_file():
        return payload
    metadata = json.loads(metadata_path.read_text())
    accepted = {Path(metadata['checkpoint']).resolve()}
    if metadata.get('continuation_local_serialization'):
        accepted.add(Path(metadata['continuation_local_serialization']).resolve())
    if Path(path).resolve() not in accepted:
        return payload
    assert payload['scheduler']['last_epoch'] == 3798
    assert len(payload['optimizer']['state']) == 798
    assert len(payload['rng_state_by_rank']) == payload['checkpoint_world_size'] == 8
    assert Path(metadata['best_fid_checkpoint']).is_file()
    assert Path(metadata['best_inception_checkpoint']).is_file()
    result = dict(payload, fid=metadata['fid'], inception_score=metadata['inception_score'],
        inception_score_std=metadata['inception_score_std'],
        best_fid=[(metadata['fid'], metadata['best_fid_checkpoint'])],
        best_inception=[(metadata['inception_score'], metadata['best_inception_checkpoint'])])
    protected = ('state_dict','optimizer','scheduler','rng_state_by_rank','config')
    assert all(result[key] is payload[key] for key in protected)
    assert all(result[key] == payload[key] for key in ('global_step','epoch','batch_idx'))
    _restored = metadata
    record('evaluated-best-restore-rank'+os.environ['RANK']+'.json',dict(
        passed=True,global_step=3798,primary_fid_seed=metadata['fid_seed'],
        fid=metadata['fid'],inception_score=metadata['inception_score'],
        protected_training_state_objects_unchanged=True))
    return result


def log_restored_plain_metrics(wb):
    if _restored is None:
        return
    wb.log({'train/global_step':3798,'train/epoch':_restored['progress_epoch'],
        'val/fid':_restored['fid'],'val/inception_score':_restored['inception_score'],
        'val/inception_score_std':_restored['inception_score_std']})
    wb.summary.update({'investigation/primary_fid_seed':_restored['fid_seed'],
        'investigation/primary_evaluation':_restored['evaluation_json'],
        'investigation/continuation_run':'imagenet-rfid421-plain-matched-3798-20261007',
        'investigation/evaluated_best_restored':True})
