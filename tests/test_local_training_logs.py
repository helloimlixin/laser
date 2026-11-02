import json
import shutil

import torch

from src.training import k4_checkpoint_io as checkpoint_io
from src.training.rq_imagenet_runtime import local_training_logs,write_json


def test_live_io_migration_preserves_checkpoint_bytes_and_local_cache(tmp_path,monkeypatch):
    base=tmp_path/'nvme';out=tmp_path/'persistent/run';folder=out/'train'
    monkeypatch.setenv('LASER_CHECKPOINT_UPLOAD_CACHE_DIR',str(base/'cache'))
    monkeypatch.setenv('LASER_CHECKPOINT_STAGING_DIR',str(base/'staging'))
    monkeypatch.setenv('LASER_CHECKPOINT_IMMUTABLE_FILES','1')
    value={'weights':torch.arange(8.),'global_step':626}
    checkpoint_io.atomic_torch_save(value,folder/'checkpoints/last.pt')
    write_json(folder/'progress-rank0.json',{'step':626})
    layout=local_training_logs(base,folder)
    assert folder.is_symlink()
    assert (out/'checkpoints/last.pt').is_file()
    payload=(out/'checkpoints/last.pt').resolve()
    local=checkpoint_io._checkpoint_upload_source(payload)
    assert local!=payload and local.is_relative_to(base/'cache')
    torch.testing.assert_close(torch.load(local,weights_only=False)['weights'],value['weights'])
    write_json(folder/'progress-rank0.json',{'step':627})
    assert json.loads((folder/'progress-rank0.json').read_text())['step']==627
    assert json.loads((out/'persistent-train/progress-rank0.json').read_text())['step']==626
    assert local_training_logs(base,folder)==layout
    # Model recovery survives losing the transient diagnostic view.
    shutil.rmtree(base/'live-training')
    assert (out/'checkpoints/last.pt').is_file()
    local_training_logs(base,folder)
    assert (folder/'checkpoints/last.pt').is_file()
    torch.testing.assert_close(torch.load(payload,weights_only=False)['weights'],value['weights'])
