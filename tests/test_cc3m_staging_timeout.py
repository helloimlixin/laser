import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from src.data import cc3m_joint


def sample(tmp_path):
    source = tmp_path/'source'
    source.mkdir()
    raw = b'original pinned CC3M shard bytes'
    (source/'sample.tar').write_bytes(raw)
    spec = dict(name='sample.tar', bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    return source, tmp_path/'local', spec


def test_local_source_copy_is_verified_and_reused(tmp_path, monkeypatch):
    source, local, spec = sample(tmp_path)
    target = cc3m_joint.stage_shard(spec, source, local)
    assert target.read_bytes() == (source/spec['name']).read_bytes()
    assert json.loads(target.with_suffix('.verified.json').read_text())['sha256'] == spec['sha256']
    monkeypatch.setattr(cc3m_joint.subprocess, 'run', lambda *a, **k: pytest.fail('verified shard copied again'))
    assert cc3m_joint.stage_shard(spec, source, local) == target


def test_timed_out_mount_copy_retries_then_checks_original_hash(tmp_path, monkeypatch):
    source, local, spec = sample(tmp_path)
    run, calls = subprocess.run, []

    def interrupted(command, **kwargs):
        calls.append(command)
        assert kwargs['timeout'] == 120
        assert command[-2] == str(source/spec['name'])
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(command, 120)
        return run(command, **kwargs)

    monkeypatch.setattr(cc3m_joint.subprocess, 'run', interrupted)
    monkeypatch.setattr(cc3m_joint.time, 'sleep', lambda _: None)
    assert cc3m_joint.stage_shard(spec, source, local).read_bytes() == (source/spec['name']).read_bytes()
    assert len(calls) == 2


def test_corrupt_mount_copy_never_gets_a_verification_receipt(tmp_path):
    source, local, spec = sample(tmp_path)
    (source/spec['name']).write_bytes(b'incorrect bytes')
    with pytest.raises(ValueError, match='checksum mismatch'):
        cc3m_joint.stage_shard(spec, source, local)
    assert not (local/'sample.verified.json').exists()
