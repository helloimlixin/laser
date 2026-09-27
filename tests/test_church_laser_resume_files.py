"""Selected checkpoints must survive node-local allocation changes."""
import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location('church_laser_resume_state',
    Path(__file__).resolve().parents[1] / 'scripts/tools/church_laser_resume_state.py')
resume = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resume)


def test_selected_files_move_from_resume_and_legacy_output(tmp_path):
    source = tmp_path / 'resume'
    legacy = tmp_path / 'train'
    destination = tmp_path / 'local'
    for folder in (source, legacy, destination):
        folder.mkdir()
    rows = [dict(path=f'fid-{i}.pt') for i in range(3)]
    (source / rows[0]['path']).write_bytes(b'resume file')
    (legacy / rows[1]['path']).write_bytes(b'legacy file')
    (destination / rows[2]['path']).write_bytes(b'already staged')
    resume.restore_ranked_checkpoints(rows, destination, source, legacy)
    assert [(destination / row['path']).read_bytes() for row in rows] == [
        b'resume file', b'legacy file', b'already staged']
    assert (source / rows[0]['path']).is_file()


def test_missing_selected_checkpoint_is_not_silently_dropped(tmp_path):
    with pytest.raises(FileNotFoundError, match='missing.pt'):
        resume.restore_ranked_checkpoints([dict(path='missing.pt')], tmp_path / 'local', tmp_path)


def test_selected_checkpoint_cannot_escape_destination(tmp_path):
    with pytest.raises(ValueError, match='plain filename'):
        resume.restore_ranked_checkpoints([dict(path='../last.pt')], tmp_path / 'local', tmp_path)
