import io
import json
from pathlib import Path
import tarfile

import pytest
import torch

from src.training.cli import load_config
from src.training.continuous_compound import pilot_images, run


def config(tmp_path, name, steps):
    root = Path(__file__).resolve().parents[1]
    return load_config(root / 'configs/experiments/continuous-compound-smoke.yaml', [
        f'output_dir={tmp_path}/{name}', f'runtime_base={tmp_path}/{name}-runtime',
        f'pilot.steps={steps}', 'pilot.device=cpu', 'pilot.cpu_threads=1',
        'arch.body.block.resid_pdrop=0.1', 'arch.head.block.resid_pdrop=0.1'])


def test_public_yaml_dispatches_the_continuous_backend_and_rejects_stage_one(tmp_path):
    cfg = config(tmp_path, 'unused', 1)
    assert cfg.backend == 'continuous_compound' and cfg.target_mode == 'original_omp_continuous'
    with pytest.raises(ValueError, match='requires stage2'):
        load_config(Path(__file__).resolve().parents[1] / 'configs/experiments/continuous-compound-smoke.yaml',
                    ['stage=stage1'])


def test_fixed_batch_pilot_resume_reproduces_uninterrupted_optimizer_with_dropout(tmp_path):
    uninterrupted = config(tmp_path, 'uninterrupted', 3)
    run(uninterrupted)
    interrupted = config(tmp_path, 'interrupted', 2)
    run(interrupted)
    resumed = config(tmp_path, 'resumed', 3)
    resumed.resume = str(tmp_path / 'interrupted-runtime/last.pt')
    run(resumed)
    a = torch.load(tmp_path / 'uninterrupted-runtime/last.pt', weights_only=False)
    b = torch.load(tmp_path / 'resumed-runtime/last.pt', weights_only=False)
    for name, value in a['state_dict'].items():
        torch.testing.assert_close(value, b['state_dict'][name], rtol=0, atol=0)
    for index, state in a['optimizer']['state'].items():
        for field, value in state.items():
            torch.testing.assert_close(value, b['optimizer']['state'][index][field], rtol=0, atol=0)
    report = json.loads((tmp_path / 'resumed/result.json').read_text())
    assert report['passed'] and report['synthetic'] and not report['generation_fid_measured']
    assert json.loads((tmp_path / 'resumed/status.json').read_text())['phase'] == 'completed'
    with pytest.raises(FileExistsError, match='Completed pilot'):
        run(resumed)


def test_real_pilot_rejects_wrong_tokenizer_before_creating_run_artifacts(tmp_path):
    cfg = config(tmp_path, 'wrong-source', 1)
    cfg.data.synthetic = False
    cfg.tokenizer.checkpoint = str(tmp_path / 'stage1.pt')
    cfg.data.train_archive = str(tmp_path / 'images.tar')
    (tmp_path / 'stage1.pt').write_bytes(b'wrong checkpoint')
    (tmp_path / 'images.tar').write_bytes(b'not read before identity verification')
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        run(cfg)
    assert not (tmp_path / 'wrong-source').exists()


def test_archive_pilot_reads_only_two_images_and_reuses_verified_class_offsets(tmp_path, monkeypatch):
    from PIL import Image
    inner_bytes = io.BytesIO()
    with tarfile.open(fileobj=inner_bytes, mode='w:') as inner:
        for name, color in [('b.JPEG', 'red'), ('a.JPEG', 'blue')]:
            image = io.BytesIO()
            Image.new('RGB', (8, 8), color).save(image, format='JPEG')
            data = image.getvalue()
            entry = tarfile.TarInfo(name); entry.size = len(data)
            inner.addfile(entry, io.BytesIO(data))
    archive = tmp_path / 'images.tar'
    with tarfile.open(archive, 'w:') as outer:
        for i in range(1000):
            entry = tarfile.TarInfo(f'n{i:08d}.tar'); entry.size = len(inner_bytes.getvalue())
            outer.addfile(entry, io.BytesIO(inner_bytes.getvalue()))
    original = tarfile.TarFile.getmembers

    def no_inner_enumeration(self):
        assert not isinstance(self.fileobj, tarfile.ExFileObject)
        return original(self)

    monkeypatch.setattr(tarfile.TarFile, 'getmembers', no_inner_enumeration)
    index = tmp_path / 'class-offsets.json'
    a = pilot_images(archive, 3, 256, index_path=index)
    assert a[0].shape == a[1].shape == (3, 3, 256, 256)
    assert a[2].tolist() == [0, 500, 999]
    assert all(row['fit'] == 'b.JPEG' and row['heldout'] == 'a.JPEG' for row in a[3])
    monkeypatch.setattr(tarfile.TarFile, 'getmembers', lambda _: pytest.fail('Class offsets were not reused'))
    b = pilot_images(archive, 3, 256, index_path=index)
    torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
    with archive.open('ab') as stream:
        stream.write(b'\0')
    with pytest.raises(ValueError, match='archive changed'):
        pilot_images(archive, 3, 256, index_path=index)
