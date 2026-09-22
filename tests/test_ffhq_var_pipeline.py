import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

from PIL import Image
import pytest
import torch

from scripts.tools.prepare_ffhq_var_data import archive_index, generate_rows
from src.training.cli import load_config


def source_fixture(tmp_path):
    metadata = {}
    archive = tmp_path/'images.zip'
    with zipfile.ZipFile(archive, 'w') as output:
        for index, split in enumerate(('training', 'validation')):
            buffer = io.BytesIO()
            Image.new('RGB', (1024, 1024), (index * 200, 25, 50)).save(buffer, format='PNG')
            data = buffer.getvalue()
            output.writestr(f'images1024x1024/{index:05d}.png', data)
            metadata[str(index)] = dict(category=split, image=dict(
                file_size=len(data), file_md5=hashlib.md5(data).hexdigest()))
    path = tmp_path/'metadata.json'
    path.write_text(json.dumps(metadata))
    return archive, path, metadata


def test_official_ffhq_split_resize_and_checksum_validation(tmp_path):
    archive, path, metadata = source_fixture(tmp_path)
    assert len(archive_index(archive, metadata, dict(training=1, validation=1))) == 2
    rows = {}
    for split in ('training', 'validation'):
        rows[split] = list(generate_rows([0], str(archive), str(path), split, 1, str(tmp_path)))
        assert len(rows[split]) == 1 and rows[split][0]['label'] == 0
        image = Image.open(io.BytesIO(rows[split][0]['image']['bytes']))
        assert image.size == (256, 256) and image.mode == 'RGB'
    assert rows['training'][0]['image_id'] == 0
    assert rows['validation'][0]['image_id'] == 1
    metadata['0']['image']['file_md5'] = 'incorrect'
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='MD5 mismatch'):
        list(generate_rows([0], str(archive), str(path), 'training', 1, str(tmp_path)))
    with pytest.raises(ValueError, match='split counts'):
        archive_index(archive, metadata, dict(training=2))


def test_ffhq_recipes_are_fully_scratch_and_unconditional():
    root = Path(__file__).resolve().parents[1]/'configs/experiments'
    tokenizer = load_config(root/'ffhq256-var341-tokenizer.yaml')
    prior = load_config(root/'ffhq256-var341-compound.yaml')
    for cfg in (tokenizer, prior):
        assert cfg.model.initialization == 'scratch' and cfg.model.pretrained_vae is None
        assert not cfg.model.get('init_tokenizer_checkpoint')
        assert cfg.data.dataset == 'ffhq'
        assert list(cfg.model.patch_nums) == [1, 2, 4, 8, 16]
        assert sum(p*p for p in cfg.model.patch_nums) == 341
        assert cfg.model.residual_scale_positions is None
        assert cfg.evaluation.preview_samples == 64 and cfg.evaluation.grid_columns == 8
        assert cfg.evaluation.fid_reference_images == 10000
        assert cfg.prior.cfg == 0.0
    assert tokenizer.execution.upload_checkpoints and prior.compound.upload_checkpoints


def test_single_class_prior_matches_teacher_forced_sampler(monkeypatch):
    import dist
    from omegaconf import OmegaConf
    from src.models.multiscale_laser_var import MultiScaleLaser, VAR
    from src.models.compound_var import compound_decompose
    from src.training.compound_var import CompoundExperiment
    monkeypatch.setattr(dist, 'get_device', lambda: 'cpu')
    q = MultiScaleLaser(channels=4, atoms=8, sparsity=2, patch_nums=(1, 2, 3),
                        coefficient_bins=33, coefficient_max=3.).eval()
    experiment = CompoundExperiment.__new__(CompoundExperiment)
    experiment.cfg = OmegaConf.create(dict(seed=0, model=dict(depth=2),
        compound=dict(local_width=16, dropout=0., atom_loss_weight=1.5, coefficient_top_p=1.)))
    experiment.vae = SimpleNamespace(quantize=q)
    experiment.num_classes, experiment.device = 1, torch.device('cpu')
    model = experiment.build_prior().eval()
    assert model.num_classes == 1 and model.cond_drop_rate == 0.
    codes = compound_decompose(q, torch.randn(2, 4, 3, 3))
    labels = torch.zeros(2, dtype=torch.long)
    with torch.no_grad():
        features = VAR.forward(model, labels, codes['inputs'])
        atoms, coefficients = model.token_logits(features, codes['atoms'], codes['coefficients'])
        sampled = model.sample(labels, cfg=0, teacher_codes=codes, return_details=True)
    torch.testing.assert_close(atoms, sampled['atom_logits'], atol=3e-6, rtol=2e-5)
    torch.testing.assert_close(coefficients, sampled['coefficient_logits'], atol=3e-6, rtol=2e-5)


def test_identical_checkpoint_states_deduplicate_across_filenames(tmp_path, monkeypatch):
    from src.training import var_laser
    monkeypatch.setattr(torch.cuda, 'get_rng_state', lambda: torch.tensor([1, 2], dtype=torch.uint8))
    monkeypatch.setattr(var_laser.dist, 'get_world_size', lambda: 1)
    monkeypatch.setattr(var_laser.dist, 'get_rank', lambda: 0)
    monkeypatch.setattr(var_laser.dist, 'all_gather_object', lambda gathered, value: gathered.__setitem__(0, value))
    monkeypatch.setattr(var_laser.dist, 'barrier', lambda: None)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    paths = [tmp_path/'last.pt', tmp_path/'best.pt']
    for path in paths:
        var_laser.save_checkpoint(path, model, optimizer, dict(step=1))
    assert paths[0].read_bytes() == paths[1].read_bytes()
    loaded = torch.load(paths[1], map_location='cpu', weights_only=False)
    assert loaded['progress']['step'] == 1 and loaded['optimizer']['state']
    for name, value in model.state_dict().items():
        torch.testing.assert_close(loaded['model'][name], value)
