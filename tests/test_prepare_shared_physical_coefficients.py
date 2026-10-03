"""End-to-end cache preparation, publication boundary and legacy-codec rejection."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/tools/prepare_shared_physical_coefficients.py'
spec = importlib.util.spec_from_file_location('prepare_shared_grid_cli', SCRIPT)
prepare_cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_cli)


def test_preparation_preserves_bank_variants_rejects_legacy_and_verifies_durable(tmp_path):
    canonical = tmp_path / 'canonical.pt'
    bank = tmp_path / 'bank.pt'
    canonical_alias = tmp_path / 'canonical-alias.pt'
    common = dict(coeff_scales=[2., 1.], coeff_scale=2., coeff_max=3.,
        coeff_vocab_size=2048, num_atoms=8, clip_coefficients=False,
        coefficient_storage='fp32', stage1_checkpoint_sha256='a' * 64)
    atoms = torch.tensor([0, 1, 2, 3], dtype=torch.int16).reshape(2, 1, 1, 2)
    coefficients = torch.tensor([-.5, .25, 1., -1.5]).reshape_as(atoms)
    labels = torch.tensor([4, 5])
    torch.save(dict(atoms=atoms, coeffs=coefficients, labels=labels,
                    meta=dict(common, format='laser_compound_pairs_v1')), canonical)
    canonical_alias.symlink_to(canonical)
    bank_atoms = torch.stack([atoms, atoms.flip(-1)], dim=-2)
    bank_coefficients = torch.stack([coefficients, coefficients.flip(-1)], dim=-2)
    torch.save(dict(atoms=bank_atoms, coeffs=bank_coefficients,
        bank_log_normalizers=torch.full_like(bank_coefficients, 99.), labels=labels,
        meta=dict(common, format='laser_stochastic_compound_bank_v1',
                  coefficient_log_partition_temperature=.0625)), bank)
    audit = tmp_path / 'audit.json'
    audit.write_text(json.dumps(dict(passed=True, recommended_grid=dict(upper=6.),
        sources={str(p): dict(sha256=prepare_cli.sha256(p), bytes=p.stat().st_size)
                 for p in (canonical_alias, bank)})))
    ranges = tmp_path / 'ranges.json'
    ranges.write_text(json.dumps(dict(bound=6.)))
    checkpoint = tmp_path / 'legacy.pt'
    torch.save(dict(config=dict(common), global_step=10540), checkpoint)
    arguments = SimpleNamespace(canonical=canonical, bank=bank, source_audit=audit,
        range_audit=ranges, output=tmp_path/'local', durable=tmp_path/'durable',
        cpu_threads=2, recovery_helper=None, tokenizer=None, legacy_checkpoint=checkpoint)
    prepare_cli.prepare(arguments)
    converted = torch.load(arguments.output/'bank-shared-physical.pt', weights_only=False)
    assert torch.equal(converted['atoms'], bank_atoms)
    assert torch.equal(converted['labels'], labels)
    assert torch.equal(converted['coeffs'], bank_coefficients * torch.tensor([2., 1.]))
    assert 'bank_log_normalizers' not in converted
    assert converted['meta']['source_coefficient_log_partition_temperature'] == .0625
    report = json.loads((arguments.output/'conversion-report.json').read_text())
    assert report['legacy_checkpoint_rejection']['passed']
    assert report['grid']['scales'] == [1., 1.]
    for relative, record in json.loads((arguments.output/'prepared-manifest.json').read_text()).items():
        assert prepare_cli.sha256(arguments.durable/relative) == record['sha256']
    assert json.loads((arguments.durable/'durable-copy.json').read_text())['all_sha256_verified']
    # Originals and a completed prepared cache can never be overwritten by rerun.
    assert 'bank_log_normalizers' in torch.load(bank, weights_only=False)
    with pytest.raises(FileExistsError):
        prepare_cli.prepare(arguments)
