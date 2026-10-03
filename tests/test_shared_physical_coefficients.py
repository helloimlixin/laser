from copy import deepcopy
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.shared_physical_coefficients import (
    SHARED_PHYSICAL_REPRESENTATION, build_shared_physical_grid, convert_cache_payload,
    legacy_coefficient_centers, legacy_physical_coefficients, nearest_coefficient_ids,
    require_compatible_checkpoint_codec, require_shared_physical_cache, guard_shared_physical_checkpoint,
)


def metadata(*, bank=False, scales=(2., 4., 8., 16.)):
    return dict(format='laser_stochastic_compound_bank_v1' if bank else 'laser_compound_pairs_v1',
        coeff_vocab_size=2048, coeff_max=3., coeff_scales=list(scales), coeff_scale=6.4,
        coefficient_storage='fp32', clip_coefficients=False, causal_prefix_coeffs=False,
        num_atoms=11, shape=[2, 3, 4], items=2, stage1_checkpoint_sha256='a' * 64,
        bank_identity='b' * 64, variants_per_site=5 if bank else 1)


def payload(*, bank=False):
    shape = (2, 2, 3, 5, 4) if bank else (2, 2, 3, 4)
    atoms = (torch.arange(torch.tensor(shape).prod()).reshape(shape) % 11).to(torch.int16)
    values = torch.linspace(-3., 3., atoms.numel()).reshape(shape)
    result = dict(atoms=atoms, coeffs=values, labels=torch.zeros(2, dtype=torch.int16), meta=metadata(bank=bank))
    if bank:
        result['bank_log_normalizers'] = values.square()
        result['meta'].update(soft_atom_targets=True, coefficient_log_partition_temperature=.0625,
                              atom_targets='old normalized-grid conditionals')
    return result


@pytest.mark.parametrize('shape', [(4,), (3, 4), (2, 3, 4), (2, 2, 3, 4), (2, 2, 3, 5, 4)])
def test_arbitrary_leading_dimensions_convert_once_in_fp32_without_mutation(shape):
    torch.manual_seed(90)
    normalized = torch.randn(shape)
    before = normalized.clone()
    meta = metadata()
    physical = legacy_physical_coefficients(normalized, meta)
    assert torch.equal(physical, normalized * torch.tensor(meta['coeff_scales']))
    assert torch.equal(normalized, before) and physical.dtype == torch.float32
    assert physical.data_ptr() != normalized.data_ptr()


@pytest.mark.parametrize('bank', [False, True])
def test_continuous_signed_sparse_vectors_atom_order_and_metadata_are_preserved(bank):
    source = payload(bank=bank)
    before = deepcopy(source)
    converted = convert_cache_payload(source, source_sha256='c' * 64)
    dictionary = torch.randn(11, 7)
    expected_pairs = dictionary[source['atoms'].long()] * (
        source['coeffs'] * torch.tensor(source['meta']['coeff_scales']))[..., None]
    actual_pairs = dictionary[converted['atoms'].long()] * converted['coeffs'][..., None]
    assert torch.equal(actual_pairs, expected_pairs)
    assert torch.equal(actual_pairs.sum(-2), expected_pairs.sum(-2))
    assert (converted['coeffs'] < 0).any() and (converted['coeffs'] > 0).any()
    for key in ('atoms', 'coeffs', 'labels'):
        assert torch.equal(source[key], before[key])
    assert source['meta'] == before['meta']
    assert torch.equal(converted['atoms'], source['atoms'])
    assert torch.equal(converted['labels'], source['labels'])
    assert converted['atoms'].data_ptr() != source['atoms'].data_ptr()
    assert converted['meta']['format'] == source['meta']['format']
    assert converted['meta']['variants_per_site'] == source['meta']['variants_per_site']
    assert converted['meta']['coeff_scales'] == [1.] * 4
    assert converted['meta']['coefficient_representation'] == SHARED_PHYSICAL_REPRESENTATION
    assert converted['meta']['bank_identity'] != source['meta']['bank_identity']
    require_shared_physical_cache(converted['meta'])


def test_stale_grid_tables_and_soft_target_flags_are_invalidated():
    source = payload(bank=True)
    converted = convert_cache_payload(source, source_sha256='d' * 64)
    assert 'bank_log_normalizers' not in converted
    assert converted['meta']['invalidated_coefficient_payload_keys'] == ['bank_log_normalizers']
    assert converted['meta']['soft_atom_targets'] is False
    assert 'coefficient_log_partition_temperature' not in converted['meta']
    assert converted['meta']['source_coefficient_log_partition_temperature'] == .0625
    assert converted['meta']['source_coefficient_metadata'] == source['meta']
    assert converted['meta']['requires_stage2_vocabulary_migration'] is True
    assert converted['meta']['legacy_stage2_checkpoint_compatible'] is False


def test_range_covers_every_old_grid_and_bank_roundoff_beyond_old_endpoint():
    scales = [7.662353992462158, 4.1580352783203125, 2.633323907852173, 1.6512117385864258]
    meta = metadata(scales=scales)
    normalized = torch.tensor([[3.000000476837158, -3., 1., 2.]])
    physical = legacy_physical_coefficients(normalized, meta)
    centers = build_shared_physical_grid(physical, meta)
    assert float(centers[-1]) == 22.9870662689209
    assert float(centers[0]) == -22.9870662689209
    old_grid = legacy_coefficient_centers(meta)[None] * torch.tensor(scales)[:, None]
    assert old_grid.abs().max() <= centers[-1]
    assert physical.abs().max() <= centers[-1]
    assert centers.numel() == 2048 and not (centers == 0).any()
    assert torch.equal(centers, -centers.flip(0))


def test_common_range_gives_same_codec_identity_across_greedy_and_bank_sources():
    deterministic = convert_cache_payload(payload(), source_sha256='1' * 64, additional_bound=60.)
    bank = convert_cache_payload(payload(bank=True), source_sha256='2' * 64, additional_bound=60.)
    assert deterministic['meta']['coefficient_codec_identity'] == bank['meta']['coefficient_codec_identity']
    assert deterministic['meta']['coeff_bin_centers'] == bank['meta']['coeff_bin_centers']
    assert deterministic['meta']['bank_identity'] != bank['meta']['bank_identity']


def test_additional_bound_rounds_outward_and_provided_grid_cannot_clip_tail():
    source = payload()
    bound = float(torch.tensor(48.).double() + (torch.nextafter(torch.tensor(48.), torch.tensor(float('inf'))) - 48.).double() / 4)
    converted = convert_cache_payload(source, source_sha256='e' * 64, additional_bound=bound)
    assert converted['meta']['coeff_max'] >= bound
    assert converted['meta']['coeff_max'] > 48.
    too_small = torch.linspace(-47., 47., 2048)
    with pytest.raises(ValueError, match='cover'):
        convert_cache_payload(source, source_sha256='e' * 64, centers=too_small)


def test_shared_ids_have_identical_meaning_at_every_depth_and_nearest_matches_exhaustive():
    meta = metadata(scales=(1., 2., 4., 8.))
    values = torch.tensor([[-3., -3., -3., -3.], [1.5, 1.5, 1.5, 1.5], [0., 0., 0., 0.]])
    normalized = values / torch.tensor(meta['coeff_scales'])
    physical = legacy_physical_coefficients(normalized, meta)
    centers = build_shared_physical_grid(physical, meta)
    ids = nearest_coefficient_ids(physical, centers)
    assert torch.equal(ids, ids[:, :1].expand_as(ids))
    assert torch.equal(centers[ids], centers[ids[:, :1]].expand_as(values))
    torch.manual_seed(27)
    probe = (torch.rand(1024) * 2 - 1) * centers[-1]
    expected = (probe[:, None] - centers[None]).abs().argmin(-1)
    assert torch.equal(nearest_coefficient_ids(probe, centers), expected)
    midpoint = (centers[1023] + centers[1024]) / 2
    assert nearest_coefficient_ids(midpoint, centers).item() == 1023
    with pytest.raises(ValueError, match='clipping'):
        nearest_coefficient_ids(torch.tensor([float(centers[-1]) + 1]), centers)


def test_double_conversion_lossy_storage_and_unknown_units_are_rejected():
    source = payload()
    converted = convert_cache_payload(source, source_sha256='f' * 64)
    with pytest.raises(ValueError, match='already physical'):
        legacy_physical_coefficients(converted['coeffs'], converted['meta'])
    with pytest.raises(ValueError, match='already physical'):
        convert_cache_payload(converted, source_sha256='0' * 64)
    with pytest.raises(ValueError, match='FP32'):
        legacy_physical_coefficients(source['coeffs'].half(), source['meta'])
    for key, value in [('clip_coefficients', True), ('coefficient_storage', 'fp16'),
                       ('coefficient_representation', 'unknown'), ('coeff_scales', [1., 2.])]:
        meta = deepcopy(source['meta']); meta[key] = value
        with pytest.raises(ValueError):
            legacy_physical_coefficients(source['coeffs'], meta)


def test_checkpoint_compatibility_requires_new_explicit_shared_token_meaning():
    source = payload()
    converted = convert_cache_payload(source, source_sha256='a' * 64)
    with pytest.raises(ValueError, match='migration'):
        require_compatible_checkpoint_codec(source['meta'], converted['meta'])
    assert require_compatible_checkpoint_codec(converted['meta'], converted['meta'])
    corrupted = deepcopy(converted['meta'])
    corrupted['coeff_bin_centers'][100] += .001
    with pytest.raises(ValueError):
        require_compatible_checkpoint_codec(corrupted, converted['meta'])
    corrupted = deepcopy(converted['meta']); corrupted['num_atoms'] += 1
    with pytest.raises(ValueError, match='identity'):
        require_shared_physical_cache(corrupted)


def test_checkpoint_guard_blocks_both_grid_directions_without_affecting_legacy_pairs():
    source = payload()
    converted = convert_cache_payload(source, source_sha256='d' * 64)
    shared = converted['meta']
    assert guard_shared_physical_checkpoint(source['meta'], source['meta'])
    assert guard_shared_physical_checkpoint({}, None)
    assert guard_shared_physical_checkpoint(shared, shared)
    for config, meta in [(source['meta'], shared), (shared, source['meta']), (shared, None)]:
        with pytest.raises(ValueError):
            guard_shared_physical_checkpoint(config, meta)
    missing_flag = deepcopy(shared); del missing_flag['coefficient_representation']
    with pytest.raises(ValueError):
        guard_shared_physical_checkpoint(missing_flag, source['meta'])


@pytest.mark.parametrize('payload_name', ['raw_payload', 'init_payload'])
def test_actual_trainer_resume_and_initializer_guard_before_loading_any_weights(payload_name):
    # Execute the trainer's actual load, guard and state-load statements without
    # importing its unrelated legacy third-party training CLI dependencies.
    path = Path(__file__).resolve().parents[1] / 'src/training/rqtransformer.py'
    tree = ast.parse(path.read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    loaded = next(node for node in ast.walk(main) if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == payload_name for target in node.targets))
    guarded = next(node for node in ast.walk(main) if isinstance(node, ast.Expr)
                   and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                   and node.value.func.id == 'guard_shared_physical_checkpoint'
                   and any(isinstance(child, ast.Name) and child.id == payload_name for child in ast.walk(node)))
    state_load = next(node for node in ast.walk(main) if isinstance(node, ast.Expr)
                      and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                      and node.value.func.attr == 'load_state_dict'
                      and any(isinstance(child, ast.Name) and child.id == payload_name for child in ast.walk(node)))
    assert loaded.lineno < guarded.lineno < state_load.lineno
    code = compile(ast.Module(body=[loaded, guarded, state_load], type_ignores=[]), str(path), 'exec')
    shared = convert_cache_payload(payload(), source_sha256='e' * 64)['meta']
    calls = []
    for config, expected_loads in [(metadata(), 0), (shared, 1)]:
        environment = dict(torch=SimpleNamespace(load=lambda *a, **k: dict(config=config, state_dict={'test': 1})),
            resume_checkpoint=Path('/not-read'), args=SimpleNamespace(init_stage2_checkpoint=Path('/not-read')),
            cache_meta=shared, guard_shared_physical_checkpoint=guard_shared_physical_checkpoint,
            unwrapped_model=SimpleNamespace(load_state_dict=lambda state, strict: calls.append((state, strict))))
        if expected_loads:
            exec(code, environment)
        else:
            with pytest.raises(ValueError):
                exec(code, environment)
        assert len(calls) == expected_loads


def test_actual_trainer_saved_config_stamps_shared_meaning_for_valid_future_resume():
    path = Path(__file__).resolve().parents[1] / 'src/training/rqtransformer.py'
    tree = ast.parse(path.read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    assignment = next(node for node in ast.walk(main) if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == 'runtime_config' for target in node.targets))
    names = {'coefficient_representation', 'coefficient_codec_identity', 'coeff_scales', 'coeff_bin_centers'}
    items = [(key, value) for key, value in zip(assignment.value.keys, assignment.value.values)
             if isinstance(key, ast.Constant) and key.value in names]
    assert {key.value for key, _ in items} == names
    expression = ast.fix_missing_locations(ast.Expression(body=ast.Dict(
        keys=[key for key, _ in items], values=[value for _, value in items])))
    shared = convert_cache_payload(payload(), source_sha256='f' * 64)['meta']
    config = eval(compile(expression, str(path), 'eval'), dict(cache_meta=shared,
        aux=SimpleNamespace(coeff_scales=torch.ones(4)), cached_bin_centers=shared['coeff_bin_centers']))
    config['coeff_vocab_size'] = 2048  # Preserved by the existing **vars(args).
    assert guard_shared_physical_checkpoint(config, shared)


def test_grid_validation_rejects_fitted_or_mutated_centers_and_runtime_scale_mismatch():
    converted = convert_cache_payload(payload(), source_sha256='b' * 64)
    meta = converted['meta']
    centers = require_shared_physical_cache(meta)
    wrong = centers.clone(); wrong[111] += .001
    with pytest.raises(ValueError, match='uniform'):
        nearest_coefficient_ids(torch.zeros(4), wrong)
    wrong_meta = deepcopy(meta); wrong_meta['coeff_scales'][2] = 2.
    with pytest.raises(ValueError, match='one'):
        require_shared_physical_cache(wrong_meta)
    wrong = centers.clone(); wrong[222] += .001
    with pytest.raises(ValueError, match='runtime centers'):
        require_shared_physical_cache(meta, wrong)


def test_nonfinite_values_and_unhandled_payloads_are_rejected():
    source = payload()
    source['coeffs'][0, 0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        convert_cache_payload(source, source_sha256='c' * 64)
    source = payload(); source['unrecognized_token_table'] = torch.ones(1)
    with pytest.raises(ValueError, match='explicit conversion policy'):
        convert_cache_payload(source, source_sha256='c' * 64)
    with pytest.raises(ValueError, match='SHA256'):
        convert_cache_payload(payload(), source_sha256='missing')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA transfer check is optional')
def test_canonical_cpu_grid_retains_byte_exact_meaning_after_cuda_transfer():
    device = torch.device('cuda', torch.cuda.current_device())
    meta = metadata()
    values = torch.tensor([[1., -2., 3., -1.]])
    physical = legacy_physical_coefficients(values, meta)
    cpu = build_shared_physical_grid(physical, meta, additional_bound=60.1234567)
    gpu = build_shared_physical_grid(physical.to(device), meta, additional_bound=60.1234567)
    assert torch.equal(cpu, gpu.cpu())
    assert torch.equal(nearest_coefficient_ids(physical, cpu), nearest_coefficient_ids(physical.to(device), gpu).cpu())
