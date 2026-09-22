from copy import deepcopy

import pytest

from src.training.compound_resume import prepare_resume, epoch_fraction, schedule_position


def checkpoint():
    return dict(world_size=2, training_config=dict(prior=dict(batch_size=64, accumulation=2, lr=.0003),
                                                 model=dict(depth=16), seed=0),
                progress=dict(epoch=11, batch=2, step=1200))


def test_layout_migration_replays_only_partial_batch_and_preserves_lr_position():
    saved = checkpoint()
    config = deepcopy(saved['training_config'])
    config['prior'].update(batch_size=160, accumulation=1)
    state, reference, receipt = prepare_resume(saved, config, 3, 28000, True)
    assert reference == 109
    assert state['batch'] == 0 and state['step'] == 1200
    assert receipt['replayed_images'] == 256
    assert (state['epoch'] + state['epoch_fraction']) * reference == pytest.approx(1200)
    fractions = [epoch_fraction(state['epoch_fraction'], 0, i, 58) for i in range(59)]
    assert fractions[0] == state['epoch_fraction'] and fractions[-1] == 1.
    assert all(b > a for a, b in zip(fractions, fractions[1:]))
    saved.update(world_size=3, training_config=config, schedule_reference_updates=reference,
                 progress={**state, 'batch': 20, 'epoch_fraction': fractions[20]})
    resumed, reference, receipt = prepare_resume(saved, config, 3, 28000)
    assert reference == 109 and receipt is None
    assert epoch_fraction(resumed['epoch_fraction'], 20, 21, 58) == pytest.approx(fractions[21])


def test_layout_migration_requires_explicit_flag_and_rejects_objective_changes():
    saved = checkpoint()
    config = deepcopy(saved['training_config'])
    with pytest.raises(ValueError, match='changed on resume'):
        prepare_resume(saved, config, 3, 28000)
    config['prior']['lr'] = .001
    with pytest.raises(ValueError, match='cannot change'):
        prepare_resume(saved, config, 3, 28000, True)


def test_unchanged_layout_retains_original_schedule_exactly():
    saved = checkpoint()
    saved['progress'].update(epoch=11, batch=120, step=1259)
    state, reference, _ = prepare_resume(saved, saved['training_config'], 2, 28000)
    for batch in range(120, 218, 2):
        fraction = epoch_fraction(state['epoch_fraction'], 120, batch, 218)
        assert (11 + fraction) * reference == pytest.approx(1199 + batch // 2)


def test_budget_extension_retains_progress_and_schedule_and_requires_permission():
    saved = checkpoint()
    saved['training_config']['prior']['epochs'] = 50
    saved['progress'].update(epoch=50, batch=0, step=7800, best_fid=37.8)
    saved['schedule_reference_updates'] = 156
    contract = deepcopy(saved['training_config'])
    contract['prior']['epochs'] = 75
    with pytest.raises(ValueError, match='explicit extension'):
        prepare_resume(saved, contract, 2, 28000)
    state, updates, receipt = prepare_resume(saved, contract, 2, 28000, allow_epoch_extension=True)
    assert state['epoch'] == 50 and state['step'] == 7800 and state['best_fid'] == 37.8
    assert updates == 156 and receipt['schedule_reference_epochs'] == 50
    assert receipt['from_epochs'] == 50 and receipt['to_epochs'] == 75
    assert saved['training_config']['prior']['epochs'] == 50
    saved.update(training_config=contract, progress=state, schedule_reference_epochs=50)
    resumed, _, receipt = prepare_resume(saved, contract, 2, 28000, allow_epoch_extension=True)
    assert resumed == state and receipt is None


def test_budget_extension_does_not_allow_objective_changes_or_shortening():
    saved = checkpoint()
    saved['training_config']['prior']['epochs'] = 50
    contract = deepcopy(saved['training_config'])
    contract['prior']['epochs'] = 49
    with pytest.raises(ValueError, match='increase'):
        prepare_resume(saved, contract, 2, 28000, allow_epoch_extension=True)
    contract['prior']['epochs'] = 75
    contract['prior']['lr'] = .001
    with pytest.raises(ValueError, match='cannot change'):
        prepare_resume(saved, contract, 2, 28000, allow_layout_change=True, allow_epoch_extension=True)


def test_extended_schedule_holds_original_final_lr_without_jump():
    from src.training.var_laser import lr_wd_annealing
    from types import SimpleNamespace
    optimizer = SimpleNamespace(param_groups=[dict(lr_sc=1., wd_sc=1.)])
    assert schedule_position(30.25, 156, 50) == 30.25*156
    values = []
    for position in [50-1/156, 50, 50.5, 74.999]:
        lr_wd_annealing('lin0', optimizer, .0003, .05, .05,
                       schedule_position(position, 156, 50), 156, 50*156, wp0=.005, wpe=.1)
        values.append(optimizer.param_groups[0]['lr'])
    assert values == pytest.approx([.00003]*4)
