import pytest
from src.tts_schedule import learning_rate_at


def test_continuation_does_not_restart_original_peak_lr():
    config = {'continuation': {'start_step': 28545, 'start_lr': 3e-5, 'end_lr': 1e-5}}
    values = [learning_rate_at(step, config, 42840) for step in range(28545, 42841)]
    assert values[0] == pytest.approx(3e-5)
    assert values[-1] == pytest.approx(1e-5)
    assert all(a >= b for a, b in zip(values, values[1:]))
    with pytest.raises(ValueError):
        learning_rate_at(0, config, 42840)


def test_original_warmup_and_cosine_endpoints():
    config = {'learning_rate': 3e-4, 'warmup_steps': 500, 'min_lr_ratio': .1}
    assert learning_rate_at(0, config, 28560) == pytest.approx(6e-7)
    assert learning_rate_at(500, config, 28560) == pytest.approx(3e-4)
    assert learning_rate_at(28560, config, 28560) == pytest.approx(3e-5)


def test_reheat_is_continuous_and_bounded():
    config = {'continuation': {'start_step': 42821, 'start_lr': 1e-5,
        'peak_lr': 1e-4, 'warmup_steps': 1000, 'end_lr': 1e-5}}
    assert learning_rate_at(42821, config, 114240) == pytest.approx(1e-5)
    assert learning_rate_at(43821, config, 114240) == pytest.approx(1e-4)
    assert learning_rate_at(114240, config, 114240) == pytest.approx(1e-5)
    assert learning_rate_at(43820, config, 114240) < learning_rate_at(43821, config, 114240)
    assert learning_rate_at(43822, config, 114240) < learning_rate_at(43821, config, 114240)
    with pytest.raises(ValueError):
        learning_rate_at(42821, config, 43000)
