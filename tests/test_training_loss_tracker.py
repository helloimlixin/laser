import pytest

from src.training.training_loss_tracker import TrainingLossTracker


def test_running_mean_weights_images_and_separates_auxiliary():
    tracker = TrainingLossTracker(40000)
    tracker.update(3., 2., .5, 2., 100, 40001)
    metrics = tracker.update(5., 4., .5, 2., 20, 40002)
    assert metrics['train/loss_running_mean'] == pytest.approx(400 / 120)
    assert metrics['train/cross_entropy_running_mean'] == pytest.approx(280 / 120)
    assert metrics['train/loss_ema'] == pytest.approx(3.1)
    assert metrics['train/coefficient_crps_weight'] == 2.


def test_resume_and_rewind_preserve_tracker_at_saved_model():
    tracker = TrainingLossTracker(40064)
    tracker.update(6.51, 6.5, .2, .05, 2048, 40065)
    best = tracker.state_dict()
    resumed = TrainingLossTracker(40065, best)
    expected = tracker.update(6.41, 6.4, .2, .05, 2048, 40066)
    assert resumed.update(6.41, 6.4, .2, .05, 2048, 40066) == expected
    assert best['last_step'] == 40065
    rewind = TrainingLossTracker(40065, best)
    assert rewind.state_dict() == best
    with pytest.raises(ValueError):
        TrainingLossTracker(40066, best)


def test_rejects_nonfinite_stats_wrong_objective_and_skipped_steps():
    for values in ((float('nan'), 1., 0., 0., 2048, 1),
                   (2., 1., 0., 0., 2048, 1),
                   (1., 1., 0., 0., 0, 1),
                   (1., 1., 0., 0., 2048, 2)):
        with pytest.raises(ValueError):
            TrainingLossTracker(0).update(*values)
