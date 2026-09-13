import copy
import json
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from src.church_fid_lr import FidLearningRate, synchronize_observation
from src.church_joint_geometry import early_decay_lr


def score(fid, **kwargs):
    return {'fid': fid, 'samples': 4096, 'seed': 12701, 'world_size': 2,
            'generation_batch_per_gpu': 512, **kwargs}


def test_reduce_only_after_two_misses_then_cooldown():
    monitor = FidLearningRate()
    assert monitor.observe(10, score(22.6), 5e-5)['decision'] == 'baseline'
    assert monitor.observe(20, score(22.8), 5e-5)['decision'] == 'watch'
    assert monitor.learning_rate(5e-5) == 5e-5
    assert monitor.observe(30, score(22.5), 5e-5)['decision'] == 'reduced'
    assert monitor.learning_rate(5e-5) == 2.5e-5
    assert monitor.observe(40, score(23), 5e-5)['decision'] == 'cooldown'
    assert monitor.observe(50, score(23), 5e-5)['decision'] == 'cooldown'
    assert monitor.observe(60, score(23), 5e-5)['decision'] == 'watch'
    assert monitor.observe(70, score(23), 5e-5)['decision'] == 'reduced'


def test_improvement_resets_streak_and_floor_never_raises_base_schedule():
    m = FidLearningRate()
    m.observe(1, score(22), 5e-5)
    m.observe(2, score(23), 5e-5)
    assert m.observe(3, score(21.75), 5e-5)['decision'] == 'improved'
    assert m.bad_checks == 0
    for step in range(4, 50): m.observe(step, score(22), 1e-6)
    assert m.learning_rate(1e-6) == 1e-6 and m.reductions == 0


def test_no_reduction_preserves_lr_exactly_and_reduction_retains_decay():
    m = FidLearningRate()
    for step in (0, 4930, 7395, 9860, 147900):
        base = early_decay_lr(step, 147900, 493)
        assert m.learning_rate(base) == base
    for step, fid in enumerate((22, 23, 23)): m.observe(step, score(fid), 5e-5)
    rates = [m.learning_rate(early_decay_lr(s, 147900, 493)) for s in (9860, 20000, 147900)]
    assert rates[0] > rates[1] > rates[2] == 1e-6


def test_duplicate_cached_evaluation_and_resume_do_not_repeat_a_cut():
    m = FidLearningRate()
    for step, fid in enumerate((22, 23, 23)): m.observe(step, score(fid), 5e-5)
    saved = json.loads(json.dumps(m.state_dict()))
    resumed = FidLearningRate.from_state(saved)
    assert resumed.observe(2, score(23), 5e-5) is None
    assert resumed.state_dict() == saved
    for step, fid in enumerate((23, 23, 23, 23, 20), start=3):
        assert m.observe(step, score(fid), 4e-5) == resumed.observe(step, score(fid), 4e-5)
    assert m.state_dict() == resumed.state_dict()


def test_protocol_change_rebaselines_and_rejects_50k_and_invalid_scores():
    m = FidLearningRate()
    m.observe(1, score(22), 5e-5); m.observe(2, score(23), 5e-5)
    assert m.observe(3, score(24, seed=999), 5e-5)['decision'] == 'baseline'
    assert m.bad_checks == 0 and m.multiplier == 1
    before = m.state_dict()
    for invalid in (score(float('nan')), score(-1), score(23, samples=50000), score(23, optimizer_step=999)):
        with pytest.raises(ValueError): m.observe(4, invalid, 5e-5)
        assert m.state_dict() == before
    with pytest.raises(ValueError): m.observe(3, score(99, seed=999), 5e-5)


def _sync_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://'+rendezvous, rank=rank, world_size=2)
    try:
        m = FidLearningRate()
        parameter = torch.nn.Parameter(torch.tensor([1.]))
        optimizer = torch.optim.AdamW([parameter], lr=5e-5)
        for step, fid in enumerate((22, 23, 23, 23, 23)):
            # Only rank zero's metric may determine the shared decision.
            event = synchronize_observation(m, step, score(fid if rank == 0 else 99), 5e-5)
            for group in optimizer.param_groups: group['lr'] = m.learning_rate(5e-5)
            optimizer.zero_grad(); parameter.square().sum().backward(); optimizer.step()
            if step == 2:
                state = copy.deepcopy({'model': parameter.detach(), 'optimizer': optimizer.state_dict(), 'monitor': m.state_dict()})
                m = FidLearningRate.from_state(state['monitor'])
                optimizer.load_state_dict(state['optimizer'])
            assert event['lr_after'] == optimizer.param_groups[0]['lr']
        values = [None, None]
        dist.all_gather_object(values, {'parameter': parameter.item(), 'monitor': m.state_dict(), 'lr': optimizer.param_groups[0]['lr']})
        assert values[0] == values[1] and values[0]['lr'] == 2.5e-5
    finally: dist.destroy_process_group()


def test_reduction_and_resume_are_synchronized_across_ranks(tmp_path):
    mp.spawn(_sync_worker, args=(str(tmp_path/'rendezvous'),), nprocs=2, join=True)
