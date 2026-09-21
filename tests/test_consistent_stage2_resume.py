"""Compare uninterrupted and resumed stochastic cached-data training exactly."""
import copy
from itertools import islice
import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler

from src.training.stage2_resume import (capture_rng_state, validate_resume_payload,
    restore_training_state, resumed_iterator)


CACHE = dict(checkpoint_sha256='tokenizer', codebook_sha256='book', cache_sha256='latents',
             data_protocol_sha256='pixels', shape=[23, 4], images=23, world_size=2)
CALIBRATION = dict(selected_temperature=.125)


def seed(value):
    torch.manual_seed(value)
    np.random.seed(value)
    random.seed(value)


def make_state(rank):
    model = nn.Sequential(nn.Linear(4, 7), nn.GELU(), nn.Dropout(.3), nn.Linear(7, 4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=6)
    scaler = torch.amp.GradScaler('cpu', init_scale=16., growth_interval=2)
    dataset = torch.arange(92, dtype=torch.float32).reshape(23, 4) / 92
    sampler = DistributedSampler(dataset, num_replicas=2, rank=rank, seed=13)
    loader = DataLoader(dataset, sampler=sampler, batch_size=2, num_workers=2, persistent_workers=True)
    return model, optimizer, scheduler, scaler, loader, sampler


def checkpoint(state, epoch, batch, step, rank):
    model, optimizer, scheduler, scaler, _, _ = state
    rng = capture_rng_state('cpu')
    return copy.deepcopy(dict(state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), rng_states=[rng, rng],
        epoch=epoch, batch_in_epoch=batch, step=step, attempts=step, skipped_amp_updates=0,
        tokenizer=CACHE, temperature_calibration=CALIBRATION, initial_weights_sha256='initial',
        config=dict(experiment=dict(batch_size=2, total_batch_size=8, accumulation_steps=2))))


def train(state, rank, *, saved=None, save_at=None):
    model, optimizer, scheduler, scaler, loader, sampler = state
    start_epoch, start_batch, step = (0, 0, 0) if saved is None else (saved['epoch'], saved['batch_in_epoch'], saved['step'])
    records = []
    captured = None
    for epoch in range(start_epoch, 2):
        consumed = start_batch if epoch == start_epoch else 0
        iterator = resumed_iterator(loader, sampler, epoch, consumed,
            rng_state=saved['rng_states'][rank] if saved is not None and epoch == start_epoch else None)
        while batches := list(islice(iterator, 2)):
            optimizer.zero_grad(set_to_none=True)
            count = sum(len(batch) for batch in batches)
            loss_sum = 0.
            for batch in batches:
                # Model dropout, target sampling, NumPy and Python each consume
                # a saved RNG stream; worker startup must not perturb them.
                target = torch.multinomial(torch.full((len(batch), 4), .25), 1).squeeze(-1)
                prediction = model(batch + float(np.random.rand()) * .01 + random.random() * .01)
                loss = nn.functional.cross_entropy(prediction, target) * len(batch) / count
                scaler.scale(loss).backward()
                loss_sum += float(loss.detach())
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            consumed += len(batches)
            step += 1
            records.append((epoch, consumed, loss_sum))
            if (epoch, consumed) == save_at:
                captured = checkpoint(state, epoch, consumed, step, rank)
        if (epoch + 1, 0) == save_at:
            captured = checkpoint(state, epoch + 1, 0, step, rank)
    return records, captured


def assert_nested_equal(actual, expected):
    if torch.is_tensor(expected):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_nested_equal(a, b)
    else:
        assert actual == expected


@pytest.mark.parametrize('rank', [0, 1])
@pytest.mark.parametrize('cursor', [(0, 2), (0, 6), (1, 0)])
def test_resume_reproduces_loss_parameters_optimizer_scheduler_scaler_and_rng(tmp_path, rank, cursor):
    seed(91)
    control = make_state(rank)
    seed(400 + rank)
    control_records, saved = train(control, rank, save_at=cursor)
    expected_rng = capture_rng_state('cpu')
    path = tmp_path / 'last.pt'
    torch.save(saved, path)
    payload = torch.load(path, weights_only=False, mmap=True)
    validate_resume_payload(payload, world=2, batch_size=2, cache=CACHE, calibration=CALIBRATION,
                            loader_batches=6, accumulation=2)
    seed(9999)
    resumed = make_state(rank)
    restore_training_state(payload, *resumed[:4])
    resumed_records, _ = train(resumed, rank, saved=payload)
    assert resumed_records == control_records[payload['step']:]
    for actual, expected in zip(resumed[:4], control[:4]):
        assert_nested_equal(actual.state_dict(), expected.state_dict())
    assert torch.equal(capture_rng_state('cpu')['torch'], expected_rng['torch'])
    assert random.getstate() == expected_rng['python']
    np.testing.assert_array_equal(np.random.get_state()[1], expected_rng['numpy'][1])
    for state in (control, resumed):
        state[4]._iterator._shutdown_workers()


@pytest.mark.parametrize('fault,match', [('world','world size'), ('batch','batch size'),
    ('cache','cache mismatch'), ('offset','boundary'), ('missing','Incomplete'), ('calibration','calibration')])
def test_resume_rejects_incompatible_or_incomplete_states(fault, match):
    state = make_state(0)
    payload = checkpoint(state, 0, 2, 1, 0)
    options = dict(world=2, batch_size=2, cache=CACHE, calibration=CALIBRATION,
                   loader_batches=6, accumulation=2)
    if fault == 'world': options['world'] = 1
    if fault == 'batch': options['batch_size'] = 1
    if fault == 'cache': options['cache'] = dict(CACHE, codebook_sha256='other')
    if fault == 'offset': payload['batch_in_epoch'] = 1
    if fault == 'missing': del payload['optimizer']
    if fault == 'calibration': options['calibration'] = dict(selected_temperature=.5)
    with pytest.raises(ValueError, match=match):
        validate_resume_payload(payload, **options)


def test_depth_temperature_calibration_is_accepted_only_when_identical():
    calibration = dict(selected_temperature=[.2,.4,.3,.25], target_policy='test')
    payload = checkpoint(make_state(0),0,2,1,0)
    payload['temperature_calibration'] = calibration
    validate_resume_payload(payload,world=2,batch_size=2,cache=CACHE,calibration=calibration)
    with pytest.raises(ValueError,match='calibration'):
        validate_resume_payload(payload,world=2,batch_size=2,cache=CACHE,
                                calibration=dict(calibration,selected_temperature=[.2,.4,.3,.3]))
