import json

import pytest

from scripts.tools.queue_imagenet_expanded_scratch import (
    claim_start, completion_gate, gpu_memory_idle, validate_final_payload,
)


def test_successful_exit_alone_does_not_start_scratch():
    launch = dict(state='finished', returncode=0, training_pid=12, supervisor_pid=11)
    assert completion_gate(launch, 'Epoch 95: FID=20.0; saved /x', '/parent', lambda *_: False) == 'waiting_for_final_fid'
    assert completion_gate(launch, 'Epoch 100: FID=nan; saved /x', '/parent', lambda *_: False) == 'waiting_for_final_fid'
    assert completion_gate(launch, 'Epoch 100: FID=21.1234; IS=60; saved /x', '/parent', lambda *_: False) is None


@pytest.mark.parametrize('state,code', [('running', None), ('failed', 1), ('stopped', 0), ('finished', 1)])
def test_running_failed_and_interrupted_parents_never_trigger_scratch(state, code):
    assert completion_gate(dict(state=state, returncode=code), 'Epoch 100: FID=20; saved /x', '/parent', lambda *_: False)


def test_finished_parent_still_holding_processes_is_not_ready():
    launch = dict(state='finished', returncode=0, training_pid=12, supervisor_pid=11)
    assert completion_gate(launch, 'Epoch 100: FID=20; saved /x', '/parent', lambda *_: True) == 'waiting_for_parent_process_exit'


def test_all_eight_gpus_must_be_released():
    assert gpu_memory_idle(['0'] * 8)
    assert not gpu_memory_idle(['0'] * 7)
    assert not gpu_memory_idle(['0'] * 7 + ['120000'])


def test_one_shot_claim_cannot_launch_twice(tmp_path):
    claim_start(tmp_path, {'run_id': 'scratch'})
    with pytest.raises(FileExistsError):
        claim_start(tmp_path, {'run_id': 'duplicate'})
    assert json.loads((tmp_path / 'start-request.json').read_text())['run_id'] == 'scratch'


def final_payload():
    return dict(epoch=100, global_step=63500, fid=20.,
        config=dict(wandb_id='parent', epochs=100, combination_target_policy={'version': 'v2'},
                    total_batch_size=2016, optimizer_steps_per_epoch=635,
                    training_data_mode='online-fresh-images'),
        checkpoint_world_size=8, rng_state_by_rank=[{} for _ in range(8)],
        state_dict={i: None for i in range(870)},
        optimizer=dict(state={i: {'step': 63500} for i in range(870)}),
        scheduler=dict(last_epoch=63500, _laser_schedule_config={'total_steps': 63500}),
        best_fid=[(20., '/x')], best_inception=[(60., '/y')])


def test_last_training_step_checkpoint_without_final_epoch_eval_is_rejected():
    x = final_payload()
    validate_final_payload(x, 'parent', {'version': 'v2'})
    x['epoch'] = 99
    x['batch_idx'] = 635
    with pytest.raises(AssertionError):
        validate_final_payload(x, 'parent', {'version': 'v2'})


def test_final_checkpoint_must_match_run_policy_and_optimizer_budget():
    x = final_payload()
    with pytest.raises(AssertionError):
        validate_final_payload(x, 'different-run', {'version': 'v2'})
    with pytest.raises(AssertionError):
        validate_final_payload(x, 'parent', {'version': 'v1'})
    x['optimizer']['state'][0]['step'] = 63499
    with pytest.raises(AssertionError):
        validate_final_payload(x, 'parent', {'version': 'v2'})
