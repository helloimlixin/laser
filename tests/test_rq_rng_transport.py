"""Checkpoint RNG bytes survive tensor transport without consuming a stream."""
import torch

from src.training import rqtransformer as training


def test_rng_tensor_transport_reconstructs_each_rank(monkeypatch):
    torch.manual_seed(822)
    before = torch.get_rng_state().clone()
    other = torch.Generator().manual_seed(823).get_state()
    monkeypatch.setattr(training.dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(training.dist, 'get_backend', lambda: 'gloo')
    monkeypatch.setattr(training.dist, 'get_world_size', lambda: 2)
    monkeypatch.setattr(training, 'rank', lambda: 0)
    def collect(received, local):
        assert local.dtype == torch.uint8 and local.device.type == 'cpu'
        received[0].copy_(local)
        received[1].copy_(other)
    monkeypatch.setattr(training.dist, 'all_gather', collect)
    def reject_object(*a, **kw):
        raise AssertionError('RNG checkpointing must not use object collectives')
    monkeypatch.setattr(training.dist, 'gather_object', reject_object)
    states = training.gather_rank_rng_states(torch.device('cpu'))
    assert torch.equal(states[0]['torch_cpu'], before)
    assert torch.equal(states[1]['torch_cpu'], other)
    assert torch.equal(torch.get_rng_state(), before)
    assert states[0]['torch_cpu'].data_ptr() != before.data_ptr()


def test_nonroot_returns_no_checkpoint_payload(monkeypatch):
    monkeypatch.setattr(training.dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(training.dist, 'get_backend', lambda: 'gloo')
    monkeypatch.setattr(training.dist, 'get_world_size', lambda: 2)
    monkeypatch.setattr(training, 'rank', lambda: 1)
    monkeypatch.setattr(training.dist, 'all_gather', lambda outputs, local: None)
    assert training.gather_rank_rng_states(torch.device('cpu')) is None


def test_single_rank_captures_exact_state(monkeypatch):
    before = torch.get_rng_state().clone()
    monkeypatch.setattr(training.dist, 'is_initialized', lambda: False)
    states = training.gather_rank_rng_states(torch.device('cpu'))
    assert len(states) == 1 and torch.equal(states[0]['torch_cpu'], before)
