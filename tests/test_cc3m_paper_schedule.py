import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch

from src.training.cc3m_text import create_lr_scheduler


def test_cosine_sequence_matches_authors_scheduler_and_saved_adam_resume():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('authors_scheduler',
        root / 'third_party/rq-vae-transformer/rqvae/optimizer/scheduler.py')
    authors = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(authors)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005, betas=(.9, .95), weight_decay=.0001)
    reference = torch.optim.AdamW(torch.nn.Linear(2, 1).parameters(), lr=.0005)
    options = dict(epochs=100, lr=.0005, min_lr=0., warmup_epochs=0, lr_schedule='cosine')
    schedule = create_lr_scheduler(optimizer, options, 12)
    published = authors.create_scheduler(reference, SimpleNamespace(
        epoch=0, buffer_epoch=0, multiplier=1, min_lr=0., mode='fix', start_from_zero=True), 12, 100)
    assert optimizer.param_groups[0]['lr'] == .0005
    for step in range(1200):
        assert schedule.get_last_lr() == published.get_last_lr()
        if step == 12:
            restored_model = copy.deepcopy(model)
            restored = torch.optim.AdamW(restored_model.parameters(), lr=.0005)
            restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))
            resumed = create_lr_scheduler(restored, options, 12, step, schedule.state_dict(), options)
        model(torch.ones(2, 2)).square().mean().backward()
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        reference.step(); schedule.step(); published.step()
        if step >= 12:
            restored_model(torch.ones(2, 2)).square().mean().backward()
            restored.step(); resumed.step(); restored.zero_grad(set_to_none=True)
            assert restored.param_groups[0]['lr'] == optimizer.param_groups[0]['lr']
            for a, b in zip(model.parameters(), restored_model.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert schedule.get_last_lr() == [0.]
    assert schedule.state_dict()['last_epoch'] == 1200
