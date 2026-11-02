import math
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from src import rq_upstream
from rqvae.optimizer.scheduler import create_scheduler


def test_no_warmup_matches_published_cosine_over_every_optimizer_update():
    recipe = OmegaConf.load(Path(__file__).parents[1]/
        'configs/stage2/imagenet-laser-compound512-original-schedule-scratch100-20261016.yaml')
    assert recipe.optimization.selection == 'published_original_rq'
    assert recipe.optimization.warmup_epochs == 0 and recipe.optimization.minimum_lr_ratio == 0
    assert recipe.epochs == recipe.lr_schedule_epochs == 100
    config = OmegaConf.create(dict(epoch=0,multiplier=1,buffer_epoch=0,min_lr=0.,
        mode='fix',start_from_zero=True))
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(()))], lr=.0005)
    scheduler = create_scheduler(optimizer, config, 626, recipe.lr_schedule_epochs)
    assert scheduler.warmup_scheduler is None
    assert optimizer.param_groups[0]['lr'] == .0005
    assert scheduler.after_scheduler.T_max == 62600
    for step in range(1,62601):
        optimizer.step();scheduler.step()
        expected = .0005*(1+math.cos(math.pi*step/62600))/2
        assert optimizer.param_groups[0]['lr'] == pytest.approx(expected,abs=1.e-14)
    assert optimizer.param_groups[0]['lr'] == 0.
