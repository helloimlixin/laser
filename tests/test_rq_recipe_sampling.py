from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from src.training.rq_imagenet_recipes import RQRecipe


def profile(tmp_path, top_k=16384, top_p=.92):
    (tmp_path/'inputs').mkdir()
    config = OmegaConf.create(dict(experiment=dict(sample=dict(top_k=top_k, top_p=top_p))))
    OmegaConf.save(config, tmp_path/'inputs/stage2-training-config.yaml')
    return RQRecipe(tmp_path)


@pytest.mark.parametrize('top_k,top_p', [(16384, .92), ([256, 512, 1024, 2048], [.9, .92, .95, 1.])])
def test_native_sampling_uses_and_records_actual_config(tmp_path, top_k, top_p):
    recipe = profile(tmp_path, top_k, top_p)
    calls = []
    model = SimpleNamespace(sample=lambda tokens, **kwargs: calls.append(kwargs) or tokens)
    labels = torch.tensor([1, 999])
    recipe.sample(model, None, 2, labels)
    expected = dict(temperature=1., top_k=top_k if isinstance(top_k, list) else [top_k],
                    top_p=top_p if isinstance(top_p, list) else [top_p])
    assert recipe.sampling_settings() == expected
    assert recipe.checkpoint_extra()['native_sampling'] == expected
    assert recipe.metadata()['native_sampling'] == expected
    assert {k: calls[0][k] for k in expected} == expected


def test_legacy_resume_explicitly_preserves_previously_hard_coded_sampler(tmp_path):
    recipe = profile(tmp_path)
    recipe.verify_resume(dict(global_step=8764))
    expected = dict(temperature=1., top_k=[256], top_p=[.95])
    assert recipe.sampling_settings() == expected
    assert recipe.checkpoint_extra()['native_sampling'] == expected
    assert recipe.metadata()['native_sampling'] == expected


def test_new_resume_preserves_recorded_sampler_when_yaml_changes(tmp_path):
    recipe = profile(tmp_path)
    saved = recipe.checkpoint_extra()
    assert saved['native_sampling']['top_k'] == [16384]
    recipe._native_sampling_override = dict(temperature=.9, top_k=[512], top_p=[.9])
    recipe.verify_resume(saved)
    assert recipe.sampling_settings() == saved['native_sampling']


def test_released_checkpoint_sampling_takes_precedence_over_training_defaults(tmp_path):
    recipe = profile(tmp_path)
    archive = OmegaConf.create(dict(sampling=dict(top_k=[256], top_p=[.95], temp=.9)))
    OmegaConf.save(archive, tmp_path/'inputs/stage2-archive-config.yaml')
    assert recipe.sampling_settings() == dict(temperature=.9, top_k=[256], top_p=[.95])
