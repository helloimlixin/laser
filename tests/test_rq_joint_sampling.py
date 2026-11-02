import pytest
from omegaconf import OmegaConf

from src.training.rq_joint_sampling import joint_top_k


def test_scalar_keeps_existing_native_behavior():
    assert joint_top_k(2048,5,20481)==[2048]


def test_per_depth_limits_preserve_native_order_in_recipe_lists():
    limits=[2048,2048,4096,8192,20481]
    assert joint_top_k(OmegaConf.create(limits),5,20481)==limits


def test_unrestricted_and_large_limits_match_native_clamping():
    assert joint_top_k(None,5,20481)==[20481]
    assert joint_top_k([2048,30000,4096,8192,20481],5,20481)==[2048,20481,4096,8192,20481]


@pytest.mark.parametrize('bad',[[1,2],[],0,-1,1.5,'2048',[1,2,3,4,0],True])
def test_invalid_limits_fail_before_generation(bad):
    with pytest.raises(ValueError):joint_top_k(bad,5,20481)
