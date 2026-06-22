from omegaconf import OmegaConf

from train import _sample_text_prompts


def test_sample_text_prompts_prefers_train_ar_prompts():
    cfg = OmegaConf.create(
        {
            "ar": {"sample_text_prompts": ["old location"]},
            "train_ar": {"sample_text_prompts": ["train location"]},
        }
    )

    assert _sample_text_prompts(cfg) == ["train location"]


def test_sample_text_prompts_falls_back_to_ar_prompts():
    cfg = OmegaConf.create(
        {
            "ar": {"sample_text_prompts": ["old location"]},
            "train_ar": {"sample_text_prompts": []},
        }
    )

    assert _sample_text_prompts(cfg) == ["old location"]
