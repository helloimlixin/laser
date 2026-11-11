"""Execution controls must also apply through train.py's direct run(cfg) route."""
import json
import pytest
import torch
from src.training import cc3m_joint as trainer
from src.training import cc3m_paper_metrics


def test_public_run_applies_encoder_execution_overrides_before_gpu_setup(tmp_path, monkeypatch):
    path=tmp_path/'execution.json'
    path.write_text(json.dumps(dict(encoder_batch=32,cudnn_benchmark=True,cudnn_benchmark_limit=3)))
    monkeypatch.setenv('LASER_CC3M_EXECUTION_OVERRIDES',str(path))
    before=(torch.backends.cudnn.benchmark,torch.backends.cudnn.benchmark_limit,torch.backends.cudnn.allow_tf32)
    def stop_before_gpu(cfg):
        assert trainer.execution_options(cfg)['encoder_batch']==32
        assert torch.backends.cudnn.benchmark
        assert torch.backends.cudnn.benchmark_limit==3
        assert not torch.backends.cudnn.allow_tf32
        raise RuntimeError('verified before GPU setup')
    monkeypatch.setattr(cc3m_paper_metrics,'load_reference',stop_before_gpu)
    try:
        with pytest.raises(RuntimeError,match='verified before GPU setup'):
            trainer.run(dict(encoder_batch=8,target_chunk=4))
    finally:
        trainer.configure_execution({})
        torch.backends.cudnn.benchmark,torch.backends.cudnn.benchmark_limit,torch.backends.cudnn.allow_tf32=before


@pytest.mark.parametrize('options',[dict(cudnn_benchmark=1),dict(cudnn_benchmark_limit=0)])
def test_invalid_convolution_execution_controls_are_rejected(options):
    with pytest.raises(ValueError):trainer.configure_execution(options)
