import importlib.util
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from src.models.mdctcodec_rvq import MDCTCodecRQBottleneck
from src.mdctcodec_matched import PairedAudioDataset, PairedSampler, PairedAudit, TrainingCoefficientRange


def test_rvq_matches_authors_forward_losses_gradients_and_serialized_decoder():
    spec=importlib.util.spec_from_file_location('reference_rvq',
        Path('outputs/mdctcodec_reference/MDCTCodec/quantize.py'))
    reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
    torch.manual_seed(12)
    adapter=MDCTCodecRQBottleneck(num_embeddings=1024,embedding_dim=8,code_depth=4)
    original=reference.ResidualVectorQuantize(input_dim=8,codebook_dim=8,n_codebooks=4,codebook_size=1024)
    original.load_state_dict(adapter.quantizer.state_dict(),strict=True)
    x=torch.randn(2,8,1,12,requires_grad=True)
    x2=x.detach().clone().squeeze(2).requires_grad_(True)
    y,commit,codes=adapter(x)
    ref,ids,_,ref_commit,ref_codebook=original(x2)
    torch.testing.assert_close(y.squeeze(2),ref,rtol=0,atol=0)
    torch.testing.assert_close(codes.support.squeeze(1).transpose(1,2),ids,rtol=0,atol=0)
    torch.testing.assert_close(commit,ref_commit,rtol=0,atol=0)
    codebook=adapter._last_dictionary_loss_for_backward
    torch.testing.assert_close(codebook,ref_codebook,rtol=0,atol=0)
    (y.square().mean()+2.5*commit+10*codebook).backward()
    (ref.square().mean()+2.5*ref_commit+10*ref_codebook).backward()
    torch.testing.assert_close(x.grad.squeeze(2),x2.grad,rtol=1e-6,atol=1e-7)
    for (_,a),(_,b) in zip(adapter.quantizer.named_parameters(),original.named_parameters()):
        torch.testing.assert_close(a.grad,b.grad,rtol=1e-6,atol=1e-7)
    assert all(torch.isfinite(p.grad).all() for p in adapter.parameters())
    from archive.scripts.benchmark_mdctcodec_trained_rvq import payload_roundtrip
    for i in range(2):
        payload,parsed=payload_roundtrip(ids[i:i+1].numpy())
        assert len(payload)==12*5
        decoded=adapter.quantizer.from_codes(torch.from_numpy(parsed))[0]
        torch.testing.assert_close(decoded,ref[i:i+1],rtol=1e-4,atol=1e-5)


def test_pair_data_order_and_crops_ignore_global_rng_and_worker_order(tmp_path):
    paths=[]
    for i in range(5):
        path=tmp_path/f'p225_{i:03d}_mic2.flac'
        sf.write(path,np.linspace(-0.5,0.5,20000,dtype=np.float32),48000)
        paths.append(str(path))
    ds=PairedAudioDataset(paths,seed=1234)
    a=list(PairedSampler(ds,lambda:7))
    torch.rand(1234)
    b=list(PairedSampler(ds,lambda:7))
    assert a==b and list(PairedSampler(ds,lambda:8))!=a
    outputs={key:ds[key] for key in a}
    for key in reversed(b):
        torch.rand(17)
        x,_,meta=ds[key]
        from src.audio_logging import extract_audio_metadata_from_batch
        assert extract_audio_metadata_from_batch((x,0,meta)) is not None
        torch.testing.assert_close(x,outputs[key][0],rtol=0,atol=0)
        assert meta['crop_offset']==outputs[key][2]['crop_offset']
    assert any(ds[(8,i)][2]['crop_offset']!=ds[(7,i)][2]['crop_offset'] for i in range(5))


def test_data_audit_restores_pending_epoch_end_without_duplicate_rows(tmp_path):
    from types import SimpleNamespace
    audit=PairedAudit(tmp_path,100)
    audit.records=['p225_001_mic2.flac:34\n','p225_002_mic2.flac:92\n']
    audit.batches=1
    restored=PairedAudit(tmp_path,100)
    restored.load_state_dict(audit.state_dict())
    trainer=SimpleNamespace(current_epoch=0)
    model=SimpleNamespace(_manual_train_step=torch.tensor(1))
    restored.on_train_epoch_end(trainer,model)
    restored.on_train_epoch_end(trainer,model)
    assert len((tmp_path/'data_order.jsonl').read_text().splitlines())==1


def test_range_observer_follows_scale_drift_and_restores_state(tmp_path):
    from src.models.dictionary_learner import DictionaryLearning
    quantizer=DictionaryLearning(num_embeddings=8,embedding_dim=4,sparsity_level=2,
        coefficient_quantization_bits=7,coefficient_quantization_max=0.9775147438049316)
    values=torch.linspace(30,50,2400).reshape(-1,2)
    quantizer._quantize_coefficients(values)
    assert float(quantizer._last_coefficient_saturation_fraction)==1
    policy=TrainingCoefficientRange(tmp_path)
    bound=policy.observe(quantizer.coefficient_quantization_max,
        float(quantizer._last_coefficient_abs_p999),1,100)
    quantizer.coefficient_quantization_max=bound
    result=quantizer._quantize_coefficients(values)
    assert float(quantizer._last_coefficient_saturation_fraction)==0
    assert (result-values).abs().max()<=bound/126+1e-5
    restored=TrainingCoefficientRange(tmp_path)
    restored.load_state_dict(policy.state_dict())
    assert policy.observe(bound,20,0,101)==restored.observe(bound,20,0,101)
    assert policy.last_bound>bound*0.999  # Slow release, not abrupt shrinkage.
    assert '_last_coefficient_abs_p999' not in quantizer.state_dict()


def test_range_callback_changes_only_next_training_bound_and_guard_fails(tmp_path):
    from types import SimpleNamespace
    import pytest
    bottleneck=SimpleNamespace(coefficient_quantization_max=1.,
        _last_coefficient_abs_p999=torch.tensor(50.),
        _last_coefficient_saturation_fraction=torch.tensor(1.))
    model=SimpleNamespace(bottleneck_type='dictionary',training=True,bottleneck=bottleneck,
        _manual_train_step=torch.tensor(1),hparams={'coefficient_quantization_max':1.},
        logger=SimpleNamespace(log_metrics=lambda *a,**kw:None))
    trainer=SimpleNamespace(global_step=2)
    callback=TrainingCoefficientRange(tmp_path,window=3,guard_start=3)
    callback.on_train_batch_end(trainer,model,None,None,0)
    assert bottleneck.coefficient_quantization_max==model.hparams['coefficient_quantization_max']==55.
    model.training=False
    callback.on_validation_end(trainer,model)
    assert bottleneck.coefficient_quantization_max==55.
    model.training=True
    model._manual_train_step=torch.tensor(2)
    callback.on_train_batch_end(trainer,model,None,None,1)
    model._manual_train_step=torch.tensor(3)
    with pytest.raises(RuntimeError,match='Range guard'):
        callback.on_train_batch_end(trainer,model,None,None,2)
