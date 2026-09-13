import numpy as np
import pytest
import torch

from src.audio_k4_entropy import SparseHuffman
from src.mdctcodec_k4 import K4Dictionary


def test_huffman_roundtrip_with_repeats_unseen_symbols_and_real_bytes():
    counts = np.zeros((4, 1026)); counts[:, 0] = 1000; counts[:,-1] = 10000
    codec = SparseHuffman(counts)
    tokens = np.zeros((150,4), dtype=np.int64); tokens[75] = [1024, 2, 3, 4]
    payload = codec.encode(tokens)
    np.testing.assert_array_equal(codec.decode(payload), tokens)
    np.testing.assert_array_equal(SparseHuffman(counts.copy()).decode(payload), tokens)
    assert len(payload)*8 < 6000  # compressible synthetic sequence; not an audio rate claim
    with pytest.raises(ValueError): codec.decode(payload[:-1])
    bad = bytearray(payload); bad[-1] ^= 1
    with pytest.raises(ValueError): codec.decode(bytes(bad))
    with pytest.raises(ValueError): codec.encode(tokens+1025)
    random = np.random.default_rng(5).integers(0,1025,size=(13,4))
    np.testing.assert_array_equal(codec.decode(codec.encode(random)),random)


def quantizer(atoms=4096):
    return K4Dictionary(num_embeddings=atoms, embedding_dim=8, sparsity_level=4,
        omp_ridge=.05, coefficient_quantization_bits=4, coefficient_quantization_max=1.,
        dictionary_update_mode='alternating_residual')


@pytest.mark.parametrize('atoms',[1024,4096])
def test_k4_learned_atoms_quantized_values_gradients_and_transport(atoms):
    q = quantizer(atoms); z = torch.randn(2,8,1,12,requires_grad=True)
    reconstruction, loss, codes = q(z)
    assert q.dictionary.shape == (8,atoms) and codes.support.shape[-1] == 4
    tokens = q.joint_tokens(codes.support,codes.values)
    support,values = q.from_joint_tokens(tokens)
    restored = q._reconstruct_sparse(support,values,1,12)
    torch.testing.assert_close(restored,reconstruction,atol=2e-7,rtol=1e-6)
    (reconstruction.square().mean()+loss).backward()
    assert z.grad is not None and torch.isfinite(z.grad).all() and z.grad.abs().sum()>0
    before = q.coefficient_levels.clone(); q.update_levels_after_batch()
    assert not torch.equal(before,q.coefficient_levels)
    q.eval(); levels=q.coefficient_levels.clone(); q(z.detach())
    torch.testing.assert_close(q.coefficient_levels,levels,atol=0,rtol=0)
    clone=quantizer(atoms); clone.load_state_dict(q.state_dict(),strict=True)
    torch.testing.assert_close(clone(z.detach())[0],q(z.detach())[0],atol=0,rtol=0)
    # Exercise the final atom and coefficient IDs, which exceed old A1024 bounds.
    boundary=torch.tensor([[[[1,8,1+(atoms-1)*8,atoms*8]]]])
    support,values=q.from_joint_tokens(boundary)
    torch.testing.assert_close(q.joint_tokens(support,values),boundary,atol=0,rtol=0)
    transport=SparseHuffman(np.zeros((4,q.joint_vocabulary+1)))
    np.testing.assert_array_equal(transport.decode(transport.encode(boundary.reshape(-1,4).numpy())),boundary.reshape(-1,4).numpy())


def test_wrong_sparsity_and_atom_count_are_rejected():
    with pytest.raises(ValueError,match='K=4'):
        K4Dictionary(num_embeddings=1024,embedding_dim=8,sparsity_level=2)
    with pytest.raises(ValueError,match='1024'):
        K4Dictionary(num_embeddings=128,embedding_dim=8,sparsity_level=4)


def test_epoch_audit_rejects_a_different_crop_stream(tmp_path):
    import json
    from types import SimpleNamespace
    from src.mdctcodec_k4 import K4TrainingStatistics
    record={'epoch':0,'generator_updates':852,'batches':852,'data_order_sha256':'matched'}
    reference=tmp_path/'rvq.jsonl'; reference.write_text(json.dumps(record)+'\n')
    actual=tmp_path/'data_order.jsonl';actual.write_text(json.dumps(record)+'\n')
    callback=K4TrainingStatistics(output=tmp_path,reference_order=reference)
    trainer=SimpleNamespace(limit_train_batches=1.,current_epoch=0)
    callback.on_train_epoch_end(trainer,None)
    record['data_order_sha256']='different'
    actual.write_text(json.dumps(record)+'\n')
    with pytest.raises(RuntimeError,match='data/crop audit'):
        callback.on_train_epoch_end(trainer,None)
    record['batches']=2;record['generator_updates']=2
    actual.write_text(json.dumps(record)+'\n')
    # An intentional mid-epoch stop preserves a checkpoint without claiming
    # the partial epoch matched a complete 852-batch reference digest.
    callback.on_train_epoch_end(trainer,SimpleNamespace(continuation_stopped=True))
