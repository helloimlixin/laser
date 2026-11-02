import math
import pytest
import torch

from src.models.joint_rate_distortion import JointVectorBook, NeuralJointDecoder, rate_statistics


def _write_reports(directory, rank):
    from pathlib import Path
    from src.training.joint_rate_distortion import write
    for iteration in range(60):
        write(Path(directory), 'calibration.json', {'rank':rank, 'iteration':iteration})


def test_integer_stream_roundtrip_and_exact_greedy_search():
    table = torch.tensor([[[0., 0.], [2., 0.], [0., 2.]],
                          [[0., 0.], [1., 0.], [0., 1.]]])
    book = JointVectorBook(table)
    x = torch.tensor([[3., 0.], [0., 3.], [0., 0.]])
    result = book.quantize(x, chunk_size=1)
    assert result['codes'].tolist() == [[1, 1], [2, 2], [0, 0]]
    torch.testing.assert_close(book.embed(result['codes']).sum(-2), result['quantized'])
    torch.testing.assert_close(result['quantized'], x)


def test_joint_refit_reduces_final_error_and_keeps_null_token():
    torch.manual_seed(11)
    table = torch.randn(4, 12, 7); table[:, 0] = 0
    book = JointVectorBook(table)
    x = torch.randn(80, 7)
    codes = book.quantize(x)['codes']
    result = book.refit(x, codes)
    assert result['fixed_code_mse_after'] < result['fixed_code_mse_before']
    assert not book.vectors[:, 0].count_nonzero()


def test_neural_decoder_identity_initialization_and_learning():
    torch.manual_seed(12)
    vectors = torch.randn(2, 3, 3, 4, 8)
    decoder = NeuralJointDecoder(depth=4, channels=8, width=16)
    torch.testing.assert_close(decoder(vectors), vectors.sum(-2), rtol=0, atol=0)
    target = vectors.sum(-2) + .3
    optimizer = torch.optim.Adam(decoder.parameters(), lr=.01)
    initial = (decoder(vectors) - target).square().mean().item()
    for _ in range(4):
        optimizer.zero_grad(); loss = (decoder(vectors)-target).square().mean()
        loss.backward(); optimizer.step()
    assert (decoder(vectors)-target).square().mean() < initial


def test_rate_distinguishes_known_depth_from_transmitted_bits():
    result = rate_statistics([[2, 0, 0, 0], [0, 2, 0, 0]])
    assert result['depth_conditional_entropy_bits_per_token'] == 0
    assert result['marginal_entropy_bits_per_token'] == 1
    assert result['fixed_width_bits_per_token'] == 2
    assert result['fixed_width_bpp'] == 2 * 256 / 65536
    assert not result['entropy_is_autoregressive_rate']
    with pytest.raises(ValueError):rate_statistics([[0, 0]])


def test_report_writers_do_not_share_temporary_files(tmp_path):
    import json
    import multiprocessing as mp
    ctx = mp.get_context('spawn')
    workers = [ctx.Process(target=_write_reports,args=(str(tmp_path),r)) for r in range(4)]
    for worker in workers:worker.start()
    for worker in workers:
        worker.join(timeout=30)
        assert worker.exitcode == 0
    result = json.loads((tmp_path/'calibration.json').read_text())
    assert result['iteration'] == 59 and 0 <= result['rank'] < 4
