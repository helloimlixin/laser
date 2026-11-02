import pytest
import torch

from src.models.multiscale_laser_var import MultiScaleLaser
from src.models.joint_multiscale_var import JointMultiScaleQuantizer, JointLaserVAR


def quantizer():
    torch.manual_seed(19)
    source = MultiScaleLaser(channels=4, atoms=8, sparsity=2,
                             patch_nums=(1, 2, 3), coefficient_bins=33)
    levels = torch.tensor([-1., -.25, .25, 1.]).expand(3, 8, 4).clone()
    return JointMultiScaleQuantizer(source, source.normalized_dictionary(), levels)


def prior(q, monkeypatch):
    import dist
    monkeypatch.setattr(dist, 'get_device', lambda: 'cpu')
    model = JointLaserVAR(q, depth=2, width=32, heads=2, num_classes=3)
    model.cond_drop_rate = 0
    return model.eval()


def test_joint_ids_match_expanded_nearest_neighbors_and_decode_without_side_data():
    q = quantizer()
    x = torch.randn(11, 4)
    codes, reconstructed = q.quantize_scale(x, 1, chunk_size=3)
    vectors = q.embed_joint(torch.arange(q.vocab_size), 1)
    residual = x.clone()
    for d in range(q.sparsity):
        distances = (residual[:, None] - vectors).square().sum(-1)
        if d:
            used = codes[:, :d] // q.levels
            mask = (torch.arange(q.vocab_size)[None, :, None] // q.levels == used[:, None]).any(-1)
            distances.masked_fill_(mask, torch.inf)
        assert torch.equal(codes[:, d], distances.argmin(-1))
        residual.sub_(vectors[codes[:, d]])
    torch.testing.assert_close(reconstructed, q.embed_joint(codes, 1).sum(-2))
    result = q.decompose(torch.randn(2, 4, 3, 3))
    decoded, inputs = q.from_codes(result['codes'])
    torch.testing.assert_close(decoded, result['latent'])
    torch.testing.assert_close(inputs, result['inputs'])
    assert (result['codes'][..., 0] // q.levels != result['codes'][..., 1] // q.levels).all()


def test_depth_and_scale_causality_and_finite_gradients(monkeypatch):
    q = quantizer()
    m = prior(q, monkeypatch)
    result = q.decompose(torch.randn(2, 4, 3, 3))
    features = torch.randn(2, 14, m.C)
    scores = m.token_logits(features, result['codes'])
    changed = result['codes'].clone()
    changed[..., 1] = (changed[..., 1] + 1) % q.vocab_size
    torch.testing.assert_close(scores[..., 0, :], m.token_logits(features, changed)[..., 0, :])
    changed = result['codes'].clone()
    changed[:, 1:] = (changed[:, 1:] + q.levels) % q.vocab_size
    _, inputs = q.from_codes(changed)
    torch.testing.assert_close(inputs[:, :4], result['inputs'][:, :4])
    loss = m(torch.tensor([0, 1]), result['inputs'], result['codes'])
    loss.backward()
    assert loss.isfinite()
    for name, parameter in m.named_parameters():
        assert parameter.grad is not None and parameter.grad.isfinite().all(), name


@pytest.mark.parametrize('cfg', [0., 1.5])
def test_cached_sampling_matches_teacher_forcing(monkeypatch, cfg):
    import src.models.joint_multiscale_var as module
    q = quantizer()
    m = prior(q, monkeypatch)
    result = q.decompose(torch.randn(2, 4, 3, 3))
    codes, labels = result['codes'], torch.tensor([0, 1])
    both_labels = torch.cat((labels, torch.full_like(labels, m.num_classes)))
    features = super(JointLaserVAR, m).forward(both_labels, result['inputs'].repeat(2, 1, 1))
    expected = m.token_logits(features, codes.repeat(2, 1, 1))
    calls = []

    def force_codes(logits, **kwargs):
        scale, depth = divmod(len(calls), q.sparsity)
        lo, hi = m.begin_ends[scale]
        score = expected[:, lo:hi, depth].clone()
        # CFG must combine finite values before masking all levels of used atoms.
        mask = ~score[:2].isfinite()
        score.masked_fill_(~score.isfinite(), 0.)
        guidance = cfg * scale / m.num_stages_minus_1
        score = ((1+guidance)*score[:2] - guidance*score[2:]).masked_fill(mask, -torch.inf)
        torch.testing.assert_close(logits, score, atol=1.e-6, rtol=2.e-4)
        calls.append((scale, depth))
        return codes[:, lo:hi, depth, None]

    monkeypatch.setattr(module, 'sample_with_top_k_top_p_', force_codes)
    latent = m.sample(labels, cfg=cfg)
    torch.testing.assert_close(latent, result['latent'])
    assert len(calls) == len(q.v_patch_nums) * q.sparsity
    assert all(not b.attn.caching for b in m.blocks)


def test_vocabulary_can_match_original_var_exactly():
    source = MultiScaleLaser(channels=4, atoms=1024, sparsity=2,
                             patch_nums=(1, 2), coefficient_bins=33)
    q = JointMultiScaleQuantizer(source, source.normalized_dictionary(),
                                 torch.tensor([-1., -.25, .25, 1.]).expand(2, 1024, 4))
    assert q.vocab_size == 4096
