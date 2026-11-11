"""Native CC3M labels must use stochastic prefixes from the upstream posterior."""
import torch
from types import SimpleNamespace
from src import rq_upstream
from rqvae.models.rqvae.quantizations import RQBottleneck
from src.training.cc3m_joint import image_targets


def test_native_teacher_matches_upstream_probabilities_and_sampled_prefixes():
    torch.manual_seed(31)
    q = RQBottleneck(latent_shape=[2, 2, 3], code_shape=[2, 2, 4],
                     n_embed=11, decay=.99, shared_codebook=False,
                     restart_unused_codes=False).eval().requires_grad_(False)
    x = torch.randn(3, 2, 2, 3)
    cfg = dict(tokenizer_backend='native_vector', temperatures=.085,
               encoder_batch=2, target_chunk=2)
    torch.manual_seed(17)
    expected = [q.get_soft_codes(part, temp=.085, stochastic=True) for part in x.split(2)]
    torch.manual_seed(17)
    targets, codes = image_targets(SimpleNamespace(quantizer=q), x, cfg)
    torch.testing.assert_close(targets, torch.cat([p[0] for p in expected]), rtol=0, atol=0)
    assert torch.equal(codes, torch.cat([p[1] for p in expected]))
    torch.testing.assert_close(targets.sum(-1), torch.ones_like(codes, dtype=torch.float32))
    residual = x.clone()
    for depth in range(4):
        distances = q.codebooks[depth].compute_distances(residual)
        torch.testing.assert_close(targets[..., depth, :], (-distances/.085).softmax(-1))
        residual -= q.codebooks[depth].embed(codes[..., depth])
    torch.testing.assert_close(q.embed_code(codes), x-residual, rtol=1e-5, atol=1e-6)
