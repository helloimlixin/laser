"""Cross-check epoch partitioning, reproducibility, and native text causality."""
import numpy as np
import torch
import random
import pytest

from src.data.cc3m_joint import epoch_order, text_tokenizer
from src.data.seeded_bpe import SeededBPE
from src import rq_upstream
from src.models.laser_joint_tokenizer import JointAtomCoefficientRQ
from src.training.rq_sdpa_backend import sdpa_attention


def test_real_augmentation_replays_without_reseeding_training_cuda(tmp_path):
    from PIL import Image
    from src.data.cc3m_joint import CC3MJointDataset
    raw = tmp_path/'image.png'
    pixels = np.random.default_rng(19).integers(0, 256, (320, 400, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(raw)
    data = CC3MJointDataset([dict(items=1)], tmp_path, tmp_path, train=True, seed=261017)
    data.records[0] = (raw, [(0, raw.stat().st_size, 'a photograph of a mountain')])
    cpu_before = torch.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state().clone() if torch.cuda.is_available() else None
    first, text, _ = data[0]
    second, repeated_text, _ = data[0]
    assert torch.equal(first, second) and torch.equal(text, repeated_text)
    assert first.shape == (3, 256, 256) and -1 <= first.min() <= first.max() <= 1
    assert torch.equal(cpu_before, torch.get_rng_state())
    if cuda_before is not None:
        assert torch.equal(cuda_before, torch.cuda.get_rng_state())


def test_loader_does_not_advance_training_rng():
    from src.training.cc3m_joint import loader
    data = torch.utils.data.TensorDataset(torch.arange(8))
    before = torch.get_rng_state().clone()
    assert torch.equal(torch.cat([x[0] for x in loader(data, range(8), 2, 0)]), torch.arange(8))
    assert torch.equal(before, torch.get_rng_state())


def test_evaluation_setup_and_failures_preserve_all_rng_and_model_mode():
    from src.training.cc3m_joint import isolated_evaluation
    from src.training.rq_rng import configure_training_rng
    model = torch.nn.Linear(2, 2).eval()
    configure_training_rng(721, 0)
    expected = (random.random(), np.random.random(), torch.rand(8))
    configure_training_rng(721, 0)

    @isolated_evaluation
    def failing_evaluator(model, aux, cfg, device):
        # Metric model construction used to happen outside the RNG guard.
        torch.nn.Linear(8, 8)
        random.random()
        np.random.random()
        model.train()
        raise RuntimeError('metric setup failed')

    with pytest.raises(RuntimeError, match='metric setup failed'):
        failing_evaluator(model, None, {'seed': 999}, 'cpu')
    actual = (random.random(), np.random.random(), torch.rand(8))
    assert actual[:2] == expected[:2]
    assert torch.equal(actual[2], expected[2])
    assert not model.training


def test_group_order_is_complete_and_distributed_resume_has_no_overlap():
    specs = [dict(items=i+17) for i in range(37)]
    order = epoch_order(specs, 261017, 4)
    np.testing.assert_array_equal(np.sort(order), np.arange(sum(s['items'] for s in specs)))
    np.testing.assert_array_equal(order, epoch_order(specs, 261017, 4))
    assert not np.array_equal(order, epoch_order(specs, 261017, 5))
    full = order[:len(order)//64*64]
    partitions = [full[rank::8] for rank in range(8)]
    np.testing.assert_array_equal(np.stack(partitions, axis=1).ravel(), full)
    offset = 3*64
    for rank in range(8):
        np.testing.assert_array_equal(full[offset:][rank::8], partitions[rank][offset//8:])


def test_seeded_caption_dropout_and_eval_match_released_tokenizer():
    tok = text_tokenizer()
    deterministic = SeededBPE(dropout=0., tokenizer=tok)
    stochastic = SeededBPE(dropout=.1, tokenizer=tok)
    captions = ['A brown dog running through a grassy field.', 'A white building near the sea.']
    for caption in captions:
        assert deterministic.encode(caption, 51).ids == tok.encode(caption).ids
        assert stochastic.encode(caption, 51).ids == stochastic.encode(caption, 51).ids


def test_preview_reaches_wandb_as_full_range_uint8():
    import wandb
    from src.training.cc3m_joint import preview_image
    pixels = torch.linspace(0, 1, 16*16).reshape(1, 16, 16).expand(3, -1, -1)
    logged = wandb.Image(preview_image(pixels))
    image = np.asarray(logged.image)
    assert image.dtype == np.uint8
    assert image.min() == 0 and image.max() == 255
    assert 120 < image.mean() < 135


def test_native_text_prefix_is_causal_and_cached_sdpa_matches_full_forward():
    from omegaconf import OmegaConf
    torch.manual_seed(11)
    cfg = OmegaConf.create(dict(embed_dim=32, input_embed_dim=8,
        block_size=[2, 2, 5], vocab_size=17, vocab_size_cond=23, block_size_cond=4,
        input_emb_vqvae=True, head_emb_vqvae=True, cumsum_depth_ctx=True,
        shared_cls_emb=True, shared_tok_emb=True,
        body=dict(n_layer=2, block=dict(n_head=4, resid_pdrop=0.)),
        head=dict(n_layer=2, block=dict(n_head=4, resid_pdrop=0.))))
    model = rq_upstream.RQTransformer(rq_upstream.RQTransformerConfig.create(cfg)).eval()
    quantizer = JointAtomCoefficientRQ(torch.randn(8, 7),
        torch.randint(0, 7, (5, 17)), torch.cat((torch.zeros(5, 1), torch.randn(5, 16)), 1))
    codes, text = torch.randint(0, 17, (2, 2, 2, 5)), torch.randint(0, 23, (2, 4))
    with torch.no_grad():
        original, original_text = model(codes, quantizer, cond=text)
        with sdpa_attention(model):
            full, text_logits = model(codes, quantizer, cond=text)
            torch.testing.assert_close(full, original, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(text_logits, original_text, atol=1e-6, rtol=1e-5)
            changed = text.clone()
            changed[:, -1] = (changed[:, -1]+1)%23
            _, changed_logits = model(codes, quantizer, cond=changed)
            torch.testing.assert_close(text_logits, changed_logits, atol=0, rtol=0)
            model.init_cache()
            for h in range(2):
                for w in range(2):
                    for d in range(5):
                        cached = model.cached_forward(codes[:, :h+1], quantizer, cond=text,
                            sample_loc=(h, w, d))
                        torch.testing.assert_close(cached, full[:, h, w, d], atol=1e-6, rtol=1e-5)
            model.init_cache()


def test_native_bf16_guard_overrides_the_legacy_fp16_default():
    from omegaconf import OmegaConf
    from src.training.rq_precision import native_bf16_autocast
    import pytest
    if not torch.cuda.is_available():
        pytest.skip('CUDA mixed precision regression')
    cfg = OmegaConf.create(dict(embed_dim=32, input_embed_dim=8,
        block_size=[2, 2, 5], vocab_size=17, vocab_size_cond=23, block_size_cond=4,
        input_emb_vqvae=True, head_emb_vqvae=True, cumsum_depth_ctx=True,
        shared_cls_emb=True, shared_tok_emb=True,
        body=dict(n_layer=1, block=dict(n_head=4, resid_pdrop=0.)),
        head=dict(n_layer=1, block=dict(n_head=4, resid_pdrop=0.))))
    model = rq_upstream.RQTransformer(rq_upstream.RQTransformerConfig.create(cfg)).cuda()
    quantizer = JointAtomCoefficientRQ(torch.randn(8, 7),
        torch.randint(0, 7, (5, 17)), torch.cat((torch.zeros(5, 1), torch.ones(5, 16)), 1)).cuda()
    codes = torch.randint(0, 17, (2, 2, 2, 5), device='cuda')
    text = torch.randint(0, 23, (2, 4), device='cuda')
    observed = []
    hook = model.body_transformer.blocks[0].attn.query.register_forward_hook(
        lambda module, args, output: observed.append(output.dtype))
    with native_bf16_autocast(), sdpa_attention(model), torch.autocast('cuda', dtype=torch.bfloat16):
        logits, text_logits = model(codes, quantizer, cond=text, amp=True)
        assert logits.dtype == text_logits.dtype == torch.bfloat16
        assert observed == [torch.bfloat16]
        (logits.float().square().mean()+text_logits.float().square().mean()).backward()
    hook.remove()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
