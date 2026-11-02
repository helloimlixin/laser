from types import MethodType, SimpleNamespace

import torch
from omegaconf import OmegaConf

from src.training import laser_rq_minimal as recipe
from src.models.laser_tokenizer_adapter import FrozenLaserTokenizer


def example():
    torch.manual_seed(317)
    arch = OmegaConf.create(dict(type="rq-transformer", block_size=[2, 2, 4],
        embed_dim=12, input_embed_dim=4, shared_tok_emb=True, shared_cls_emb=True,
        input_emb_vqvae=True, head_emb_vqvae=True, cumsum_depth_ctx=True,
        vocab_size=7, vocab_size_cond=1000, block_size_cond=1,
        body=dict(n_layer=1, block=dict(n_head=3, resid_pdrop=0.)),
        head=dict(n_layer=1, block=dict(n_head=3, resid_pdrop=0.))))
    model = recipe.build_model(arch, num_atoms=7, coefficient_bins=5, depth=4)
    aux = SimpleNamespace(encoder=lambda x: x, quant_conv=lambda x: x,
        dictionary=torch.nn.functional.normalize(torch.randn(4, 7), dim=0),
        coeff_bins=torch.linspace(-3, 3, 5), coeff_scales=torch.tensor([3., 2., 1., .5]),
        num_atoms=7, coeff_vocab_size=5, vocab_size=12, sparsity_level=4,
        clamp_coeffs=False, soft_target_physical=False)
    aux.sparse_targets = MethodType(FrozenLaserTokenizer.sparse_targets, aux)
    teacher = OmegaConf.create(dict(encoder_batch_size=1, atom_temperature=.4,
        coefficient_temperature=.5, variants=4, site_chunk_size=3))
    return model, aux, torch.randn(2, 4, 2, 2), torch.tensor([0, 999]), teacher


def test_online_teacher_and_single_ce_match_dense_token_likelihood():
    model, aux, images, labels, teacher = example()
    rng = torch.get_rng_state()
    targets = recipe.image_targets(aux, images, teacher)
    recipe.verify_targets(aux, targets)
    torch.set_rng_state(rng)
    replay = recipe.image_targets(aux, images, teacher)
    for key in targets:
        torch.testing.assert_close(targets[key], replay[key], rtol=0, atol=0)
    logits = model(targets["tokens"], aux, labels)
    actual = recipe.token_cross_entropy(logits, targets)
    dense = torch.zeros_like(logits["atom_logits"]).scatter_add(
        -1, targets["atom_ids"], targets["atom_weights"])
    log_atoms = logits["atom_logits"].log_softmax(-1)
    safe = torch.where(dense > 0, log_atoms, 0.)
    atom_ce = -(dense * safe).sum(-1)
    coefficient_ce = -(targets["coefficient_probabilities"]
                       * logits["coeff_logits"].log_softmax(-1)).sum(-1)
    expected = (atom_ce + coefficient_ce).mean() / 2
    torch.testing.assert_close(actual, expected)
    actual.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    assert not any("history" in name or "geometry" in name for name, _ in model.named_parameters())


def test_optimizer_checkpoint_replays_next_update_and_preserves_pair_sampler(tmp_path):
    model, aux, images, labels, teacher = example()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005, betas=(.9, .95), weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=62600)

    def update(prior, optim, schedule):
        optim.zero_grad(set_to_none=True)
        target = recipe.image_targets(aux, images, teacher)
        loss = recipe.token_cross_entropy(prior(target["tokens"], aux, labels), target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prior.parameters(), 1., error_if_nonfinite=True)
        optim.step(); schedule.step()
        return loss.detach()

    update(model, optimizer, scheduler)
    checkpoint = tmp_path / "state.pt"
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(), rng=torch.get_rng_state()), checkpoint)
    expected = update(model, optimizer, scheduler)
    restored, _, _, _, _ = example()
    optim = torch.optim.AdamW(restored.parameters())
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=62600)
    saved = torch.load(checkpoint, weights_only=True)
    restored.load_state_dict(saved["model"], strict=True)
    optim.load_state_dict(saved["optimizer"]); schedule.load_state_dict(saved["scheduler"])
    torch.set_rng_state(saved["rng"])
    torch.testing.assert_close(update(restored, optim, schedule), expected, rtol=0, atol=0)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    assert len(optim.state) == len(list(restored.parameters()))
    assert all(int(value["step"]) == 2 for value in optim.state.values())
    restored.eval()
    tokens = restored.sample_sparse(2, aux, cond=labels, amp=False,
        atom_top_k=7, atom_top_p=.95, coeff_top_p=.95)
    assert (tokens[..., 0::2].sort(-1).values.diff(dim=-1) > 0).all()
    assert ((tokens[..., 1::2] >= 7) & (tokens[..., 1::2] < 12)).all()
