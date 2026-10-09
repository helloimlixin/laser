from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from src.data.seeded_bpe import SeededBPE
from scripts.tools.build_cc3m_compound_cache import text_tokenizer
from src.models.physical_pair_scalar_prior import PhysicalPairScalarRQTransformer
from src.models.rqtransformer.configs import RQTransformerConfig
from src.training.cc3m_text import (cpu_snapshot, preview_image, config_digest,
                                    create_lr_scheduler, verify_resume_config, verify_resume_metrics)
from src.training.cc3m_text import verify_checkpoint_progress


def case():
    config = RQTransformerConfig.create(OmegaConf.create({
        'type': 'rq-transformer', 'block_size': [2, 2, 4], 'embed_dim': 12,
        'input_embed_dim': 4, 'shared_tok_emb': True, 'shared_cls_emb': True,
        'input_emb_vqvae': True, 'head_emb_vqvae': True, 'cumsum_depth_ctx': True,
        'vocab_size': 12, 'vocab_size_cond': 13, 'block_size_cond': 3,
        'body': {'n_layer': 1, 'block': {'n_head': 3, 'resid_pdrop': 0.}},
        'head': {'n_layer': 1, 'block': {'n_head': 3, 'resid_pdrop': 0.}},
    }))
    model = PhysicalPairScalarRQTransformer(config, 7).eval()
    aux = SimpleNamespace(dictionary=torch.randn(4, 7), coeff_bins=torch.linspace(-1, 1, 5),
        coeff_scales=torch.tensor([1., 2.]), num_atoms=7, coeff_vocab_size=5, sparsity_level=2)
    tokens = torch.tensor([0, 8, 1, 9]).reshape(1, 1, 1, 4).expand(1, 2, 2, 4).clone()
    return model, aux, tokens, torch.tensor([[3, 5, 8]])


def test_text_prefix_is_causal_and_cached_matches_teacher_forcing():
    torch.manual_seed(9)
    model, aux, tokens, text = case()
    outputs, text_logits = model(tokens, aux, text)
    changed = text.clone()
    changed[:, -1] = 7
    other, other_text_logits = model(tokens, aux, changed)
    torch.testing.assert_close(text_logits, other_text_logits, rtol=0, atol=0)
    assert not torch.equal(outputs['coeff_logits'], other['coeff_logits'])
    generated = torch.zeros_like(tokens)
    generated[..., 1::2] = 9
    model.init_cache()
    for h in range(2):
        for w in range(2):
            for d in range(4):
                logits = model.cached_forward(generated[:, :h+1], aux, text, sample_loc=(h, w, d))
                actual = logits[:, :7] if d % 2 == 0 else logits[:, 7:]
                expected = outputs['atom_logits' if d % 2 == 0 else 'coeff_logits'][:, h, w, d//2]
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=5e-6)
                generated[:, h, w, d] = tokens[:, h, w, d]


def test_dropout_tokens_reproduce_every_future_epoch_and_match_released_vocabulary():
    captions = ['Café, DOGS! 你好', 'a red cat [PAD]', '[UNK] is here',
        'A photograph of a small wooden house beside a lake in autumn.']
    official = text_tokenizer(0.)
    no_dropout = SeededBPE(0.)
    assert [no_dropout.encode(c, 7).ids for c in captions] == [official.encode(c).ids for c in captions]
    first, resumed = SeededBPE(.1), SeededBPE(.1)
    for epoch in (0, 1, 45, 99):
        kwargs = dict(indices=[3, 19, 500, 788], epoch=epoch, seed=42)
        assert [r.ids for r in first.encode_batch(captions, **kwargs)] == [r.ids for r in resumed.encode_batch(captions, **kwargs)]
    variants = {tuple(first.encode(captions[-1], i).ids) for i in range(50)}
    assert len(variants) > 1


def test_snapshot_retains_adam_and_scheduler_without_live_tensor_aliases():
    torch.manual_seed(55)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    x = torch.randn(4, 3)
    model(x).square().mean().backward()
    optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
    snapshot = cpu_snapshot(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict()))
    restored = torch.nn.Linear(3, 2)
    restored.load_state_dict(snapshot['model'])
    opt2 = torch.optim.AdamW(restored.parameters(), lr=.005)
    opt2.load_state_dict(snapshot['optimizer'])
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=10)
    sch2.load_state_dict(snapshot['scheduler'])
    for m, opt, sch in [(model, optimizer, scheduler), (restored, opt2, sch2)]:
        m(x).square().mean().backward(); opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
    for left, right in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert scheduler.state_dict() == sch2.state_dict()
    assert not torch.equal(snapshot['model']['weight'], model.weight)


def test_wandb_preview_preserves_full_pixel_range():
    import numpy as np
    import wandb
    pixels = torch.tensor([0., .5, 1.]).reshape(1, 1, 3).expand(3, 2, 3)
    logged = wandb.Image(preview_image(pixels))
    actual = np.asarray(logged.image)
    expected = (pixels * 255).byte().permute(1, 2, 0).numpy()
    np.testing.assert_array_equal(actual, expected)
    assert actual.max() == 255


def test_fid_schedule_migration_preserves_adam_and_resumes_plateau_decisions():
    import copy
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005)
    original = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    for _ in range(9):
        model(torch.ones(2, 3)).square().mean().backward()
        optimizer.step(); original.step(); optimizer.zero_grad(set_to_none=True)
    adam = cpu_snapshot(optimizer.state_dict()['state'])
    source = dict(lr=.0005, min_lr=0.)
    options = dict(epochs=10, lr=.00025, min_lr=.00002,
        lr_schedule='fid_adaptive_cosine', fid_lr_policy=dict(
            baseline_fid=27.53, patience=3, min_delta=.1, factor=.5, cooldown=1),
        lr_schedule_migration=dict(source_step=9, source_config_sha256=config_digest(source)))
    scheduler = create_lr_scheduler(optimizer, options, 10, 9, original.state_dict(), source)
    assert optimizer.param_groups[0]['lr'] < .00025
    for key, state in optimizer.state_dict()['state'].items():
        for name, value in state.items():
            torch.testing.assert_close(value, adam[key][name], rtol=0, atol=0)
    scheduler.step(); assert scheduler.observe(28.)['decision'] == 'watch'
    checkpoint = cpu_snapshot(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                  scheduler=scheduler.state_dict()))
    clone = copy.deepcopy(model)
    resumed = torch.optim.AdamW(clone.parameters(), lr=options['lr'])
    resumed.load_state_dict(checkpoint['optimizer'])
    restored = create_lr_scheduler(resumed, options, 10, 10, checkpoint['scheduler'], options)
    for fid in (29., 28., 27., 26.):
        for m, opt, sch in ((model, optimizer, scheduler), (clone, resumed, restored)):
            m(torch.ones(2, 3)).square().mean().backward()
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        assert scheduler.observe(fid) == restored.observe(fid)
        assert scheduler.state_dict() == restored.state_dict()
        for left, right in zip(model.parameters(), clone.parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert scheduler.reductions == 1


def test_schedule_change_needs_verified_source_and_rejects_wrong_optimizer_lr():
    import copy
    import pytest
    source = dict(batch_size=128, accumulation=2, total_batch_size=2048, epochs=10,
        lr=.0005, min_lr=0., seed=42, coeff_scales=[1.], coeff_target_temperature=.01,
        cache_sha256={'train': 'verified'}, runtime_sha256={'trainer': 'original'})
    target = copy.deepcopy(source)
    target.update(lr=.00025, lr_schedule='fid_adaptive_cosine', min_lr=.00002,
        runtime_sha256={'trainer': 'updated'}, fid_lr_policy=dict(
            baseline_fid=27.53, patience=3, min_delta=.1, factor=.5, cooldown=1))
    with pytest.raises(ValueError, match='verified schedule migration'):
        verify_resume_config(source, target)
    target['lr_schedule_migration'] = dict(source_config_sha256=config_digest(source), source_step=9)
    verify_resume_config(source, target)
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=.0005)
    original = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    for _ in range(9):
        optimizer.step(); original.step()
    optimizer.param_groups[0]['lr'] *= .9
    with pytest.raises(ValueError, match='optimizer and cosine scheduler LR disagree'):
        create_lr_scheduler(optimizer, target, 10, 9, original.state_dict(), source)


def test_lower_adaptive_lr_preserves_optimizer_and_controller_and_resumes_exactly():
    import copy
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.00025)
    source = dict(epochs=10, lr=.00025, min_lr=.00002,
        lr_schedule='fid_adaptive_cosine', fid_lr_policy=dict(
            baseline_fid=27.53, patience=1, min_delta=.1, factor=.5, cooldown=1))
    scheduler = create_lr_scheduler(optimizer, source, 10)
    for _ in range(10):
        model(torch.ones(2, 3)).square().mean().backward()
        optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
    scheduler.observe(28.)
    state = copy.deepcopy(scheduler.state_dict())
    adam = cpu_snapshot(optimizer.state_dict()['state'])
    target = copy.deepcopy(source)
    target.update(lr=.0001, min_lr=.00001, lr_schedule_migration=dict(
        source_config_sha256=config_digest(source), source_step=10,
        source_scheduler_sha256=config_digest(state)))
    target['fid_lr_policy']['patience'] = 2
    target['fid_lr_policy'].update(decay_start_step=10, decay_steps=20)
    lower = create_lr_scheduler(optimizer, target, 10, 10, state, source)
    assert lower.state_dict() == dict(state, policy=lower.policy)
    assert optimizer.param_groups[0]['lr'] < .0001
    for key, values in optimizer.state_dict()['state'].items():
        for name, value in values.items():
            torch.testing.assert_close(value, adam[key][name], rtol=0, atol=0)
    clone = copy.deepcopy(model)
    resumed = torch.optim.AdamW(clone.parameters(), lr=target['lr'])
    resumed.load_state_dict(cpu_snapshot(optimizer.state_dict()))
    restored = create_lr_scheduler(resumed, target, 10, 10, lower.state_dict(), target)
    for fid in (27., 28., 29.):
        for m, opt, sch in ((model, optimizer, lower), (clone, resumed, restored)):
            m(torch.ones(2, 3)).square().mean().backward()
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        assert lower.observe(fid) == restored.observe(fid)
        for left, right in zip(model.parameters(), clone.parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_adaptive_resume_rejects_inconsistent_optimizer_and_unverified_policy_change():
    import copy
    import pytest
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=.00025)
    source = dict(epochs=10, lr=.00025, min_lr=.00002,
        lr_schedule='fid_adaptive_cosine', fid_lr_policy=dict(
            baseline_fid=27.53, patience=3, min_delta=.1, factor=.5, cooldown=1))
    scheduler = create_lr_scheduler(optimizer, source, 10)
    optimizer.step(); scheduler.step()
    state = copy.deepcopy(scheduler.state_dict())
    original_lr = optimizer.param_groups[0]['lr']
    optimizer.param_groups[0]['lr'] *= .9
    with pytest.raises(ValueError, match='optimizer and adaptive scheduler LR disagree'):
        create_lr_scheduler(optimizer, source, 10, 1, state, source)
    optimizer.param_groups[0]['lr'] = original_lr
    target = dict(source, lr=.0001, min_lr=.00001)
    with pytest.raises(ValueError, match='verified source adaptive checkpoint'):
        create_lr_scheduler(optimizer, target, 10, 1, state, source)
    target['lr_schedule_migration'] = dict(source_config_sha256=config_digest(source),
        source_step=1, source_scheduler_sha256='incorrect')
    with pytest.raises(ValueError, match='verified source adaptive checkpoint'):
        create_lr_scheduler(optimizer, target, 10, 1, state, source)


def test_extend_training_horizon_preserves_lr_adam_and_fid_controller():
    import copy
    import pytest
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0002)
    source = dict(epochs=40, lr=.0002, min_lr=.00001,
        lr_schedule='fid_adaptive_cosine', warmup_epochs=1,
        fid_lr_policy=dict(baseline_fid=None, patience=3, min_delta=.1,
            factor=.5, cooldown=2, warmup_steps=10, decay_start_step=10, decay_steps=390))
    scheduler = create_lr_scheduler(optimizer, source, 10)
    for _ in range(100):
        model(torch.ones(2, 3)).square().mean().backward()
        optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
    scheduler.observe(28.2)
    state = copy.deepcopy(scheduler.state_dict())
    before = optimizer.param_groups[0]['lr']
    adam = cpu_snapshot(optimizer.state_dict()['state'])
    target = dict(source, epochs=100, lr=before, warmup_epochs=0,
        fid_lr_policy=dict(baseline_fid=None, patience=3, min_delta=.1,
            factor=.5, cooldown=2, decay_start_step=100, decay_steps=900),
        lr_schedule_migration=dict(source_config_sha256=config_digest(source),
            source_step=100, source_scheduler_sha256=config_digest(state)))
    extended = create_lr_scheduler(optimizer, target, 10, 100, state, source)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(before, abs=1e-16)
    assert extended.best == scheduler.best and extended.reductions == scheduler.reductions
    assert extended.last_observation_step == scheduler.last_observation_step
    for key, values in optimizer.state_dict()['state'].items():
        for name, value in values.items():
            torch.testing.assert_close(value, adam[key][name], rtol=0, atol=0)
    clone = copy.deepcopy(model)
    resumed = torch.optim.AdamW(clone.parameters(), lr=target['lr'])
    resumed.load_state_dict(cpu_snapshot(optimizer.state_dict()))
    restored = create_lr_scheduler(resumed, target, 10, 100, extended.state_dict(), target)
    for fid in [29., 30., 31., 32.]:
        for m, opt, sch in [(model, optimizer, extended), (clone, resumed, restored)]:
            m(torch.ones(2, 3)).square().mean().backward()
            opt.step();sch.step();opt.zero_grad(set_to_none=True)
        assert extended.observe(fid) == restored.observe(fid)
        assert extended.state_dict() == restored.state_dict()
        for left, right in zip(model.parameters(), clone.parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    for _ in range(1000-extended.last_epoch):extended.step()
    assert optimizer.param_groups[0]['lr'] == target['min_lr']


def test_epoch_extension_requires_verified_source():
    import copy
    import pytest
    source = dict(batch_size=128, accumulation=2, total_batch_size=2048, epochs=40,
        lr=.0002, min_lr=.00001, seed=42, coeff_scales=[1.],
        coeff_target_temperature=.01, cache_sha256={'train':'verified'},
        runtime_sha256={'trainer':'old'})
    target = copy.deepcopy(source)
    target['epochs'] = 100
    with pytest.raises(ValueError, match='verified schedule extension'):
        verify_resume_config(source, target)
    target['lr_schedule_migration'] = dict(source_config_sha256=config_digest(source))
    verify_resume_config(source, target)
    target['epochs'] = 20
    with pytest.raises(ValueError, match='verified schedule extension'):
        verify_resume_config(source, target)


def test_resume_evaluation_rejects_a_different_checkpoint_or_protocol():
    import pytest
    expected = dict(fid=25.615, clip_score=.23522, items=13443)
    verify_resume_metrics(dict(expected, seconds=65.), expected)
    for key, value in [('fid', 25.616), ('clip_score', .23523), ('items', 13442)]:
        with pytest.raises(ValueError, match='did not reproduce'):
            verify_resume_metrics(dict(expected, **{key: value}), expected)


def test_resume_rejects_mixed_optimizer_scheduler_and_data_cursor():
    import copy
    import pytest
    state = dict(global_step=12, epoch=1, next_microbatch=4, resume_capable=True,
        world_size=2, rng_state_by_rank=[{}, {}], scheduler=dict(last_epoch=12),
        optimizer=dict(state={0: dict(step=torch.tensor(12.))}))
    verify_checkpoint_progress(state, 10, 2, 2)
    bad = copy.deepcopy(state)
    bad['optimizer']['state'][0]['step'] = torch.tensor(11.)
    with pytest.raises(ValueError, match='disagree'):
        verify_checkpoint_progress(bad, 10, 2, 2)
    for changes in (dict(next_microbatch=6), dict(scheduler=dict(last_epoch=11)),
                    dict(rng_state_by_rank=[{}])):
        with pytest.raises(ValueError, match='disagree'):
            verify_checkpoint_progress(dict(state, **changes), 10, 2, 2)


def test_restart_prefers_new_local_commit_and_ignores_other_continuations(tmp_path):
    from scripts.tools.resume_cc3m_text import newest_matching_checkpoint
    options = dict(runtime_sha256={'trainer':'current'}, fid_lr_policy={'patience':2})
    paths = [tmp_path/'local.pt', tmp_path/'persistent.pt']
    def save(path, step, config):
        torch.save(dict(resume_capable=True, config=config, global_step=step,
            scheduler=dict(kind='fid-adaptive-cosine-v1',last_epoch=step)), path)
    save(paths[0], 30, options);save(paths[1], 20, options)
    assert newest_matching_checkpoint(options, paths)==paths[0]
    save(paths[0], 40, dict(options, runtime_sha256={'trainer':'different'}))
    assert newest_matching_checkpoint(options, paths)==paths[1]
    paths[0].write_text('interrupted write')
    assert newest_matching_checkpoint(options, paths)==paths[1]
def test_fresh_adaptive_warmup_factory_preserves_adam_and_controller_after_resume():
    import copy
    options = dict(epochs=4, lr=.0002, min_lr=.00001, warmup_epochs=1,
        lr_schedule='fid_adaptive_cosine', fid_lr_policy=dict(
            baseline_fid=None, patience=3, min_delta=.1, factor=.5, cooldown=2,
            warmup_steps=4, decay_start_step=4, decay_steps=12))
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=options['lr'])
    schedule = create_lr_scheduler(optimizer, options, 4)
    x = torch.ones(2, 3)
    def update(m, opt, scheduler):
        m(x).square().mean().backward()
        opt.step(); scheduler.step(); opt.zero_grad(set_to_none=True)
    update(model, optimizer, schedule); schedule.observe(190.)
    clone = copy.deepcopy(model)
    restored_optimizer = torch.optim.AdamW(clone.parameters(), lr=options['lr'])
    restored_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    restored_schedule = create_lr_scheduler(restored_optimizer, options, 4,
        completed_steps=1, state_dict=schedule.state_dict(), saved_config=options)
    for fid in (80., 40., 41., 42., 43., 40., 39.):
        for m, opt, scheduler in ((model, optimizer, schedule),
                                  (clone, restored_optimizer, restored_schedule)):
            update(m, opt, scheduler)
        assert schedule.observe(fid) == restored_schedule.observe(fid)
        assert schedule.state_dict() == restored_schedule.state_dict()
        for a, b in zip(model.parameters(), clone.parameters()):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
