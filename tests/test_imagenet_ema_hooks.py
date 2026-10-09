import json
from contextlib import contextmanager
from types import SimpleNamespace

import torch

from scripts.tools.imagenet_ema_hooks import install


def test_paired_evaluation_preserves_raw_adam_and_training_rng(tmp_path, monkeypatch):
    base, evidence = tmp_path / 'base', tmp_path / 'evidence'
    base.mkdir(); evidence.mkdir()
    checkpoint_dir = evidence / 'continuation-20261005/train/checkpoints'
    checkpoint_dir.mkdir(parents=True)
    initial = {kind: dict(score=9999 if kind == 'fid' else -9999,
        path=str(checkpoint_dir / f'initial-{kind}.pt'), global_step=0)
        for kind in ('fid', 'is')}
    (base / 'ema-trial.json').write_text(json.dumps(dict(source_step=0, decay=.999, initial_ema_best=initial)))
    model = torch.nn.Linear(3, 2, dtype=torch.float64)
    opt = torch.optim.AdamW(model.parameters(), lr=.01)
    recorded, draws = {}, []
    rng = torch.random.fork_rng
    monkeypatch.setattr(torch, 'load', torch.load)
    monkeypatch.setattr(torch.cuda, 'get_rng_state', lambda *_: torch.zeros(3, dtype=torch.uint8))
    monkeypatch.setattr(torch.cuda, 'manual_seed', lambda *_: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda *_: 0)
    monkeypatch.setattr(torch.random, 'fork_rng', lambda devices: rng(devices=[]))
    monkeypatch.setattr('scripts.tools.imagenet_ema_hooks.dist.get_rank', lambda: 0)
    monkeypatch.setattr(torch.optim.AdamW, 'step', torch.optim.AdamW.step)
    adam_step = torch.optim.AdamW.step
    ns = dict(BASE=base, EVIDENCE=evidence, VERIFY=evidence / 'verification',
        ARGS=None, INITIAL_RECOVERY={'global_step':0}, UPDATES=0, UPLOADER=None, WB=None,
        os=SimpleNamespace(environ={'RANK':'0'}), record=lambda p, v: recorded.update({p.name:v}))

    def underlying(model, *args, **kwargs):
        draws.append(torch.rand(3))
        value = float(next(model.parameters()).detach().square().sum())
        return value, value + 1, .1

    def raw_evaluate(model, *args, **kwargs):
        with rng(devices=[]):
            torch.random.default_generator.manual_seed(261001)
            result = underlying(model, *args, **kwargs)
        ns['OFFICIAL_METRICS'] = dict(global_step=ns['UPDATES'], fid=result[0])
        return result

    def underlying_step(optimizer, *args, **kwargs):
        result = adam_step(optimizer, *args, **kwargs); ns['UPDATES'] += 1
        return result

    torch.optim.AdamW.step = underlying_step
    training = SimpleNamespace(wrap_distributed_model=lambda m, *a, **k:m,
                               evaluate_generation_metrics=raw_evaluate,
                               atomic_torch_save=lambda *a, **k:None)
    ns.update(training=training, original_evaluate=underlying)
    install(ns)
    training.wrap_distributed_model(model)
    for _ in range(20):
        model(torch.ones(2, 3, dtype=torch.float64)).square().sum().backward()
        opt.step(); opt.zero_grad()
    raw = {k: p.detach().clone() for k, p in model.named_parameters()}
    moments = {p: {k: t.clone() for k, t in state.items()} for p, state in opt.state.items()}
    cpu_rng = torch.get_rng_state().clone()
    result = training.evaluate_generation_metrics(model, None, None, 50000, metric_backend='original-rqvae')
    assert torch.equal(draws[0], draws[1]) and torch.equal(cpu_rng, torch.get_rng_state())
    assert all(torch.equal(p, raw[k]) for k, p in model.named_parameters())
    for p, fields in moments.items():
        assert all(torch.equal(opt.state[p][k], t) for k, t in fields.items())
    assert recorded['ema-step20-rank0.json']['updates'] == 20
    paired = recorded['paired-evaluation-step20-rank0.json']
    assert paired['ema']['weight_state'] == 'ema'
    assert paired['raw']['fid'] == result[0] and paired['ema']['fid'] != result[0]
