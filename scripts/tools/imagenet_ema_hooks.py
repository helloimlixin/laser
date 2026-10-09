"""Attach resumable weight EMA and paired official evaluations to the frozen trainer."""
import copy
import json
from pathlib import Path

import torch
import torch.distributed as dist

from src.training.parameter_ema import ParameterEMA


def install(ns):
    training = ns['training']
    base, evidence, verify = ns['BASE'], ns['EVIDENCE'], ns['VERIFY']
    record = ns['record']
    control = json.loads((base / 'ema-trial.json').read_text())
    checkpoints = evidence / 'continuation-20261005/train/checkpoints'
    ctx = dict(model=None, ema=None, saved=None, origin=control['source_step'],
               metrics=None, best=copy.deepcopy(control['initial_ema_best']))

    original_load = torch.load

    def load(path, *args, **kwargs):
        payload = original_load(path, *args, **kwargs)
        args_cfg = ns['ARGS']
        if args_cfg is not None and args_cfg.resume and isinstance(path, (str, Path)) \
                and Path(path).resolve() == args_cfg.resume_checkpoint.resolve():
            ctx['saved'] = payload.get('parameter_ema')
            if ctx['saved'] is not None:
                ctx['origin'] = ctx['saved']['origin_global_step']
                assert ctx['origin'] == control['source_step']
                assert ctx['saved']['updates'] == payload['global_step'] - ctx['origin']
            ctx['best'] = copy.deepcopy(payload.get('ema_best', ctx['best']))
            for winner in ctx['best'].values():
                winner['path'] = str(checkpoints / Path(winner['path']).name)
                assert Path(winner['path']).is_file()
        return payload

    torch.load = load
    original_wrap = training.wrap_distributed_model

    def wrap(model, *args, **kwargs):
        wrapped = original_wrap(model, *args, **kwargs)
        ctx['model'] = model
        ctx['ema'] = ParameterEMA(model, control['decay'], ctx['saved'])
        ctx['saved'] = None
        initial = ctx['ema'].updates == 0
        assert ctx['ema'].updates == ns['INITIAL_RECOVERY']['global_step'] - ctx['origin']
        if initial:
            assert all(torch.equal(ctx['ema'].values[k], p) for k, p in model.named_parameters())
        record(verify / ('ema-startup-rank' + ns['os'].environ['RANK'] + '.json'),
               dict(decay=ctx['ema'].decay, updates=ctx['ema'].updates,
                    origin_global_step=ctx['origin'], initialized_from_raw_exactly=initial,
                    parameter_tensors=len(ctx['ema'].values),
                    parameter_elements=sum(p.numel() for p in ctx['ema'].values.values())))
        return wrapped

    training.wrap_distributed_model = wrap
    original_step = torch.optim.AdamW.step

    def step(optimizer, *args, **kwargs):
        # The underlying wrapper checks uploads and completes the real Adam step.
        result = original_step(optimizer, *args, **kwargs)
        ctx['ema'].update(ctx['model'])
        global_step = ns['INITIAL_RECOVERY']['global_step'] + ns['UPDATES']
        assert ctx['ema'].updates == global_step - ctx['origin']
        if ns['UPDATES'] in (1, 20):
            finite = bool(torch.stack([torch.isfinite(t).all() for t in ctx['ema'].values.values()]).all())
            assert finite
            record(verify / ('ema-step' + str(ns['UPDATES']) + '-rank' + ns['os'].environ['RANK'] + '.json'),
                   dict(global_step=global_step, updates=ctx['ema'].updates,
                        decay=ctx['ema'].decay, finite=True,
                        updated_after_adam=True,
                        cuda_peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30))
        return result

    torch.optim.AdamW.step = step
    original_evaluate = training.evaluate_generation_metrics
    upstream_evaluate = ns['original_evaluate']

    def evaluate(model, *args, **kwargs):
        assert kwargs['metric_backend'] == 'original-rqvae'
        assert args[2] == 50000
        assert model is ctx['model']
        device = next(model.parameters()).device
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state(device).clone()
        raw_result = original_evaluate(model, *args, **kwargs)
        # Both passes start from exactly the same evaluation RNG; neither
        # consumes the training streams or changes the live Adam parameters.
        with torch.random.fork_rng(devices=[device.index]):
            torch.random.default_generator.manual_seed(261001 + dist.get_rank())
            torch.cuda.manual_seed(261001 + dist.get_rank())
            with ctx['ema'].apply(model):
                ema_result = upstream_evaluate(model, *args, **kwargs)
        assert torch.equal(cpu_rng, torch.get_rng_state())
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
        global_step = ns['INITIAL_RECOVERY']['global_step'] + ns['UPDATES']
        ctx['metrics'] = dict(global_step=global_step, fid=ema_result[0],
            inception_score=ema_result[1], inception_score_std=ema_result[2],
            metric_backend='original_rqtransformer', weight_state='ema',
            real_images=50000, generated_images=50000, real_split='val',
            inception_splits=10, seed=261001, ema_updates=ctx['ema'].updates,
            ema_decay=ctx['ema'].decay)
        for kind, field, better in (('fid', 'fid', lambda a, b: a < b),
                                    ('is', 'inception_score', lambda a, b: a > b)):
            score = float(ctx['metrics'][field])
            if better(score, ctx['best'][kind]['score']):
                epoch = global_step // 626
                path = checkpoints / f'best_ema_{kind}_{score:.4f}_epoch_{epoch:03d}.pt'
                ctx['best'][kind] = dict(score=score, path=str(path), global_step=global_step,
                                         epoch=epoch, metrics=copy.deepcopy(ctx['metrics']))
        record(verify / ('paired-evaluation-step' + str(global_step) + '-rank' + ns['os'].environ['RANK'] + '.json'),
               dict(raw=ns['OFFICIAL_METRICS'], ema=ctx['metrics'],
                    training_rng_restored_exactly=True, raw_parameters_restored=True))
        if dist.get_rank() == 0:
            record(evidence / 'continuation-20261005' / f'official-ema-metrics-step{global_step}.json', ctx['metrics'])
            if ns['WB'] is not None:
                ns['WB'].log({'train/global_step': global_step, 'train/epoch': global_step // 626,
                    'eval/ema/fid_original_rqtransformer': ema_result[0],
                    'eval/ema/inception_score_original_rqtransformer': ema_result[1],
                    'eval/ema/inception_score_std_original_rqtransformer': ema_result[2],
                    'train/parameter_ema_updates': ctx['ema'].updates})
        return raw_result

    training.evaluate_generation_metrics = evaluate

    def submit_upload():
        if ns['UPLOADER'] is None:
            return
        sources = [('last.pt', checkpoints / 'last.pt')]
        for kind, ranking in (('fid', ns['BEST_FID']), ('is', ns['BEST_IS'])):
            if ranking:sources.append((f'best-raw-{kind}-resume.pt', Path(ranking[0][1])))
        sources.extend((f'best-ema-{kind}-resume.pt', Path(winner['path']))
                       for kind, winner in ctx['best'].items())
        if any(not p.is_file() for _, p in sources):
            return
        slots = base / 'upload-slots'; slots.mkdir(exist_ok=True)
        paths = []
        for name, source in sources:
            target = slots / name
            ns['checkpoint_io']._replace_hard_link(ns['checkpoint_io']._checkpoint_upload_source(source), target)
            paths.append(target)
        ns['UPLOADER'].submit(paths, ns['CHECKPOINT_EPOCH'])

    ns['submit_upload'] = submit_upload
    original_save = training.atomic_torch_save

    def save(payload, target):
        global_step = int(payload['global_step'])
        ema = ctx['ema']
        assert ema.updates == global_step - ctx['origin']
        state = ema.state_dict(cpu=False)
        state['origin_global_step'] = ctx['origin']
        payload = dict(payload, parameter_ema=state, ema_best=copy.deepcopy(ctx['best']),
                       training_state_weights='raw', optimizer_state_weights='raw')
        if ctx['metrics'] is not None and ctx['metrics']['global_step'] == global_step:
            payload['ema_original_rqtransformer_metrics'] = copy.deepcopy(ctx['metrics'])
        new_aliases = [Path(w['path']) for w in ctx['best'].values() if w['global_step'] == global_step]
        result = original_save(payload, target)
        if new_aliases:
            def commit_aliases():
                immutable = Path(target).resolve(strict=True)
                for alias in new_aliases:
                    temporary = alias.with_suffix('.alias.tmp'); temporary.unlink(missing_ok=True)
                    temporary.symlink_to(immutable.relative_to(alias.parent)); temporary.replace(alias)
                # Retain one immutable full checkpoint per independent EMA winner.
                keep = {Path(w['path']) for w in ctx['best'].values()}
                for old in checkpoints.glob('best_ema_*.pt'):
                    if old not in keep:ns['checkpoint_io'].remove_checkpoint(old)
                submit_upload()
            ns['WRITER'].append(commit_aliases)
        return result

    training.atomic_torch_save = save
