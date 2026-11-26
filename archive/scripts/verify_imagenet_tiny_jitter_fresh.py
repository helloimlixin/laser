"""CPU preflight of frozen launch code, target probabilities, and W&B logging."""
import importlib.util
import json
import os
from pathlib import Path
import py_compile
import tempfile

import torch
import yaml

from launch_imagenet_tiny_jitter_fresh import BASE, OUT, ROOT, write


def main():
    torch.set_num_threads(4)
    os.environ.update(LASER_PAIR_MEMORY_BASE=str(BASE), LASER_PAIR_MEMORY_OUTPUT=str(OUT),
                      RANK='0', WANDB_SILENT='true', WANDB_CONSOLE='off')
    for path in (BASE/'entry.py', BASE/'launch.py', BASE/'source/src/training/coefficient_jitter.py',
                 BASE/'source/src/training/generation_logging.py'):
        py_compile.compile(str(path), doraise=True)
    spec = importlib.util.spec_from_file_location('frozen_tiny_entry', BASE/'entry.py')
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    captured = {}
    entry.record = lambda name, value: captured.update({name:value})
    from src.training.options import options_to_argv
    from src.training.coefficient_jitter import jitter_probabilities
    from src.training.generation_logging import define_generation_metrics
    config = yaml.safe_load((BASE/'train.yaml').read_text())['options']
    args = entry.training.build_parser().parse_args(options_to_argv(config))
    assert not args.resume and args.resume_checkpoint is None and args.init_stage2_checkpoint is None
    assert args.epochs == args.lr_schedule_epochs == 100 and args.lr == 0.0005
    assert args.coefficient_noise_sigma_bins == 0.01 and args.coefficient_noise_cap_bins == 0.025
    assert args.fid_num_samples == 50000 and args.fid_every == 2 and args.fid_seed == 261001
    aux = entry.training.LaserAux.__new__(entry.training.LaserAux)
    torch.nn.Module.__init__(aux)
    aux.sparsity_level = 4
    aux.coeff_vocab_size = 2048
    aux.soft_target_physical = False
    bins = torch.linspace(-3., 3., 2048)
    aux.register_buffer('coeff_bins',bins)
    aux.register_buffer('coeff_scales',torch.tensor(config['coeff_scales']))
    mids = (bins.double()[:-1]+bins.double()[1:])/2
    width = 6/2047
    cases = torch.cat([bins[::17].double(), mids[::17], mids[::17]+0.026*width,
                       mids[::17]-0.026*width,torch.tensor([-3.,3.],dtype=torch.float64)])
    cases = cases[:(len(cases)//4)*4].reshape(-1,4).float()
    ids, p = aux.compound_coeff_ids(cases, temp=args.coeff_target_temperature)
    nearest, expected = jitter_probabilities(cases,bins)
    torch.testing.assert_close(p,expected,rtol=0,atol=0)
    assert torch.equal(nearest,(cases[...,None]-bins).abs().argmin(-1))
    assert torch.allclose(p.sum(-1),torch.ones_like(cases),atol=1e-7)
    assert (p>=0).all() and (p>0).sum(-1).max()<=2
    assert (ids>=0).all() and (ids<2048).all()
    assert (p.gather(-1,ids[...,None])>0).all()
    hard_ids, hard_p = aux.compound_coeff_ids(cases,stochastic=False,hard=True)
    assert torch.equal(hard_ids,nearest) and (hard_p.max(-1).values==1).all()
    assert captured['target-noise-rank0.json']['passed']
    # Exercise the actual entry's logging adapter using real offline W&B SDK.
    # Synthetic values stay in /tmp, in this offline test run only.
    with tempfile.TemporaryDirectory(prefix='laser-metric-preflight-') as folder:
        run = entry.native_wandb_init(project='laser-preflight',mode='offline',dir=folder,
                                     settings=entry.wandb.Settings(console='off'))
        define_generation_metrics(run)
        old_out = entry.OUT
        entry.OUT = Path(folder)
        logged = entry.LoggedRun(run)
        for epoch, step, fid, inception in [(1,626,100.,10.),(2,1252,90.,12.)]:
            logged.log({'train/epoch':epoch,'train/global_step':step,
                'val/fid':fid,'val/inception_score':inception,'val/inception_score_std':0.25})
        assert run.summary['evaluation/best_fid']==90.
        assert run.summary['evaluation/best_inception_score']==12.
        assert run.summary['evaluation/last_global_step']==1252
        payload = json.loads((Path(folder)/'train/evaluations/generation_step_0001252.json').read_text())
        assert payload['eval/fid']==payload['val/fid']==90.
        assert payload['eval/inception_score']==payload['val/inception_score']==12.
        assert captured['metric-logging.json']['passed']
        run.finish()
        entry.OUT = old_out
    write(OUT/'preflight-verification.json',dict(passed=True,
        frozen_entry_import_and_parser=True,fresh_stage2_no_resume=True,
        native_nearest_bin_parity=True,boundaries_centers_edges_tested=True,
        exact_jitter_probability_parity=True,hard_mode_and_probability_normalization=True,
        actual_wandb_sdk_offline_metric_adapter=True,metric_axes_and_best_summaries=True,
        durable_evaluation_json=True,synthetic_metrics_never_sent_online=True,
        calibration_passed=json.loads((OUT/'noise-calibration.json').read_text())['passed']))
    print((OUT/'preflight-verification.json').read_text())


if __name__ == '__main__':
    main()
