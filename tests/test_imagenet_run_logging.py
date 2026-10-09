import ast
from pathlib import Path


def test_gaussian_run_logs_clean_error_and_refreshes_scalar_summary_after_resume():
    # Load only the logging adapter; importing the executable entry point would
    # initialize CUDA, distributed training, and the online W&B run.
    path = Path(__file__).parents[1]/'scripts/tools/imagenet_repaired_scratch_entry.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LoggedRun')
    diagnostics = {'train/coeff_mode_mae_bins': 96.8, 'train/coeff_cross_entropy_nats': 5.4}
    namespace = {'DIAGNOSTIC_METRICS': diagnostics}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), 'exec'), namespace)

    class Run:
        def __init__(self):
            self.summary = {'diagnostics/coeff_mode_mae_last_bins': 334.,
                            'diagnostics/coeff_mode_mae_best_bins': 330.}
            self.history = []

        def log(self, data, *args, **kwargs):
            self.history.append(data)

    run = Run()
    namespace['LoggedRun'](run).log({'train/loss': 7., 'train/global_step': 30})
    assert run.history[-1]['train/coeff_mode_mae_bins'] == 96.8
    assert run.summary['diagnostics/coeff_mode_mae_last_bins'] == 96.8
    assert run.summary['diagnostics/coeff_mode_mae_best_bins'] == 96.8
    assert run.summary['diagnostics/coeff_mode_mae_global_step'] == 30
    assert not any(key.startswith('evaluation/') for key in run.summary)

    # A process restart restores the scalar best, then a noisier batch changes
    # the last value without losing the best or retaining stale preflight MAE.
    diagnostics['train/coeff_mode_mae_bins'] = 100.
    namespace['LoggedRun'](run).log({'train/loss': 7., 'train/global_step': 40})
    assert run.history[-1]['train/coeff_mode_mae_bins'] == 100.
    assert run.summary['diagnostics/coeff_mode_mae_last_bins'] == 100.
    assert run.summary['diagnostics/coeff_mode_mae_best_bins'] == 96.8
    assert run.summary['diagnostics/coeff_mode_mae_global_step'] == 40
