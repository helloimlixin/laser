import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.training import original_generation_logging as logging


def test_trainer_logs_one_fid_and_matching_is_without_validation_aliases(tmp_path):
    path = Path(__file__).parents[1]/'scripts/tools/imagenet_repaired_scratch_entry.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LoggedRun')
    history, files, definitions = [], [], []
    run = SimpleNamespace(summary={}, log=lambda row:history.append(row),
                          save=lambda *args,**kwargs:files.append(args),
                          define_metric=lambda *args,**kwargs:definitions.append(args))
    namespace = {'DIAGNOSTIC_METRICS':{}, 'original_logging':logging, 'OUT':tmp_path}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),namespace)
    proxy = namespace['LoggedRun'](run)
    proxy.define_metric('val/fid')
    proxy.define_metric('val/inception_score')
    proxy.define_metric('val/inception_score_std')
    proxy.log({'val/fid':93.48,'val/inception_score':12.07,'val/inception_score_std':.24,
               'train/global_step':1252,'train/epoch':2})
    assert definitions == []
    assert history == [{'train/global_step':1252,'train/epoch':2,
                        logging.FID_KEY:93.48,logging.IS_KEY:12.07}]
    assert run.summary == {logging.FID_KEY:93.48,logging.IS_KEY:12.07}
    assert len(files) == 1
    import json
    result = json.loads(next((tmp_path/'train/evaluations').glob('*.json')).read_text())
    assert result['inception_score_std'] == .24 and result['generated_images'] == 50000


def test_protocol_change_rebases_ranking_without_touching_training_state():
    state = object()
    payload = {'config':{'checkpoint_fid_metric':'torchmetrics_validation50k'},
               'fid':89.38,'best_fid':[(89.38,'best.pt')],
               'best_inception':[(12.07,'is.pt')], 'optimizer':state,'state_dict':state,
               'scheduler':state,'rng_state_by_rank':state,'global_step':1500}
    previous = [{'previous_selected_fid':89.38,'original_train_fid':93.48}]
    result = logging.rebase_checkpoint_fids(payload,previous)
    assert result['fid'] == 93.48 and result['best_fid'] == [(93.48,'best.pt')]
    assert result['best_inception'] == [(12.07,'is.pt')]
    for key in ('optimizer','state_dict','scheduler','rng_state_by_rank'):
        assert result[key] is state
    assert result['global_step'] == 1500
    assert logging.rebase_checkpoint_fids(result,previous) == result


def test_protocol_change_requires_recorded_corresponding_fid():
    payload = {'config':{'checkpoint_fid_metric':'torchmetrics_validation50k'},
               'fid':None,'best_fid':[(89.38,'best.pt')]}
    with pytest.raises(ValueError,match='No unique original FID'):
        logging.rebase_checkpoint_fids(payload,[])


def test_duplicate_summary_cleanup_keeps_training_metrics_and_canonical_scores():
    summary = {'val/fid.last':89.38,'eval/fid':89.38,'eval/fid_original_val50k':89.47,
               'eval/inception_score_std':.24,'evaluation/best_fid':89.38,
               'evaluation/fid_reference':'original-rqvae/imagenet_train',
               'train/loss':8.,logging.FID_KEY:93.48,logging.IS_KEY:12.07}
    logging.clear_legacy_summaries(SimpleNamespace(summary=summary))
    assert summary == {'evaluation/fid_reference':'original-rqvae/imagenet_train',
                       'train/loss':8.,logging.FID_KEY:93.48,logging.IS_KEY:12.07}


def test_resume_summary_uses_newest_original_evaluation(tmp_path):
    import json
    folder = tmp_path/'evaluations'
    folder.mkdir()
    latest = {'train/global_step':2504, logging.FID_KEY:70.,logging.IS_KEY:20.}
    (folder/'original_generation_step_0002504.json').write_text(json.dumps(latest))
    run = SimpleNamespace(summary={})
    logging.restore_summary(run,[{'global_step':1252,'original_train_fid':93.48,
                                 'inception_score':12.07}],tmp_path)
    assert run.summary == {logging.FID_KEY:70.,logging.IS_KEY:20.}
