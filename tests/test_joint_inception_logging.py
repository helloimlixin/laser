import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location('inception_logging',
    Path(__file__).resolve().parents[1] / 'scripts/tools/log_joint20k_inception.py')
logging = importlib.util.module_from_spec(spec)
spec.loader.exec_module(logging)


def test_report_adds_is_for_primary_and_confirmation_without_changing_fid(tmp_path):
    arms = {}
    for arm, mean in [('control', 20.), ('weighted', 22.)]:
        folder = tmp_path / arm / 'evaluations'
        folder.mkdir(parents=True)
        arms[arm] = dict(out=str(folder.parent))
        for epoch in (2, 3):
            (folder / f'epoch{epoch:03d}.json').write_text(json.dumps({
                'train/global_step': 626 * epoch, 'generated_images': 50000,
                'eval/inception_score': mean, 'eval/inception_score_std': .5}))
        (folder / 'confirmation-seed261101.json').write_text(json.dumps({
            'global_step': 1878, 'generated_images': 50000,
            'inception_score': mean + 1., 'inception_score_std': .4}))
    result = dict(rows=[dict(epoch=epoch, seed=seed, control=60., weighted=59., improvement=1.)
                       for epoch, seed in [(2, 261001), (3, 261001), (3, 261101)]])
    updated = logging.enrich_comparison(dict(arms=arms), result)
    assert 'inception_score' not in result['rows'][0]
    for row in updated['rows']:
        assert row['control'] == 60. and row['weighted'] == 59. and row['improvement'] == 1.
        assert row['inception_score_improvement'] == 2.
    assert updated['rows'][-1]['inception_score']['weighted'] == dict(mean=23., std=.4)


@pytest.mark.parametrize('mean,std,count', [(float('nan'), .5, 50000),
    (20., -1., 50000), (20., .5, 5000)])
def test_invalid_or_unmatched_inception_measurement_is_rejected(mean, std, count):
    with pytest.raises(ValueError):
        logging.inception({'eval/inception_score': mean, 'eval/inception_score_std': std,
                           'generated_images': count})
