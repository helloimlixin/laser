import importlib.util
import json
from pathlib import Path


spec = importlib.util.spec_from_file_location('wandb_config_only',
    Path(__file__).parents[1] / 'scripts/tools/wandb_config_only.py')
metadata = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metadata)


def test_result_publication_does_not_read_or_write_a_live_summary():
    class Run:
        storage_id = 'existing-run'

        def __init__(self):
            self.config = {'native_sampling': {'top_k': [2048]}}
            self.calls = []

        @property
        def summary(self):
            raise AssertionError('Concurrent summary must not be accessed')

        @property
        def json_config(self):
            return json.dumps({key: {'value': value} for key, value in self.config.items()})

        def _exec(self, query, **variables):
            self.calls.append((query, variables))

    run = Run()
    metadata.update_config_only(run, {'sampler_check_epoch4': {'lowest_fid_top_k': 4096}})
    assert len(run.calls) == 1
    query, variables = run.calls[0]
    assert 'summary' not in query.lower()
    assert set(variables) == {'id', 'config'}
    assert variables['id'] == 'existing-run'
    config = json.loads(variables['config'])
    assert config['native_sampling']['value']['top_k'] == [2048]
    assert config['sampler_check_epoch4']['value']['lowest_fid_top_k'] == 4096
