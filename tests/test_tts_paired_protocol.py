import json
from pathlib import Path

import pytest

from scripts.tools.prepare_mdctcodec_tts_pair import fresh_targets
from scripts.tools.evaluate_mdctcodec_tts_pair import verify_pair
from scripts.tools.run_mdctcodec_tts_pair import preceding_campaign_done
from src.tts_pairing import file_sha


def test_fresh_evaluation_assigns_unique_texts_without_previous_targets():
    records=[{'speaker':s,'text_key':t,'split':'test','path':f'{s}/{t}'}
             for s,texts in [('a',['shared','first','excluded']),('b',['shared'])] for t in texts]
    chosen=fresh_targets(records,{'excluded'})
    assert {r['speaker'] for r in chosen}=={'a','b'}
    assert {r['text_key'] for r in chosen}=={'shared','first'}
    assert fresh_targets(records,{'excluded'})==chosen
    with pytest.raises(ValueError): fresh_targets(records,{'excluded','first'})


def paired_fixture(root):
    protocol={'train':{'epochs':2},'seed':7}
    (root/'protocol.json').write_text(json.dumps(protocol))
    audit=[{'step':1,'epoch':0,'chain':'a','learning_rate':1e-4},
           {'step':2,'epoch':1,'chain':'b','learning_rate':1e-5}]
    for arm in ('laser','rvq'):
        folder=root/arm;folder.mkdir()
        (folder/'completion.json').write_text(json.dumps({'completed_epochs':2,'step':2,
            'best_generation':[{'epoch':1,'score':.1}],'data_chain':'b'}))
        (folder/'data_order.jsonl').write_text('\n'.join(json.dumps(r) for r in audit)+'\n')
        (folder/'resolved_config.json').write_text(json.dumps({'paired':{'protocol_sha256':file_sha(root/'protocol.json')},
            'train':protocol['train'],'seed':7}))


def test_report_refuses_mismatched_training_data_or_lr(tmp_path):
    paired_fixture(tmp_path)
    verify_pair(tmp_path)
    path=tmp_path/'rvq/data_order.jsonl'
    records=[json.loads(x) for x in path.read_text().splitlines()]
    records[0]['learning_rate']=2e-4
    path.write_text('\n'.join(json.dumps(r) for r in records))
    with pytest.raises(AssertionError,match='diverged'): verify_pair(tmp_path)


def test_report_refuses_unequal_completed_epoch_budgets(tmp_path):
    paired_fixture(tmp_path)
    path=tmp_path/'rvq/completion.json';result=json.loads(path.read_text());result['completed_epochs']=1
    path.write_text(json.dumps(result))
    with pytest.raises(AssertionError,match='epoch budget'): verify_pair(tmp_path)


def test_queue_waits_for_old_campaign_and_live_children(tmp_path,monkeypatch):
    status=tmp_path/'status.json'
    status.write_text(json.dumps({'status':'running','jobs':[]}))
    assert not preceding_campaign_done(status)
    status.write_text(json.dumps({'status':'complete','jobs':[{'pid':123}]}))
    monkeypatch.setattr('scripts.tools.run_mdctcodec_tts_pair.alive',lambda pid:True)
    assert not preceding_campaign_done(status)
    monkeypatch.setattr('scripts.tools.run_mdctcodec_tts_pair.alive',lambda pid:False)
    assert preceding_campaign_done(status)
