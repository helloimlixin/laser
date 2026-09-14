import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.audio_hard6k_bitstream import frame_budget,pack_tts_codes,samples_for_frames,unpack_packet
from src.models.laser_tts import LaserTTS,TTSConfig
from src.tts_data import TTSDataset,FrameBatchSampler,collate_tts
from src.tts_pairing import batch_chain,common_state,record_signature,state_sha
from scripts.tools.prepare_mdctcodec_hard6k_tts import select_best
from scripts.tools.run_mdctcodec_tts_pair import preceding_hard6k_done,hard6k_jobs


def prior(arm='laser'):
    return LaserTTS(TTSConfig(phone_vocab=10,speakers=2,width=32,heads=4,text_layers=1,
        audio_layers=1,dropout=0,depth_layers=2,codec=arm,laser_atoms=4096,laser_sparsity=4,
        coefficient_levels=9,depth_positions=8,hard_rate_cap_bps=6000)).eval()


def test_k4_depth_is_causal_and_cached_generation_matches_teacher():
    model=prior();context=torch.randn(2,3,32)
    teacher=torch.tensor([1,2,3,4,5,6,7,8]).expand(2,3,8).clone()
    expected=model.depth_logits(context,teacher)
    for changed_field in range(8):
        changed=teacher.clone();changed[...,changed_field]=(changed[...,changed_field]+1)%(4096 if changed_field%2==0 else 9)
        actual=model.depth_logits(context,changed)
        for d in range(changed_field+1):torch.testing.assert_close(actual[d],expected[d])
    fields=[];caches=[{} for _ in model.depth_blocks]
    for d in range(8):
        actual=model.depth_next_logits(context.flatten(0,1),fields,caches)
        if d and d%2==0:actual=actual.scatter(1,teacher.flatten(0,1)[:,:d:2],-1e4)
        torch.testing.assert_close(actual.reshape_as(expected[d]),expected[d],atol=1e-5,rtol=1e-5)
        fields.append(teacher[...,d].flatten())


@pytest.mark.parametrize('arm',['laser','rvq'])
def test_generated_fields_pack_at_hard_rate_and_fixed_seconds(arm):
    model=prior(arm)
    with torch.no_grad():
        for head in model.heads:head.weight.zero_();head.bias.fill_(-100);head.bias[0]=100
    tokens,info=model.generate(torch.tensor([[3,2]]),torch.tensor([0]),temperature=0,
        min_seconds=.1,max_seconds=.1)
    assert len(tokens)==frame_budget(4800,arm)
    packet=pack_tts_codes(tokens.numpy(),arm);parsed=unpack_packet(packet,arm)
    assert 8*len(packet)*48000<=6000*parsed['samples'] and parsed['samples']<=4800
    assert info['seconds']==parsed['samples']/48000
    if arm=='laser':assert (np.diff(np.sort(tokens.numpy()[:,0::2],axis=1),axis=1)>0).all()


@pytest.mark.parametrize('arm',['laser','rvq'])
def test_duration_roundtrip_keeps_all_frames_and_header_under_cap(arm):
    for frames in [1,2,3,30,100,1000,2250]:
        samples=samples_for_frames(frames,arm)
        fields=np.tile([4,0,1,8,3,2,2,5] if arm=='laser' else [0,1023,0,1023],(frames,1))
        payload=pack_tts_codes(fields,arm)
        parsed=unpack_packet(payload)
        assert parsed['samples']==samples and parsed['frames']==frames
        assert frame_budget(samples,arm)==frames and 8*len(payload)*48000<=6000*samples


def cache(arm):
    records=[]
    for i,samples in enumerate([24000,48000,32000,100000,200000]):
        row={'path':f'/p/{i}.flac','speaker':'p1','text':'test text','text_key':'test text',
            'phonemes':'a b','split':'train','samples':samples,'seconds':samples/48000,
            'codes':torch.tensor([1,2,3,4,5,6,7,8] if arm=='laser' else [1,2,3,4],dtype=torch.int16).repeat(frame_budget(samples,arm),1)}
        records.append(row)
    return {'records':records,'phone_to_id':{'a':3,'b':4},'speaker_to_id':{'p1':0},'batch_length_basis':'native_150hz'}


def test_shared_data_audit_matches_different_field_counts_and_actual_lengths():
    caches=[cache(a) for a in ('laser','rvq')];datasets=[TTSDataset(c,'train') for c in caches]
    assert record_signature(caches[0],False)==record_signature(caches[1],False)
    assert record_signature(caches[0])!=record_signature(caches[1])
    batches=[list(FrameBatchSampler(d.batch_lengths,frame_budget=1000,max_batch=4)) for d in datasets]
    assert batches[0]==batches[1]
    for indices in batches[0]:
        pair=[collate_tts([d[i] for i in indices]) for d in datasets]
        assert batch_chain('00'*32,pair[0])==batch_chain('00'*32,pair[1])
        assert not torch.equal(pair[0]['lengths'],pair[1]['lengths'])
    models=[prior(a) for a in ('laser','rvq')]
    models[1].load_state_dict(common_state(models[0]),strict=False)
    assert state_sha(common_state(models[0]))==state_sha(common_state(models[1]))
    for model,dataset in zip(models,datasets):
        loss=model(collate_tts([dataset[0]]))['loss'];loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(model.depth_blocks[0].attention.q.weight.grad).all()


def selection_fixture(root):
    (root/'laser').mkdir();(root/'protocol.json').write_text(json.dumps({'generator_updates':600000}))
    best=root/'best.ckpt';latest=root/'last.ckpt'
    torch.save({'hyper_parameters':{'hard_rate_cap_bps':6000,'num_embeddings':4096,'sparsity_level':4},
        'state_dict':{'_manual_train_step':torch.tensor(575210)}},best)
    latest.write_bytes(b'not the best')
    done={'status':'complete','generator_updates':600000,'best_checkpoint':str(best),
        'best_validation_visqol':4.24,'best_three':{str(best):4.24,str(latest):4.20},'url':'test'}
    (root/'laser/completion.json').write_text(json.dumps(done));return done


def test_selects_best_visqol_instead_of_last_and_requires_finished_stage1(tmp_path):
    done=selection_fixture(tmp_path)
    source,metadata=select_best(tmp_path,'laser')
    assert source==tmp_path/'best.ckpt' and metadata['stage1_selected_updates']==575210
    done['best_checkpoint']=str(tmp_path/'last.ckpt')
    (tmp_path/'laser/completion.json').write_text(json.dumps(done))
    with pytest.raises(ValueError,match='highest'):select_best(tmp_path,'laser')
    done['status']='running';(tmp_path/'laser/completion.json').write_text(json.dumps(done))
    with pytest.raises(ValueError,match='finish'):select_best(tmp_path,'laser')


def test_queue_waits_for_stage1_comparison_and_supervisor_exit(tmp_path,monkeypatch):
    selection_fixture(tmp_path);(tmp_path/'rvq').mkdir()
    (tmp_path/'rvq/completion.json').write_text((tmp_path/'laser/completion.json').read_text())
    state={'status':'complete','pid':123,'comparison':{'status':'complete'},'jobs':[]}
    (tmp_path/'status.json').write_text(json.dumps(state));(tmp_path/'comparison_complete.json').write_text('{}')
    monkeypatch.setattr('scripts.tools.run_mdctcodec_tts_pair.alive',lambda pid:True)
    assert not preceding_hard6k_done(tmp_path)
    monkeypatch.setattr('scripts.tools.run_mdctcodec_tts_pair.alive',lambda pid:False)
    assert preceding_hard6k_done(tmp_path)
    (tmp_path/'comparison_complete.json').unlink();assert not preceding_hard6k_done(tmp_path)
    jobs=hard6k_jobs(tmp_path,tmp_path,'python')
    assert {j['name'] for j in jobs if j['gpu'] is not None}=={'cache_laser','cache_rvq'}
    assert next(j for j in jobs if j['name']=='finalize')['deps']==['merge_laser','merge_rvq']
