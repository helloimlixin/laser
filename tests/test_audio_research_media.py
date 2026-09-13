import numpy as np
import pytest

from src.audio_research_media import (spectral_features, distortion_metrics,
    decode_payload_tokens, select_validation_examples, render_audio_comparison, render_tts_preview)
from src.mdctcodec_bitstream import pack_frames
from archive.scripts.benchmark_mdctcodec_trained_rvq import payload_roundtrip


def test_log_mel_uses_power_and_preserves_six_db_gain_difference():
    time=np.arange(48000)/48000
    x=(.2*np.sin(2*np.pi*1000*time)).astype(np.float32)
    features=spectral_features(x)
    quiet=spectral_features(x*.5)
    active=features['mel_db']>-60
    np.testing.assert_allclose((quiet['mel_db']-features['mel_db'])[active],
                               20*np.log10(.5),atol=.001)
    assert np.isfinite(features['mel_db']).all() and np.isrealobj(features['mel_db'])
    silence=spectral_features(np.zeros(200,dtype=np.float32))
    assert np.isfinite(silence['mel_db']).all()
    assert distortion_metrics(x,x,(features,features))['preview_lsd_db']==0
    assert distortion_metrics(x,x*.5,(features,quiet))['rms_gain_db']==pytest.approx(-6.0206,abs=.001)


def test_token_diagnostics_read_the_actual_five_byte_formats():
    atoms=np.array([[0,8191],[234,876]])
    coefficients=np.array([[-63,63],[0,-7]])
    ids,values=decode_payload_tokens(pack_frames(atoms,coefficients),'laser')
    np.testing.assert_array_equal(ids,atoms);np.testing.assert_array_equal(values,coefficients)
    codes=np.array([[[0,1023],[123,44],[71,88],[999,12]]])
    payload,_=payload_roundtrip(codes)
    ids,values=decode_payload_tokens(payload,'rvq')
    np.testing.assert_array_equal(ids,codes[0].T);assert values is None
    with pytest.raises(ValueError):decode_payload_tokens(payload[:-1],'rvq')


def test_preview_selection_is_score_independent_and_excludes_test():
    manifest={'heldout_speakers':['p2','p1'],'validation':['/p1/b.flac','/p2/a.flac','/p1/a.flac'],
              'test':['/p2/c.flac'],'train':['/p3/a.flac']}
    assert select_validation_examples(manifest)==['/p1/a.flac','/p2/a.flac']
    manifest['test'].append('/p1/a.flac')
    with pytest.raises(ValueError):select_validation_examples(manifest)


def test_figures_render_silent_reference_and_preserve_reconstruction_gain(tmp_path):
    x=np.zeros(2400,dtype=np.float32)
    y=np.full(2400,.02,dtype=np.float32)
    images,metrics=render_audio_comparison({'reference':x,'laser':y},tmp_path,'Silence diagnostic')
    assert set(images)=={'log_mel','stft','log_mel_error','waveform_mdct','waveforms','frequency_profile','temporal_error'}
    assert all((tmp_path/(key+'.png')).stat().st_size>1000 for key in images)
    assert metrics['laser']['waveform_mae']==pytest.approx(.02)


def test_tts_previews_allow_independent_durations_and_show_text_attention(tmp_path):
    reference=np.zeros(2400,dtype=np.float32)
    generated=np.zeros(4000,dtype=np.float32)
    attention=np.full((12,3),1/3)
    images=render_tts_preview(reference,generated,tmp_path,'TTS diagnostic',attention,['h','i','EOS'])
    assert set(images)=={'waveform_log_mel','text_alignment'}
    assert all((tmp_path/(key+'.png')).stat().st_size>1000 for key in images)
