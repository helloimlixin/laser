"""Comparable audio research figures with fixed units and reference-independent gain.

Plots never normalize each reconstruction separately. Metrics use full recordings;
the waveform zoom is chosen once from reference energy, without inspecting outputs.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
import torch

from src.audio_logging import _mel_filterbank
from src.models.audio_codec import MDCTCodecOrthonormalAnalysis
from src.mdctcodec_bitstream import unpack_frames

SAMPLE_RATE = 48000
N_FFT, HOP, MEL_BINS = 2048, 240, 128
LABELS = {'reference':'Reference', 'laser':'LASER', 'rvq':'RVQ',
          'released_mdctcodec':'Released MDCTCodec'}
COLORS = {'reference':'#263238', 'laser':'#0072B2', 'rvq':'#D55E00',
          'released_mdctcodec':'#009E73'}


def select_validation_examples(manifest):
    """One lexically first validation utterance per speaker; never select by score."""
    selected = [sorted(p for p in manifest['validation'] if Path(p).parent.name == speaker)[0]
                for speaker in sorted(manifest['heldout_speakers'])]
    if set(selected) & (set(manifest['train']) | set(manifest['test'])):
        raise ValueError('Preview recordings must be validation-only')
    return selected


def spectral_features(waveform):
    x = torch.as_tensor(np.asarray(waveform).copy(), dtype=torch.float32).reshape(-1)
    if x.numel() == 0 or not torch.isfinite(x).all():
        raise ValueError('Expected finite, nonempty waveform')
    window = torch.hann_window(N_FFT)
    spec = torch.stft(x, N_FFT, HOP, window=window, return_complex=True,
                      center=True, pad_mode='constant')
    # Real, nonnegative power BEFORE mel filtering. Applying a mel bank to
    # complex STFT coefficients can cancel energy and is not a mel spectrogram.
    power = spec.abs().square() / window.sum().square()
    mel_power = _mel_filterbank(sample_rate=SAMPLE_RATE, n_fft=N_FFT, n_mels=MEL_BINS) @ power
    mdct = MDCTCodecOrthonormalAnalysis(40)(x[None,None])[0]
    return {'stft_db':(10 * power.clamp_min(1e-12).log10()).numpy(),
            'mel_db':(10 * mel_power.clamp_min(1e-12).log10()).numpy(),
            'power':power.numpy(), 'mdct':mdct.numpy()}


def decode_payload_tokens(payload, arm):
    if len(payload) % 5 or not payload:
        raise ValueError('Expected nonempty five-byte frames')
    if arm == 'laser':
        ids, coefficients = unpack_frames(payload)
        return ids, coefficients
    if arm not in {'rvq','released_mdctcodec'}:
        raise ValueError(f'Unknown codec {arm}')
    octets = np.frombuffer(payload, np.uint8).astype(np.uint64).reshape(-1,5)
    words = np.bitwise_or.reduce(octets << np.array([32,24,16,8,0],np.uint64), axis=1)
    ids = ((words[:,None] >> np.array([30,20,10,0],np.uint64)) & 1023).astype(np.int64)
    return ids, None


def usage_metrics(ids, size):
    counts = np.bincount(np.asarray(ids).reshape(-1), minlength=size)
    probabilities = counts[counts > 0] / max(1, counts.sum())
    entropy = float(-(probabilities * np.log2(probabilities)).sum())
    return {'used_codes':int((counts > 0).sum()), 'used_fraction':float((counts > 0).mean()),
            'entropy_bits':entropy, 'effective_codes':float(2**entropy)}, counts


def distortion_metrics(reference, decoded, features=None):
    r, y = np.asarray(reference), np.asarray(decoded)
    if r.shape != y.shape or r.size == 0 or not np.isfinite(y).all():
        raise ValueError('Reconstruction must match the finite reference waveform')
    fr, fy = features or (spectral_features(r), spectral_features(y))
    eps = 1e-12
    error = y - r
    frame_lsd = np.sqrt(np.mean((fr['stft_db'] - fy['stft_db'])**2, axis=0))
    rms = lambda a: float(np.sqrt(np.mean(a.astype(np.float64)**2)))
    return {'waveform_mae':float(np.mean(np.abs(error))),
            'snr_db':float(10*np.log10((np.mean(r.astype(np.float64)**2)+eps) /
                                      (np.mean(error.astype(np.float64)**2)+eps))),
            'rms_gain_db':float(20*np.log10((rms(y)+eps)/(rms(r)+eps))),
            'waveform_endpoint_fraction':float(np.mean(np.abs(y) >= 0.9999)),
            'log_mel_mae_db':float(np.mean(np.abs(fr['mel_db'] - fy['mel_db']))),
            'preview_lsd_db':float(frame_lsd.mean())}


def save_figure(fig, path):
    fig.savefig(path, dpi=130, facecolor='white')
    plt.close(fig)
    return str(path)


def render_audio_comparison(waves, output, title):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    names = list(waves)
    reference = np.asarray(waves['reference'])
    if any(np.shape(w) != reference.shape for w in waves.values()):
        raise ValueError('All panels must represent the same sample interval')
    features = {name:spectral_features(wave) for name,wave in waves.items()}
    duration = len(reference) / SAMPLE_RATE
    times = np.arange(len(reference)) / SAMPLE_RATE
    paths = {}
    for feature, label, ylim in [('mel_db','Mel band',MEL_BINS),('stft_db','Frequency (kHz)',24)]:
        fig, axes = plt.subplots(len(names), 1, figsize=(12,2.1*len(names)),
                                 sharex=True, layout='constrained')
        for ax,name in zip(np.atleast_1d(axes),names):
            view = features[name][feature]
            im = ax.imshow(view, origin='lower', aspect='auto', extent=[0,duration,0,ylim],
                           vmin=-100, vmax=0, cmap='magma', interpolation='nearest')
            ax.set_ylabel(label); ax.set_title(LABELS[name], loc='left', fontsize=10)
        axes[-1].set_xlabel('Time (s)')
        fig.colorbar(im, ax=axes, label='Power (dB re 1; identical limits for all panels)')
        fig.suptitle(f'{title} | '+('Log-mel spectrograms' if feature=='mel_db' else 'STFT log-power spectra'))
        key = 'log_mel' if feature=='mel_db' else 'stft'
        paths[key] = save_figure(fig, output/f'{key}.png')
    others = [n for n in names if n!='reference']
    fig, axes = plt.subplots(len(others),1,figsize=(12,2.2*len(others)),sharex=True,layout='constrained')
    axes = np.atleast_1d(axes)
    for ax,name in zip(np.atleast_1d(axes),others):
        diff = features[name]['mel_db'] - features['reference']['mel_db']
        im = ax.imshow(diff, origin='lower', aspect='auto', extent=[0,duration,0,MEL_BINS],
                       vmin=-30,vmax=30,cmap='RdBu_r',interpolation='nearest')
        ax.set_ylabel('Mel band'); ax.set_title(LABELS[name]+' minus reference',loc='left',fontsize=10)
    axes[-1].set_xlabel('Time (s)'); fig.colorbar(im,ax=axes,label='Power difference (dB)')
    fig.suptitle(title+' | Log-mel error (positive = excess energy)')
    paths['log_mel_error'] = save_figure(fig,output/'log_mel_error.png')

    # Figure 4 of MDCTCodec: natural/decoded waveform and MDCT comparisons.
    # Signed coefficients and shared reference-derived limits preserve sign/scale.
    bound = max(float(np.quantile(np.abs(features['reference']['mdct']),0.995)),1e-4)
    fig,axes = plt.subplots(2,len(names),figsize=(4.6*len(names),6),layout='constrained')
    for col,name in enumerate(names):
        axes[0,col].plot(times,waves[name],lw=0.45,color=COLORS[name])
        axes[0,col].set_ylim(-1.05,1.05); axes[0,col].set_title(LABELS[name])
        axes[0,col].set_xlabel('Time (s)'); axes[0,col].set_ylabel('Amplitude')
        im=axes[1,col].imshow(features[name]['mdct'],origin='lower',aspect='auto',
            extent=[0,duration,0,24],vmin=-bound,vmax=bound,cmap='RdBu_r',interpolation='nearest')
        axes[1,col].set_xlabel('Time (s)');axes[1,col].set_ylabel('Frequency (kHz)')
    fig.colorbar(im,ax=axes[1,:],label='Signed orthonormal MDCT coefficient')
    fig.suptitle(title+' | Waveform / MDCT (80-sample window, 40-sample hop)')
    paths['waveform_mdct'] = save_figure(fig,output/'waveform_mdct.png')

    # Same 40 ms voiced interval for all models and epochs, determined by reference.
    hop = 480
    energy = np.array([np.mean(reference[i:i+hop]**2) for i in range(0,len(reference),hop)])
    start = max(0,min(int(np.argmax(energy))*hop,len(reference)-1920))
    end = min(start+1920,len(reference))
    fig,axes = plt.subplots(3,1,figsize=(12,8),layout='constrained')
    for name in names:
        axes[0].plot(times,waves[name],lw=0.5,alpha=0.65,color=COLORS[name],label=LABELS[name])
        axes[1].plot(times[start:end],waves[name][start:end],lw=0.9,color=COLORS[name],label=LABELS[name])
        if name!='reference':axes[2].plot(times,np.asarray(waves[name])-reference,lw=0.45,alpha=0.6,color=COLORS[name])
    # One shared physical-amplitude axis for the zoom: expose fine waveform
    # structure without applying independent gains or clipping any model.
    zoom_bound=max(1e-4,1.1*max(float(np.max(np.abs(waves[n][start:end]))) for n in names))
    axes[0].set_ylim(-1.05,1.05);axes[1].set_ylim(-zoom_bound,zoom_bound);axes[2].set_ylim(-2.05,2.05)
    for ax,label in zip(axes,['Full waveform','Reference-selected 40 ms zoom','Reconstruction minus reference']):
        ax.set_title(label,loc='left',fontsize=10);ax.set_xlabel('Time (s)');ax.set_ylabel('Amplitude');ax.grid(alpha=.15)
    axes[0].legend(ncol=len(names));fig.suptitle(title+' | Waveforms and residual')
    paths['waveforms'] = save_figure(fig,output/'waveforms.png')

    fig,axes = plt.subplots(2,1,figsize=(12,7),layout='constrained')
    bands = [(0,1000),(1000,4000),(4000,8000),(8000,16000),(16000,24000)]
    spectra = {}
    for name in names:
        freq,psd = welch(waves[name],SAMPLE_RATE,nperseg=min(N_FFT,len(reference)))
        spectra[name] = psd
        axes[0].plot(freq/1000,10*np.log10(psd+1e-15),color=COLORS[name],label=LABELS[name],lw=1)
    width=.8/len(others)
    for i,name in enumerate(others):
        gains=[10*np.log10((spectra[name][(freq>=lo)&(freq<hi)].sum()+1e-15)/
                           (spectra['reference'][(freq>=lo)&(freq<hi)].sum()+1e-15)) for lo,hi in bands]
        axes[1].bar(np.arange(len(bands))-.4+width/2+i*width,gains,width,color=COLORS[name],label=LABELS[name])
    axes[0].set(xlabel='Frequency (kHz)',ylabel='Welch PSD (dB / Hz)',xlim=(0,24),ylim=(-150,0))
    axes[0].legend();axes[1].axhline(0,color='black',lw=.7)
    axes[1].set_xticks(range(len(bands)),[f'{lo/1000:g}–{hi/1000:g} kHz' for lo,hi in bands])
    axes[1].set_ylabel('Band energy gain vs reference (dB)');fig.suptitle(title+' | Frequency response and band energy')
    paths['frequency_profile'] = save_figure(fig,output/'frequency_profile.png')

    fig,axes=plt.subplots(2,1,figsize=(12,6),layout='constrained')
    for name in others:
        delta=features[name]['stft_db']-features['reference']['stft_db']
        lsd=np.sqrt(np.mean(delta**2,axis=0))
        axes[0].plot(np.arange(len(lsd))*HOP/SAMPLE_RATE,lsd,color=COLORS[name],label=LABELS[name])
        error=np.asarray(waves[name])-reference
        error_rms=[10*np.log10(np.mean(error[i:i+hop]**2)+1e-12) for i in range(0,len(error),hop)]
        axes[1].plot(np.arange(len(error_rms))*hop/SAMPLE_RATE,error_rms,color=COLORS[name])
    axes[0].set_ylabel('Frame log-spectral distance (dB)');axes[0].legend()
    axes[1].set_ylabel('10 ms error RMS (dBFS)');axes[1].set_ylim(-120,5)
    for ax in axes:ax.set_xlabel('Time (s)');ax.grid(alpha=.15)
    fig.suptitle(title+' | Error over time (unaligned, original levels)')
    paths['temporal_error'] = save_figure(fig,output/'temporal_error.png')
    metrics={name:distortion_metrics(reference,waves[name],(features['reference'],features[name])) for name in others}
    return paths,metrics


def render_token_diagnostics(tokens, output, title):
    """Usage is over the preview subset only, not a whole-dataset codebook estimate."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    metrics={};fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for arm in ['laser','rvq']:
        ids=np.concatenate([item['ids'] for item in tokens[arm]])
        if arm=='laser':
            stats,counts=usage_metrics(ids,8192);metrics[arm]=stats
            axes[0,0].plot(np.arange(8192),counts,color=COLORS[arm],lw=.7)
            axes[0,0].set(title='LASER atom usage',xlabel='Atom ID',ylabel='Selections')
            coefficients=np.concatenate([item['coefficients'] for item in tokens[arm]])
            for slot in range(2):axes[1,0].hist(coefficients[:,slot],bins=np.arange(-63.5,64.5),alpha=.6,label=f'Selection {slot+1}')
            axes[1,0].set(title='LASER quantized coefficient histogram',xlabel='Signed integer (−63 … 63)',ylabel='Count');axes[1,0].legend()
            metrics[arm]['coefficient_endpoint_fraction']=float(np.mean(np.abs(coefficients)==63))
            metrics[arm]['coefficient_zero_fraction']=float(np.mean(coefficients==0))
        else:
            metrics[arm]={}
            for level in range(4):
                stats,counts=usage_metrics(ids[:,level],1024);metrics[arm][f'level_{level+1}']=stats
                axes[0,1].plot(np.arange(1024),counts,lw=.7,alpha=.7,label=f'Level {level+1}')
            axes[0,1].set(title='RVQ code usage by level',xlabel='Code ID',ylabel='Selections');axes[0,1].legend()
    for arm in ['laser','rvq']:
        groups=[metrics[arm]] if arm=='laser' else list(metrics[arm].values())
        labels=['LASER'] if arm=='laser' else [f'RVQ {i+1}' for i in range(4)]
        axes[1,1].bar(labels,[s['used_fraction']*100 for s in groups],color=COLORS[arm])
    axes[1,1].set(title='Codes observed in these previews',ylabel='Codebook used (%)',ylim=(0,100))
    fig.suptitle(title+' | Preview-subset token diagnostics (no entropy coding assumed)')
    return save_figure(fig,output/'token_diagnostics.png'),metrics


def render_tts_preview(reference, generated, output, title, attention=None, phone_labels=None, attention_cmap='Blues'):
    """Free-running speech has independent timing: no pointwise error comparison."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(13,7),layout='constrained')
    for col,(label,waveform) in enumerate([('Reference',reference),('Generated',generated)]):
        waveform=np.asarray(waveform);duration=len(waveform)/SAMPLE_RATE
        axes[0,col].plot(np.arange(len(waveform))/SAMPLE_RATE,waveform,lw=.5)
        axes[0,col].set(title=label,ylim=(-1.05,1.05),xlabel='Time (s)',ylabel='Amplitude')
        mel=spectral_features(waveform)['mel_db']
        im=axes[1,col].imshow(mel,origin='lower',aspect='auto',extent=[0,max(duration,1/SAMPLE_RATE),0,MEL_BINS],
            vmin=-100,vmax=0,cmap='magma',interpolation='nearest')
        axes[1,col].set(xlabel='Time (s)',ylabel='Mel band')
    import textwrap
    fig.suptitle('\n'.join(textwrap.wrap(title,105))+'\nIndependent durations; original levels')
    fig.colorbar(im,ax=axes[1,:],label='Mel power (dB re 1)')
    paths={'waveform_log_mel':save_figure(fig,output/'waveform_log_mel.png')}
    if attention is not None:
        attention=np.asarray(attention)
        if attention.ndim!=2 or not np.isfinite(attention).all():raise ValueError('Expected finite time-by-phoneme attention')
        fig,ax=plt.subplots(figsize=(12,5),layout='constrained')
        im=ax.imshow(attention,origin='lower',aspect='auto',cmap=attention_cmap,vmin=0,vmax=1,interpolation='nearest',
            extent=[-.5,attention.shape[1]-.5,0,max(len(generated)/SAMPLE_RATE,1/SAMPLE_RATE)])
        if phone_labels is not None:
            assert len(phone_labels)==attention.shape[1]
            ax.set_xticks(range(len(phone_labels)),phone_labels,rotation=90,fontsize=8)
        ax.set(xlabel='Input phoneme',ylabel='Generated prefix time (s)',
            title='Text–audio attention alignment\nLast decoder layer, first head')
        fig.colorbar(im,ax=ax,label='Attention probability')
        paths['text_alignment']=save_figure(fig,output/'text_alignment.png')
    return paths
