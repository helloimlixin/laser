"""Serialized inference adapters for the recorded 6 kbps codec comparison."""
from dataclasses import asdict
import hashlib
import importlib
import io
import json
import math
from pathlib import Path
import struct
import sys

import numpy as np
import torch
from torchaudio.functional import resample

from scripts.benchmark_mdctcodec_vctk import align_mdct, load_reference, pad_encodec_segment_tail
from scripts.benchmark_mdctcodec_trained_rvq import load_trained_reference
from src.mdctcodec_bitstream import pack_frames, unpack_frames
from src.models.laser import LASER


def serialize_codes(codes, metadata, bits=10):
    values=codes.detach().cpu().numpy().astype(np.int64)
    if np.any(values<0) or np.any(values>=2**bits):raise ValueError('Code outside packed vocabulary')
    bitplanes=((values.reshape(-1,1)>>np.arange(bits-1,-1,-1))&1).astype(np.uint8)
    payload=np.packbits(bitplanes.reshape(-1)).tobytes()
    header={**metadata,'shape':list(values.shape),'bits':bits}
    blob=wrap_payload(payload,header)
    decoded_header,decoded_payload=unwrap_payload(blob)
    raw=np.unpackbits(np.frombuffer(decoded_payload,dtype=np.uint8))[:values.size*bits]
    restored=(raw.reshape(-1,bits).astype(np.int64)*(1<<np.arange(bits-1,-1,-1))).sum(1).reshape(decoded_header['shape'])
    if not np.array_equal(values,restored):raise AssertionError('Packed token round trip failed')
    return blob,torch.from_numpy(restored).to(codes.device),decoded_header,len(payload)*8


def wrap_payload(payload,metadata):
    encoded=json.dumps(metadata,sort_keys=True,separators=(',',':')).encode()
    return b'LAC1'+struct.pack('<I',len(encoded))+encoded+payload


def unwrap_payload(blob):
    if blob[:4]!=b'LAC1':raise ValueError('Invalid comparison container')
    size=struct.unpack('<I',blob[4:8])[0]
    return json.loads(blob[8:8+size]),blob[8+size:]


def checkpoint_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CodecAdapter:
    def __init__(self,name,manifest,device):
        self.name,self.device=name,torch.device(device)
        self.info={'system':name,'training_control':'checkpoint comparison; training histories differ',
                   'model_sample_rate':48000,'inference_precision':'float32',
                   'container':'LAC1 experimental fixed-token container; JSON header counted separately'}
        self.first=True
        if name.startswith('laser_'):
            candidate=manifest['codec_candidates']['low_lr' if name=='laser_low_lr' else 'original']
            self.model=LASER.load_from_checkpoint(candidate['checkpoint'],map_location='cpu').to(device).eval()
            self.model.requires_grad_(False)
            assert checkpoint_hash(candidate['checkpoint'])==candidate['sha256']
            assert self.model.bottleneck.coefficient_quantization_bits==7
            self.bound=float(self.model.bottleneck.coefficient_quantization_max)
            self.model.bottleneck.coefficient_quantization_bits=0
            self.info.update(checkpoint=candidate['checkpoint'],sha256=candidate['sha256'],
                             selection_validation_visqol=candidate['validation_visqol'])
        elif name in ['mdctcodec_released','mdctcodec_trained_rvq']:
            root=Path('outputs/mdctcodec_reference/MDCTCodec')
            if name=='mdctcodec_released':
                self.modules=load_reference(root,device)
                self.info['checkpoint_sha256']={p.name:checkpoint_hash(p) for p in [root/'encoder_00200000_onlyvctk',root/'decoder_00200000_onlyvctk']}
            else:
                p=json.loads(Path('outputs/mdctcodec_recovery/trained_rvq_recovery.json').read_text())
                assert checkpoint_hash(p['checkpoint'])==p['sha256']
                self.modules,verification=load_trained_reference(p['checkpoint'],root,device)
                self.info.update(checkpoint=p['checkpoint'],sha256=p['sha256'],verification=verification)
        elif name=='dac_official':
            import dac
            path=dac.utils.download(model_type='44khz',model_bitrate='8kbps')
            self.model=dac.DAC.load(path).to(device).eval()
            self.info.update(checkpoint=str(path),sha256=checkpoint_hash(path),model_sample_rate=44100,
                             inference='official compress/decompress, full utterance, normalize_db=-16, seven codebooks')
        elif name.startswith('encodec_'):
            from encodec import EncodecModel
            self.model=(EncodecModel.encodec_model_48khz() if name=='encodec_48k' else EncodecModel.encodec_model_24khz()).to(device).eval()
            self.model.set_target_bandwidth(6.)
            self.compression=importlib.import_module('encodec.compress')
            # Official deserializer normally reloads weights for every file.
            # Reuse identical frozen weights; parsing and synthesis stay upstream.
            self.compression.MODELS[self.model.name]=lambda:self.model
            self.info.update(model_sample_rate=self.model.sample_rate,channels=self.model.channels,
                             container='official ECDC without entropy coding',
                             inference='official convert_audio + compress + decompress; mono duplicated for 48k stereo',
                             compatibility_fix='ECDC reader uses exact integer frame-count ceiling; unchanged codes/waveforms',
                             weights_sha256=hashlib.sha256(b''.join(t.detach().cpu().numpy().tobytes() for t in self.model.state_dict().values())).hexdigest())
        elif name=='flowdec_6kbps':
            import dac
            from hydra import compose,initialize_config_dir
            from hydra.utils import instantiate
            root=Path('outputs/flowdec_reference').resolve();sys.path.insert(0,str(root))
            path=root/'checkpoints/ndac/ndac-75/800k/dac/weights.pth'
            self.model=dac.DAC.load(path).to(device).eval()
            with initialize_config_dir(config_dir=str(root/'config'),version_base='1.3'):
                config=compose(config_name='flowdec_75m')
            path2=root/'checkpoints/flowdec/flowdec_75m/step=800000.ckpt'
            state=torch.load(path2,map_location='cpu',weights_only=False)
            self.postfilter=instantiate(config['model'])
            self.postfilter.load_state_dict(state['_pl_ema_state_dict'],strict=True)
            self.postfilter.to(device).eval()
            # This released NDAC decoder returns eight fewer samples than
            # hop_length * latent_frames. Measure the structural deficit with
            # zero latents, then include guard samples in the encoded payload.
            with torch.inference_mode():
                probe=self.model.decode(torch.zeros(1,self.model.latent_dim,4,device=device))
            self.tail_guard=int(max(0,4*self.model.hop_length-probe.shape[-1]))
            self.info.update(checkpoints={str(path):checkpoint_hash(path),str(path2):checkpoint_hash(path2)},
                             inference='official demo: NDAC-75, eight quantizers, EMA FlowDec-75m, midpoint N=3/NFE=6',
                             seed='per-utterance SHA256 seed fixed before evaluation',
                             encoded_tail_guard_samples=self.tail_guard,
                             output_peak_policy='official demo rescales reconstruction if peak exceeds one')
        else:raise ValueError(name)

    @torch.inference_mode()
    def reconstruct(self,reference,item_id):
        x=torch.from_numpy(reference).to(self.device)[None,None]
        n=len(reference);side_bits=0
        metadata={'model':self.name,'samples':n,'sample_rate':48000}
        if self.name.startswith('laser_'):
            padded,_=align_mdct(x)
            _,_,codes=self.model.encode(padded)
            q=(codes.values.float().clamp(-self.bound,self.bound)/(self.bound/63)).round().long()
            payload=pack_frames(codes.support.cpu().numpy(),q.cpu().numpy())
            blob=wrap_payload(payload,{**metadata,'coefficient_max':self.bound})
            header,payload=unwrap_payload(blob);atoms,values=unpack_frames(payload)
            support=torch.from_numpy(atoms).to(self.device).reshape_as(codes.support)
            coefficients=torch.from_numpy(values).to(self.device).reshape_as(q).float()*(header['coefficient_max']/63)
            y=self.model.decode_from_atoms_and_coeffs(support,coefficients)
            bits=len(payload)*8;side_bits=32
        elif self.name.startswith('mdctcodec_'):
            encoder,quantizer,decoder=self.modules
            padded,_=align_mdct(x);latent,codes,*_=quantizer(encoder(padded),n_quantizers=4)
            blob,restored,header,bits=serialize_codes(codes,metadata)
            parsed=quantizer.from_codes(restored)[0]
            # Forward RVQ uses straight-through arithmetic; code lookup avoids
            # its FP32 cancellation error. Integer round trip above is exact.
            if self.first:torch.testing.assert_close(parsed,latent,rtol=1e-4,atol=1e-4)
            y=decoder(parsed)
        elif self.name=='dac_official':
            from audiotools import AudioSignal
            from dac.model.base import DACFile
            encoded=self.model.compress(AudioSignal(x,48000),win_duration=None,normalize_db=-16,n_quantizers=7)
            details={k:v for k,v in asdict(encoded).items() if k!='codes'}
            details['input_db']=float(encoded.input_db.item())
            blob,restored,header,bits=serialize_codes(encoded.codes,{**metadata,'dac_metadata':details})
            parsed=header['dac_metadata'];parsed['input_db']=torch.tensor([parsed['input_db']],device=self.device)
            decoded_file=DACFile(codes=restored,**parsed)
            y=self.model.decompress(decoded_file).audio_data
            if self.first:
                direct=self.model.decompress(encoded).audio_data
                torch.testing.assert_close(y,direct,rtol=1e-5,atol=1e-5)
            side_bits=32
        elif self.name.startswith('encodec_'):
            from encodec.utils import convert_audio
            source=convert_audio(torch.from_numpy(reference)[None],48000,self.model.sample_rate,self.model.channels).to(self.device)
            if self.model.segment_length is not None:
                source=pad_encodec_segment_tail(source,self.model.segment_length,self.model.segment_stride)
            blob=self.compression.compress(self.model,source,use_lm=False)
            decoded,rate=self.compression.decompress(blob,device=self.device)
            if self.first:
                frames=self.model.encode(source[None]);direct=self.model.decode(frames)[0,:,:source.shape[-1]]
                torch.testing.assert_close(decoded,direct,rtol=1e-5,atol=1e-5)
            from encodec import binary
            header=binary.read_ecdc_header(io.BytesIO(blob))
            stride=self.model.segment_stride or source.shape[-1]
            length=self.model.segment_length or source.shape[-1]
            frame_counts=[(min(length,source.shape[-1]-offset)*self.model.frame_rate+rate-1)//rate for offset in range(0,source.shape[-1],stride)]
            bits=sum(frame_counts)*header['nc']*self.model.bits_per_codebook
            side_bits=32*len(frame_counts) if self.model.normalize else 0
            y=resample(decoded.mean(0,keepdim=True)[None],rate,48000) if rate!=48000 else decoded.mean(0,keepdim=True)[None]
        else:
            torch.manual_seed(int(hashlib.sha256(('flowdec-20260912:'+item_id).encode()).hexdigest()[:8],16))
            source=self.model.preprocess(torch.nn.functional.pad(x,(0,self.tail_guard)),48000)
            _,codes,*_=self.model.encode(source,n_quantizers=8)
            blob,restored,header,bits=serialize_codes(codes,metadata)
            latent=self.model.quantizer.from_codes(restored)[0]
            raw=self.model.decode(latent)
            y=self.postfilter.enhance(raw,N=3,solver='midpoint')
            if y.abs().max()>1:y=y/y.abs().max()
        self.first=False
        out=y[0,0,:n].float().cpu().numpy()
        if len(out)!=n or not np.isfinite(out).all():
            raise RuntimeError(f'Invalid waveform for {self.name}/{item_id}: samples={len(out)} expected={n}, nonfinite={int((~np.isfinite(out)).sum())}')
        return out.clip(-1,1),blob,{'payload_bits':bits,'decoder_value_side_bits':side_bits,
                                  'serialized_file_bits':len(blob)*8,'serialization_roundtrip':True}
