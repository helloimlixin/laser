"""Both codec arms train and decode through the same hard byte-budget controller."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from src.audio_hard6k_bitstream import FRAME_BITS, frame_budget, max_packet_bytes, pack_packet, unpack_packet
from src.mdctcodec_matched import MatchedModel, align_mdct, measure
from src.mdctcodec_k4 import K4AudioModel, K4TrainingStatistics


class HardRateMixin:
    def __init__(self, **kwargs):
        requested=kwargs.pop('hard_rate_cap_bps',6000)
        if requested!=6000: raise ValueError('This format has a hard 6000-bit/s limit')
        super().__init__(**kwargs)
        self.hparams['hard_rate_cap_bps']=6000
        if self.bypass_bottleneck or self.bottleneck_mix_warmup_steps:
            raise ValueError('The hard-rate bottleneck must be active throughout training')

    @property
    def arm(self): return 'rvq' if self.bottleneck_type=='mdctcodec_rvq' else 'laser'

    def budget_latents(self,x):
        original_samples=int(x.shape[-1])
        frames=frame_budget(original_samples,self.arm)
        aligned,_=align_mdct(x)
        full=self._to_bottleneck_input(self.pre_bottleneck(self.encoder(aligned)))
        pooled=F.adaptive_avg_pool1d(full.squeeze(2),frames).unsqueeze(2)
        return pooled,full.shape[-1],original_samples

    def encode(self,x):
        pooled,full_frames,samples=self.budget_latents(x)
        with self._bottleneck_autocast_context(pooled):
            quantized,loss,codes=self.bottleneck(pooled.float())
        self._last_bottleneck_mix_fraction=pooled.new_tensor(1.).detach()
        self._last_rate_samples=samples;self._last_rate_frames=pooled.shape[-1]
        self._last_rate_codes=codes
        expanded=F.interpolate(quantized.squeeze(2),size=full_frames,mode='linear',align_corners=False).unsqueeze(2)
        return expanded,loss,codes

    def forward(self,x):
        quantized,loss,codes=self.encode(x)
        return self.decode(quantized)[...,:x.shape[-1]],loss,codes

    @torch.inference_mode()
    def encode_packet(self,x):
        if x.ndim!=3 or x.shape[:2]!=(1,1): raise ValueError('Encode one mono utterance at a time')
        with torch.autocast(device_type=x.device.type,enabled=False):
            _,_,codes=self.encode(x.float())
            atoms=codes.support.reshape(-1,4).cpu().numpy()
            bins=(self.bottleneck.coefficient_bins(codes.values).reshape(-1,4).cpu().numpy()
                  if self.arm=='laser' else None)
        return pack_packet(self.arm,int(x.shape[-1]),atoms,bins)

    @torch.inference_mode()
    def decode_packet(self,packet):
        data=unpack_packet(packet,expected_arm=self.arm)
        device=next(self.parameters()).device
        ids=torch.from_numpy(data['codes']).to(device)
        with torch.autocast(device_type=device.type,enabled=False):
            if self.arm=='laser':
                ids=ids[None,None]
                bins=torch.from_numpy(data['coefficient_bins']).to(device)[None,None]
                values=self.bottleneck.coefficient_levels[bins]
                sparse=self.bottleneck._reconstruct_sparse(ids,values,1,data['frames'])
            else:
                sparse=self.bottleneck.quantizer.from_codes(ids.T[None])[0].unsqueeze(2)
            # Original sample count fully determines the backbone MDCT frame count.
            full_frames=(data['samples']+40+319)//320
            expanded=F.interpolate(sparse.squeeze(2),size=full_frames,mode='linear',align_corners=False).unsqueeze(2)
            waveform=self.decode(expanded)[...,:data['samples']]
        if waveform.shape!=(1,1,data['samples']) or not torch.isfinite(waveform).all():
            raise RuntimeError('Invalid hard-rate decoded waveform')
        return waveform.clamp(-1,1)

    def on_validation_epoch_start(self):
        MatchedModel.on_validation_epoch_start(self)
        self._validation_rates=[];self._validation_frame_rates=[]

    def validation_step(self,batch,batch_idx):
        x,_,metadata=batch
        packet=self.encode_packet(x)
        decoded=self.decode_packet(packet)
        samples=int(x.shape[-1]);rate=len(packet)*8*48000/samples
        if len(packet)>max_packet_bytes(samples) or rate>6000:
            raise RuntimeError('Validation produced an over-budget packet')
        reference=x[0,0].float().cpu().numpy();reconstruction=decoded[0,0].float().cpu().numpy()
        self._metric_jobs.append(self._metric_pool.submit(measure,(Path(metadata['path'][0]).name,reference,reconstruction)))
        self._validation_payload_bits+=len(packet)*8;self._validation_samples+=samples
        self._validation_rates.append(rate)
        self._validation_frame_rates.append(frame_budget(samples,self.arm)*48000/samples)

    def on_validation_epoch_end(self):
        MatchedModel.on_validation_epoch_end(self)
        maximum=max(self._validation_rates)
        self.log('val/maximum_packet_kbps',maximum/1000)
        self.log('val/rate_violations',0.)
        self.log('val/coded_frames_per_second',float(np.mean(self._validation_frame_rates)))
        if self.output_path:
            p=Path(self.output_path)/'validation'/f'step-{int(self._manual_train_step):07d}.json'
            data=json.loads(p.read_text())
            data.update(hard_limit_bps=6000,maximum_packet_kbps=maximum/1000,rate_violations=0,
                header_and_coefficients_included=True,
                coded_frames_per_second=float(np.mean(self._validation_frame_rates)))
            p.write_text(json.dumps(data,indent=2))


class HardRateLASER(HardRateMixin,K4AudioModel):
    def __init__(self,**kwargs):
        if kwargs.get('num_embeddings')!=4096 or kwargs.get('sparsity_level')!=4:
            raise ValueError('Hard6k LASER requires 4096 atoms and K4')
        super().__init__(**kwargs)
        self.hparams['k4_transport']='4096 atoms, K4, nine coefficient symbols; 57-bit combinatorial frames with a hard 6000-bit/s packet cap'


class HardRateRVQ(HardRateMixin,MatchedModel):
    def __init__(self,**kwargs):
        if kwargs.get('num_embeddings')!=1024 or kwargs.get('rq_code_depth')!=4 or kwargs.get('bottleneck_type')!='mdctcodec_rvq':
            raise ValueError('Hard6k RVQ requires four 1024-entry codebooks')
        super().__init__(**kwargs)


class HardRateStatistics(K4TrainingStatistics):
    @torch.no_grad()
    def on_train_batch_end(self,trainer,model,outputs,batch,batch_idx):
        if not model.training: raise RuntimeError('Training statistics outside training')
        codes=model._last_rate_codes;frames=model._last_rate_frames;samples=model._last_rate_samples
        if frames!=frame_budget(samples,model.arm) or codes.support.shape[-2:]!=(frames,4):
            raise RuntimeError('Training bypassed the hard rate controller')
        size=16+(frames*FRAME_BITS[model.arm]+7)//8
        if size>max_packet_bytes(samples): raise RuntimeError('Training representation exceeds budget')
        if model.arm=='laser':
            if model.bottleneck.dictionary.shape!=(32,4096): raise RuntimeError('LASER dictionary size changed')
            model.bottleneck.update_levels_after_batch()
        elapsed=self.state_dict()['elapsed_seconds']
        if elapsed>=self.ceiling:
            model.continuation_stopped=True;trainer.should_stop=True
        if int(model._manual_train_step)%20==0:
            metrics={'train/packet_kbps':size*8*48000/samples/1000,
                'train/rate_violations':0,'train/coded_frames':frames,'train/sparsity_or_depth':4,
                'train/crop_seconds':samples/48000,'train/coded_frames_per_second':frames*48000/samples,
                'train/total_dictionary_vectors':4096,'train/assigned_gpu_hours':elapsed/3600}
            if model.arm=='laser':
                metrics.update({'train/nonzero_coefficients_per_frame':float((codes.values!=0).float().sum(-1).mean()),
                                'train/coefficient_level_max':float(model.bottleneck.coefficient_levels[-1])})
            model.logger.log_metrics(metrics,step=trainer.global_step)


@torch.inference_mode()
def reconstruct_hard6k(model,x):
    packet=model.encode_packet(x)
    return model.decode_packet(packet),packet
