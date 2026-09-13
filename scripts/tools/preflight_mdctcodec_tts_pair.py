#!/usr/bin/env python3
"""Small GPU preflight using full utterances and the real paired initialization."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import soundfile as sf
import torch
from src.models.laser_tts import LaserTTS,TTSConfig
from src.tts_data import TTSDataset,collate_tts
from src.tts_pairing import common_state,file_sha,state_sha
from src.tts_runtime import CodecDecoder
from archive.scripts.benchmark_mdctcodec_vctk import align_mdct


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('outputs/mdctcodec_tts_paired'))
    p.add_argument('--device',default='cuda:0');args=p.parse_args()
    torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=True
    plan=json.loads((args.root/'plan.json').read_text()); started=time.monotonic()
    caches={}
    full=torch.load(plan['arms']['laser']['cache'],map_location='cpu',weights_only=True)
    shard=torch.load(args.root/'rvq_cache/shard_000000.pt',map_location='cpu',weights_only=True)
    eligible={r['path'] for r in shard['records'] if r['split']=='train'}
    selected=[r['path'] for r in full['records'] if r['path'] in eligible][:8]
    base={k:v for k,v in full.items() if k!='records'}
    for arm,records in [('laser',full['records']),('rvq',shard['records'])]:
        lookup={r['path']:r for r in records}
        caches[arm]={**base,'records':[lookup[path] for path in selected]}
    del full,shard
    results={}
    for arm,item in plan['arms'].items():
        model=LaserTTS(TTSConfig(**item['model_config']))
        assert file_sha(item['initialization'])==item['initialization_sha256']
        model.load_state_dict(torch.load(item['initialization'],map_location='cpu',weights_only=True))
        assert state_sha(common_state(model))==plan['common_initialization_sha256']
        model.to(args.device).train()
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,betas=(.9,.95),weight_decay=.01)
        data=TTSDataset(caches[arm],'train');losses=[]
        for step in range(2):
            torch.manual_seed(plan['seed']+1+step); optimizer.zero_grad(set_to_none=True)
            # Four full-utterance microbatches per update, as in production.
            for micro in range(4):
                batch=collate_tts([data[2*micro],data[2*micro+1]])
                batch={k:v.to(args.device) for k,v in batch.items()}
                with torch.autocast('cuda',dtype=torch.bfloat16): result=model(batch,guide_weight=.2)
                assert torch.isfinite(result['loss'])
                (result['loss']/4).backward();losses.append(float(result['loss']))
            assert model.phones.weight.grad.abs().sum()>0
            assert model.depth_blocks[0].attention.q.weight.grad.abs().sum()>0
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
        model.eval();decoder=CodecDecoder(item['codec_checkpoint'],args.device)
        audio,payload=decoder.decode(data[0]['codes'])
        assert len(payload)==len(data[0]['codes'])*5
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            codes,info=model.generate(data[0]['phones'][None].to(args.device),
                torch.tensor([data[0]['speaker']],device=args.device),min_frames=2,max_frames=30,temperature=0)
        generated,encoded=decoder.decode(codes)
        target=args.root/'gpu_preflight';target.mkdir(exist_ok=True)
        sf.write(target/f'{arm}_untrained_smoke.wav',generated,48000,subtype='FLOAT')
        results[arm]={'optimizer_updates':2,'full_utterance_microbatches':8,'losses':losses,
            'last_gradient_norm':float(norm),'codec_payload_bytes_per_frame':len(payload)/len(data[0]['codes']),
            'generated_frames':len(codes),'generated_payload_bytes':len(encoded),
            'common_initialization_verified':True,'audio_is_untrained_smoke_only':True}
        if arm=='rvq':
            agreements=[]
            with torch.inference_mode():
                for record in caches[arm]['records']:
                    wave,sr=sf.read(record['path'],dtype='float32');assert sr==48000
                    aligned,_=align_mdct(torch.from_numpy(wave).to(args.device)[None,None])
                    _,_,sparse=decoder.model.encode(aligned)
                    ids=sparse.support[0,0].cpu()
                    agreements.append(float((ids==record['codes']).float().mean()))
            results[arm]['cpu_cache_vs_gpu_encode_token_agreement_first8']=agreements
        del optimizer,model,decoder;torch.cuda.empty_cache()
    torch.cuda.synchronize(args.device)
    result={'status':'passed','gpu':torch.cuda.get_device_name(args.device),'seconds':time.monotonic()-started,
            'timing_is_not_a_benchmark':True,'plan_sha256':file_sha(args.root/'plan.json'),'arms':results}
    (args.root/'gpu_preflight.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
