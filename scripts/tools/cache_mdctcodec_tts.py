#!/usr/bin/env python3
"""Re-encode an audited VCTK text inventory with a selected frozen LASER codec.

Workers own disjoint 500-utterance shards. Text/phonemes may be reused; audio
tokens are always recomputed and tied to the exact checkpoint SHA-256.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import sys
import time

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
import soundfile as sf
import torch
from src.models.laser import LASER
from src.tts_data import CODEC_HELDOUT_SPEAKERS,text_key,text_split
from src.mdctcodec_bitstream import pack_frames,unpack_frames
from archive.scripts.benchmark_mdctcodec_vctk import align_mdct
from archive.scripts.benchmark_mdctcodec_trained_rvq import payload_roundtrip


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False))


def prepare(args):
    old=read(args.inventory_from)
    checkpoint=args.checkpoint.resolve();digest=sha(checkpoint)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    hp=saved['hyper_parameters']
    rvq=hp['bottleneck_type']=='mdctcodec_rvq'
    if rvq:
        assert hp['num_embeddings']==1024 and hp['rq_code_depth']==4
    else:
        assert hp['bottleneck_type']=='dictionary' and hp['num_embeddings']==8192 and hp['sparsity_level']==2
        assert hp['coefficient_quantization_bits']==7
    def check(record):
        assert 'codes' not in record and record['speaker'] not in CODEC_HELDOUT_SPEAKERS
        assert record['split']==text_split(record['text']) and record['text_key']==text_key(record['text'])
        path=Path(record['path'])
        transcript=path.parents[2]/'txt'/record['speaker']/('_'.join(path.stem.split('_')[:2])+'.txt')
        assert transcript.read_text().strip()==record['text'],path
        info=sf.info(path)
        assert info.channels==1 and info.samplerate==48000 and info.frames==record['samples'],path
        assert record['phonemes'].split()
    with ThreadPoolExecutor(max_workers=16) as pool:list(pool.map(check,old['records']))
    inventory={**old,'codec_checkpoint':str(checkpoint),'codec_sha256':digest,
        'codec_selection':args.selection,'source_inventory':{'path':str(args.inventory_from.resolve()),'sha256':sha(args.inventory_from)},
        'codec':({'type':'rvq','frame_rate':150,'codebooks':4,'vocab_sizes':[1024]*4,
                  'bits_per_frame':40,'coefficient_max':None} if rvq else
                 {**old['codec'],'coefficient_max':float(hp['coefficient_quantization_max'])}),
        'codec_generator_updates':int(saved['state_dict']['_manual_train_step']),
        'cache_policy':'Fresh FP32 encoding from full utterances; fixed checkpoint bound; text/phonemes reused after transcript checks'}
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/'inventory.json'
    if path.exists():assert read(path)==inventory
    else:write(path,inventory)
    print(json.dumps({'prepared':str(path),'records':len(inventory['records']),'codec_sha256':digest,
                      'coefficient_max':inventory['codec']['coefficient_max']},indent=2),flush=True)


def worker(args):
    inventory=read(args.output/'inventory.json')
    checkpoint=Path(inventory['codec_checkpoint']);assert sha(checkpoint)==inventory['codec_sha256']
    assert 0<=args.worker_index<args.worker_count
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    model=LASER.load_from_checkpoint(checkpoint,map_location='cpu',strict=True).to(args.device).eval()
    model.requires_grad_(False)
    rvq=model.bottleneck_type=='mdctcodec_rvq'
    bound=inventory['codec']['coefficient_max']
    if not rvq:assert model.bottleneck.coefficient_quantization_max==bound
    records=inventory['records'];started=time.monotonic();done=0;verified_roundtrip=False
    with torch.inference_mode():
        for shard_index,start in enumerate(range(0,len(records),500)):
            if shard_index%args.worker_count!=args.worker_index:continue
            path=args.output/f'shard_{start:06d}.pt'
            expected=records[start:start+500]
            if path.exists():
                saved=torch.load(path,weights_only=True)
                assert saved['codec_sha256']==inventory['codec_sha256']
                assert [r['path'] for r in saved['records']]==[r['path'] for r in expected]
                done+=len(expected);continue
            batch=[]
            for record in expected:
                x,sr=sf.read(record['path'],dtype='float32');assert sr==48000 and len(x)==record['samples']
                padded,length=align_mdct(torch.from_numpy(x).to(args.device)[None,None])
                latent,_,sparse=model.encode(padded)
                if rvq:
                    payload,parsed_ids=payload_roundtrip(sparse.support.squeeze(1).transpose(1,2).cpu().numpy())
                    codes=torch.from_numpy(parsed_ids[0].T.copy()).short()
                else:
                    integers=sparse.values.div(bound/63).round().long()
                    payload=pack_frames(sparse.support.cpu().numpy(),integers.cpu().numpy())
                    support,values=unpack_frames(payload)
                    codes=torch.empty((len(support),4),dtype=torch.int16)
                    codes[:,0::2]=torch.from_numpy(support).short()
                    codes[:,1::2]=torch.from_numpy(values+63).short()
                assert len(codes)==math.ceil((len(x)/40+1)/8)
                if not rvq:assert torch.all(codes[:,0]!=codes[:,2])
                if not verified_roundtrip:
                    if rvq:
                        direct=model.decode(latent)
                        parsed=model.decoder(model.bottleneck.quantizer.from_codes(torch.from_numpy(parsed_ids).to(args.device))[0])
                    else:
                        direct=model.decode_from_atoms_and_coeffs(sparse.support,sparse.values)
                        parsed=model.decode_from_atoms_and_coeffs(torch.from_numpy(support).to(args.device).reshape_as(sparse.support),
                            torch.from_numpy(values).to(args.device).reshape_as(sparse.values).float()*(bound/63))
                    torch.testing.assert_close(parsed,direct,rtol=1e-5,atol=1e-5)
                    sf.write(args.output/f'roundtrip_worker{args.worker_index}.wav',parsed[0,0,:length].cpu().numpy(),48000,subtype='FLOAT')
                    write(args.output/f'roundtrip_worker{args.worker_index}.json',{'source':record['path'],'frames':len(codes),
                        'payload_bytes':len(payload),'bound':bound,'serialized_decode_matches':True})
                    verified_roundtrip=True
                batch.append({**record,'codes':codes})
            temporary=path.with_suffix('.tmp')
            torch.save({'codec_sha256':inventory['codec_sha256'],'records':batch},temporary);temporary.replace(path)
            done+=len(batch)
            print('CACHE_PROGRESS',json.dumps({'worker':args.worker_index,'utterances':done,
                'last_shard':start,'seconds':round(time.monotonic()-started,1)}),flush=True)
    write(args.output/f'worker_{args.worker_index}_complete.json',{'records':done,'codec_sha256':inventory['codec_sha256']})


def merge(args):
    inventory=read(args.output/'inventory.json');records=[]
    for start in range(0,len(inventory['records']),500):
        saved=torch.load(args.output/f'shard_{start:06d}.pt',map_location='cpu',weights_only=True)
        assert saved['codec_sha256']==inventory['codec_sha256']
        expected=inventory['records'][start:start+500]
        assert [{k:v for k,v in r.items() if k!='codes'} for r in saved['records']]==expected
        for record in saved['records']:
            codes=record['codes']
            assert codes.dtype==torch.int16 and codes.ndim==2 and codes.shape[1]==4
            if inventory['codec'].get('type')=='rvq':
                assert ((codes>=0)&(codes<1024)).all()
            else:
                assert ((codes[:,0::2]>=0)&(codes[:,0::2]<8192)).all()
                assert ((codes[:,1::2]>=0)&(codes[:,1::2]<127)).all()
                assert (codes[:,0]!=codes[:,2]).all()
        records.extend(saved['records'])
    temporary=args.output/'tokens.tmp';torch.save({**inventory,'records':records},temporary)
    temporary.replace(args.output/'tokens.pt')
    write(args.output/'complete.json',{'utterances':len(records),'codec_sha256':inventory['codec_sha256'],
        'token_cache':str((args.output/'tokens.pt').resolve()),'token_cache_sha256':sha(args.output/'tokens.pt'),
        'counts':inventory['counts'],'coefficient_max':inventory['codec']['coefficient_max']})
    print('CACHE_COMPLETE',json.dumps(read(args.output/'complete.json')),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prepare',action='store_true');p.add_argument('--merge',action='store_true')
    p.add_argument('--inventory-from',type=Path);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--selection',default='Highest stage1 validation ViSQOL checkpoint')
    p.add_argument('--device',default='cuda:0');p.add_argument('--worker-index',type=int,default=0)
    p.add_argument('--worker-count',type=int,default=1)
    args=p.parse_args()
    if args.prepare:prepare(args)
    elif args.merge:merge(args)
    else:worker(args)
