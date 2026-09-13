#!/usr/bin/env python3
"""Reuse verified frozen encoder latents and calibrate the compact RQ targets."""
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
UPSTREAM=ROOT/'outputs/church-rq-baseline-scratch-20260912/upstream-source'
sys.path[:0]=[str(UPSTREAM),str(ROOT)]
import numpy as np
import torch
from src.compact_rq_training import FrozenCompactTokenizer
from src.original_rq_training import atomic_json,file_sha256,state_sha256
from calibrate_scaled_atom_temperature import measure

BASE=ROOT/'outputs/church-compact-rq-stage2-20260913'
OLD=ROOT/'outputs/church-scaled-atom-stage2-20260913'
BOOK=ROOT/'outputs/church-compact-scaled-rq-20260913/adaptive2-8passes/compact-codebook.pt'


@torch.inference_mode()
def main():
    started=time.time()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    BASE.mkdir(parents=True,exist_ok=True)
    destination=BASE/'cache'
    destination.mkdir(exist_ok=False)
    manifest=OLD/'cache/complete.json'
    original=json.loads(manifest.read_text())
    assert original['images']==126227 and original['shape']==[126227,8,8,256]
    assert original['frozen_state_unchanged'] and not original['hard_codes_cached']
    assert original['checkpoint_sha256']=='93f19adc8ca2cfae0cf629d48d2d6f5f7039440b1a48ad1c7ba43cf17932a388'
    assert original['cache_sha256']=='cad69321547c726feebedac84f9d9c2a231b54f85e4066884393bc1ec08b9312'
    for filename,digest in ((original['latent_cache'],original['cache_sha256']),
                            (original['checkpoint'],original['checkpoint_sha256']),
                            (BOOK,'ed74fc4672bb8609d72cbc7797b90119846398d5cb8aa38ef8f8bf4694e68cae')):
        assert file_sha256(filename)==digest,str(filename)
    tokenizer=FrozenCompactTokenizer(original['checkpoint'],BOOK).to('cuda:0').eval()
    frozen_hash=state_sha256(tokenizer)
    assert tokenizer.quantizer.vocab_size==32769 and list(tokenizer.code_shape)==[8,8,4]
    reference_path=OLD/'temperature-calibration.json'
    reference=json.loads(reference_path.read_text())
    assert reference['calibration_indices']==list(range(62048,62176))
    assert reference['source_checkpoint_sha256']==original['checkpoint_sha256']
    array=np.load(original['latent_cache'],mmap_mode='r')
    latents=torch.from_numpy(array[62048:62176].copy()).to('cuda:0')
    control=reference['original_rq_control']
    limit=control['sampled_to_hard_residual_mse_ratio']+.02
    results=[]
    for temperature in [.5,.25,.125,.0625,.03125,.015625]:
        row=measure(tokenizer.quantizer,latents,temperature,99200)
        results.append(row)
        print(json.dumps(dict(kind='compact_adaptive_rq2',**row)),flush=True)
        if row['sampled_to_hard_residual_mse_ratio']<=limit:
            break
    assert results[-1]['sampled_to_hard_residual_mse_ratio']<=limit
    assert state_sha256(tokenizer)==frozen_hash
    calibration=dict(selected_temperature=results[-1]['temperature'],
        calibration_indices=reference['calibration_indices'],images=128,
        selection_rule=reference['selection_rule'],original_rq_control=control,
        original_rq_control_reused=True,control_report=str(reference_path),
        control_report_sha256=file_sha256(reference_path),
        control_checkpoint=reference['control_checkpoint'],
        control_checkpoint_sha256=reference['control_checkpoint_sha256'],
        compact_rq_sweep=results,allowed_mse_ratio=limit,
        source_checkpoint_sha256=original['checkpoint_sha256'],codebook_sha256=file_sha256(BOOK),
        precision='FP32; TF32 disabled',seed=99200,
        noise='Exact RQ stochastic codeword sampling; no additive physical coefficient noise',
        elapsed_seconds=time.time()-started)
    atomic_json(BASE/'temperature-calibration.json',calibration)
    validation_hashes={}
    for rank in range(2):
        shard=OLD/f'cache/validation-rank{rank}.pt'
        (destination/shard.name).symlink_to(shard)
        validation_hashes[shard.name]=file_sha256(shard)
    prepared={**original,'codebook':str(BOOK),'codebook_sha256':file_sha256(BOOK),
        'levels':2,'levels_per_atom':2,'vocab_size':32769,
        'frozen_state_sha256':frozen_hash,'frozen_state_unchanged':True,
        'cache_reused':True,'cache_origin_manifest':str(manifest),
        'cache_origin_manifest_sha256':file_sha256(manifest),
        'cache_origin_tokenizer_state_sha256':original['frozen_state_sha256'],
        'cache_origin_codebook_sha256':original['codebook_sha256'],
        'reuse_reason':'Unquantized FP32 encoder outputs; identical frozen encoder checkpoint and transform',
        'validation_shard_sha256':validation_hashes,
        'elapsed_seconds':time.time()-started,'updated_unix':time.time()}
    atomic_json(destination/'complete.json',prepared)
    print(json.dumps(dict(phase='complete',selected_temperature=calibration['selected_temperature'],
        frozen_state_sha256=frozen_hash,elapsed_seconds=time.time()-started)),flush=True)


if __name__=='__main__':
    main()
