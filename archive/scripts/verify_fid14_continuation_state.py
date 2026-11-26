#!/usr/bin/env python3
"""Verify all historical weights/moments and the exact resized data cursor."""
import hashlib
import json
import os
from pathlib import Path
import sys

BASE=Path('/tmp/laser-fid14-continue100-20261016')
OUT=Path('/workspace/Projects/laser/outputs/imagenet-fid14-continue100-4h100-20261016')
sys.path[:0]=[str(BASE/'frozen/runtime'),str(BASE/'frozen/support'),str(BASE/'support')]
import torch
from src.training import rqtransformer as training
from src.training.full_resume_upload import recovery_metadata
from portable_imagenet_recovery import remap_payload
from refinement_schedule import rate


def main():
    torch.set_num_threads(4)
    restored=json.loads((OUT/'source-restoration.json').read_text())
    checkpoint=Path(restored['local'])
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    metadata=recovery_metadata(payload)
    assert metadata['global_step']==48747 and metadata['adam_step']==14317
    assert metadata['adam_parameters']==782 and metadata['rng_ranks']==4
    assert metadata['epoch']==77 and metadata['next_microbatch']==1090
    assert payload['scheduler']['last_epoch']==48747
    assert all(abs(lr-rate(48747)*payload['scheduler']['multiplier'])<1e-15
               for lr in metadata['saved_learning_rates'])
    with torch.device('meta'):
        model=training.build_model(18432,16384,compound=False,physical_pair_context=True,
                                   sparsity_level=4,model_preset='imagenet-1400m')
    model.load_state_dict(payload['state_dict'],strict=True,assign=True)
    assert list(dict(model.named_parameters()))==list(payload['state_dict'])
    indices=[i for group in payload['optimizer']['param_groups'] for i in group['params']]
    assert len(indices)==len(payload['optimizer']['state'])==782
    for name,index in zip(payload['state_dict'],indices):
        value=payload['state_dict'][name]
        state=payload['optimizer']['state'][index]
        assert torch.isfinite(value).all(),name
        assert int(state['step'])==14317
        assert all(state[key].shape==value.shape and torch.isfinite(state[key]).all()
                   for key in ('exp_avg','exp_avg_sq')),name
    copy=dict(payload,best_fid=[],best_inception=[])
    remapped=remap_payload(copy,world=4,accumulation=8,checkpoint_dir=BASE/'unused',
        reference=BASE/'inputs/imagenet_256_train.npz',
        reference_sha256='3f9c92d15755e76ec312964a819e3e19b9cf3aadc618a7ae99f5e8aa96501260')
    assert remapped['epoch']==77 and remapped['batch_idx']==4360
    assert remapped['global_step']==48747
    assert all(remapped[key] is payload[key] for key in ('state_dict','optimizer','scheduler','rng_state_by_rank'))
    def sha(path):
        with path.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()
    stage1=sha(BASE/'inputs/resume-stage1-tokenizer.pt')
    reference=sha(BASE/'inputs/imagenet_256_train.npz')
    assert stage1=='dd28db9306d526bdc9fbf8016403e106c4f317fd8f5c04792f0953e53310fdab'
    assert reference=='3f9c92d15755e76ec312964a819e3e19b9cf3aadc618a7ae99f5e8aa96501260'
    target=BASE/'inputs/resume-source.pt'
    if not target.exists():os.link(checkpoint,target)
    record=dict(passed=True,global_step=48747,epoch=77,next_microbatch=4360,
        completed_updates_in_epoch=545,source_accumulation=2,new_accumulation=8,
        all_adam_ages=14317,optimizer_states=782,
        parameter_count=sum(p.numel() for p in model.parameters()),strict_model_load=True,
        weights_optimizer_scheduler_rng_objects_unchanged=True,
        all_model_and_moment_values_finite=True,saved_learning_rate=metadata['saved_learning_rates'],
        scheduler=payload['scheduler'],stage1_sha256=stage1,training_reference_sha256=reference,
        checkpoint_sha256=restored['sha256'])
    (OUT/'checkpoint-verification.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(passed=True,global_step=48747,all_adam_ages=14317,strict_model_load=True)),flush=True)


if __name__=='__main__':main()
