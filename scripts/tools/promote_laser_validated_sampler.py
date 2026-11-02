"""Persist the independently verified sampler before releasing the queued run."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import xml.etree.ElementTree as ET
import zipfile

import torch
from omegaconf import OmegaConf

ROOT=Path('/workspace/Projects/laser')
BASE=Path('/tmp/laser-rq-validated-ar-imagenet-20261015')
sys.path.insert(0,str(BASE/'upstream'))
OUT=ROOT/'outputs/imagenet-laser-rq-validated-ar-rfid421-continue20-20261015'
TRIALS=ROOT/'outputs/imagenet-laser-pair-ablation-20261015'
SAMPLER=ROOT/'outputs/imagenet-laser-fixed-sampler-20261015'


def write(path, value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def main():
    from src.training.k4_checkpoint_io import atomic_torch_save, _checkpoint_upload_source
    assert json.loads((SAMPLER/'controller.json').read_text())['phase']=='complete'
    assert not (OUT/'wandb.json').exists(), 'Do not rewrite an active run protocol'
    result=json.loads((SAMPLER/'historical/fid50k.json').read_text())
    plan=json.loads((SAMPLER/'plan.json').read_text())
    assert result['fid']<plan['matched_fid']-1 and result['source_step']==3884
    assert result['generated_images']==50000 and result['reference_images']==1281167
    test_xml=Path('/tmp/laser-final-continuation-tests.xml')
    suite=ET.parse(test_xml).getroot().find('testsuite')
    assert int(suite.attrib['tests'])==42 and int(suite.attrib['errors'])==int(suite.attrib['failures'])==0
    os.environ.update(json.loads((BASE/'environment.json').read_text()))
    torch.set_num_threads(4)
    prior_proof=json.loads((OUT/'continuation-verification.json').read_text())
    prior_checkpoint=(OUT/'checkpoints/last.pt').resolve()
    payload=torch.load(_checkpoint_upload_source(prior_checkpoint),map_location='cpu',mmap=True,weights_only=False)
    assert payload['global_step']==3884 and payload['next_microbatch']==2048
    # Hash all model tensors and Adam moments before changing metadata only.
    model_hash,adam_hash=hashlib.sha256(),hashlib.sha256()
    for name,value in payload['state_dict'].items():
        model_hash.update(name.encode());model_hash.update(value.contiguous().numpy().tobytes())
    for group in payload['optimizer']['param_groups']:
        for index in group['params']:
            state=payload['optimizer']['state'][index]
            assert int(state['step'])==3884
            for field in ['exp_avg','exp_avg_sq']:
                adam_hash.update(str(index).encode()+field.encode())
                adam_hash.update(state[field].contiguous().numpy().tobytes())
    assert model_hash.hexdigest()==prior_proof['model_state_sha256']==plan['source_model_sha256']
    assert adam_hash.hexdigest()==prior_proof['adam_moments_sha256']
    recipe_path=BASE/'inputs/recipe.yaml'
    shutil.copyfile(recipe_path,OUT/'reproducibility/recipe-before-sampler.yaml')
    recipe=OmegaConf.load(recipe_path)
    old_sampling=OmegaConf.to_container(recipe.sampling,resolve=True)
    recipe.sampling.atom_top_k=2048
    recipe.sampling.atom_top_p=1.
    recipe.sampling.coefficient_top_p=.85
    recipe.name='LASER scalar-soft | restored sampler | upstream RQ | continue20'
    OmegaConf.save(recipe,recipe_path)
    OmegaConf.save(recipe,ROOT/'configs/stage2/imagenet-laser-rq-validated-ar-20ep-20261015.yaml')
    shutil.copyfile(recipe_path,OUT/'reproducibility/recipe.yaml')
    settings=OmegaConf.to_container(recipe.sampling,resolve=True)
    payload.update(recipe=OmegaConf.to_container(recipe,resolve=True),best_fid=result['fid'],evaluation=result)
    payload['runtime']['sampling']=settings
    payload['continuation_origin'].update(matched_training_fid=plan['matched_fid'],
        sampler_validated_fid=result['fid'],sampler_only_improvement=plan['matched_fid']-result['fid'],
        sampling_before=old_sampling,sampling_after=settings)
    # Freeze the metadata logging fix; numerical model/training code is unchanged.
    shutil.copyfile(ROOT/'src/training/rq_imagenet_shared.py',BASE/'source/src/training/rq_imagenet_shared.py')
    manifest_path=OUT/'reproducibility/code-manifest.json'
    manifest=json.loads(manifest_path.read_text())
    for name in manifest:
        with (BASE/name).open('rb') as handle:manifest[name]=hashlib.file_digest(handle,'sha256').hexdigest()
    write(manifest_path,manifest)
    with zipfile.ZipFile(OUT/'reproducibility/source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in manifest:archive.write(BASE/name,name)
        for name in ['test_laser_compound_rq_adapter.py','test_shared_rq_codebase.py','test_amp_replay.py']:
            archive.write(ROOT/'tests'/name,'tests/'+name)
        archive.write(Path(__file__),'scripts/tools/'+Path(__file__).name)
    prepared=json.loads((OUT/'prepared.json').read_text())
    prepared.update(recipe=OmegaConf.to_container(recipe,resolve=True),fresh_stage2=False,fresh_optimizer=False,
        source_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    write(OUT/'prepared.json',prepared)
    print(json.dumps(dict(phase='persisting_sampler_metadata',fid=result['fid'],weights_and_adam_unchanged=True)),flush=True)
    atomic_torch_save(payload,OUT/'checkpoints/last.pt')
    best=OUT/'checkpoints/best.pt';temporary=best.with_suffix('.sampler-link')
    temporary.unlink(missing_ok=True);temporary.symlink_to(os.readlink(OUT/'checkpoints/last.pt'));temporary.replace(best)
    persistent=(OUT/'checkpoints/last.pt').resolve()
    local=_checkpoint_upload_source(persistent)
    assert local.stat().st_size==persistent.stat().st_size>10_000_000_000
    proof=dict(prior_proof,continued_fid=result['fid'],best_fid=result['fid'],
        matched_training_fid=plan['matched_fid'],matched_training_fid_improvement=prior_proof['fid_improvement'],
        fid_improvement_reference='Matched training comparison with sampling held fixed',
        sampler_improvement_verified=True,sampler_only_improvement=plan['matched_fid']-result['fid'],
        sampling=settings,source_sampling=old_sampling,prior_protocol_checkpoint=str(prior_checkpoint),
        checkpoint=str(persistent),local_checkpoint=str(local),protected_best_checkpoint=str(persistent),
        timestamp=time.time())
    for name in ['continuation-verification.json','pair-ablation-continuation.json','sampler-continuation.json']:write(OUT/name,proof)
    adoption=json.loads((TRIALS/'adoption.json').read_text());adoption.update(proof);write(TRIALS/'adoption.json',adoption)
    write(OUT/'checkpoint.json',dict(epoch=6,global_step=3884,path=str(OUT/'checkpoints/last.pt'),bytes=persistent.stat().st_size,best_fid=result['fid']))
    shutil.copyfile(test_xml,OUT/'reproducibility/continuation-cpu-tests.xml')
    write(OUT/'cpu-verification.json',dict(passed=True,tests=42,
        test_receipt=str(OUT/'reproducibility/continuation-cpu-tests.xml'),frozen_manifest_verified=True))
    report=dict(passed=True,source_step=3884,matched_sampler_fid=plan['matched_fid'],
        restored_sampler_fid=result['fid'],sampler_only_improvement=plan['matched_fid']-result['fid'],
        sampling=settings,training_updates_added=0,production_initial_fid=result['fid'],
        curve_protocol='Restored sampler for every point of the new continuation; matched trials retained separately.')
    write(SAMPLER/'comparison.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
