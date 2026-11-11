"""Prepare the fresh mixture pilot while preserving the existing control."""
import json
from pathlib import Path
import shutil
from omegaconf import OmegaConf

SOURCE=Path('/tmp/laser-aligned-beam451-20261105/frozen')
ROOT=Path('/tmp/laser-native-beam-mixture-20261107')
BASE=ROOT/'frozen-mixture'
RUN='imnet-j16384-a4096-native-beam-mix-scratch4-4a100-20261107'
PUBLIC=Path('/workspace/Projects/laser/outputs')/RUN


def main():
    assert not BASE.exists(),'Inspect the prepared pilot before recreating it'
    BASE.mkdir(parents=True);(ROOT/'quality').mkdir(exist_ok=True);(ROOT/'production-nvme').mkdir(exist_ok=True)
    PUBLIC.mkdir(parents=True,exist_ok=True)
    for n in ['source','upstream','runtime']:
        shutil.copytree(SOURCE/n,BASE/n,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.caches'))
    (BASE/'inputs').mkdir()
    for n in ['stage1-model.pt','imagenet_256_train.npz','joint-codebook.pt','reconstruction-codebook.pt','rq-upstream-core.json']:
        (BASE/'inputs'/n).hardlink_to(SOURCE/'inputs'/n)
    for n in ['stage2-training-config.yaml','stage2-archive-config.yaml','paired-tokenizer-artifact.json']:
        shutil.copyfile(SOURCE/'inputs'/n,BASE/'inputs'/n)
    (BASE/'data').symlink_to(SOURCE/'data',target_is_directory=True)
    (ROOT/'torch-cache').symlink_to(SOURCE.parent/'torch-cache',target_is_directory=True)
    for p in Path(__file__).parent.glob('*.py'):shutil.copyfile(p,BASE/'runtime'/p.name)
    recipe=OmegaConf.load(SOURCE/'inputs/recipe.yaml')
    recipe.run_id=RUN;recipe.name='ImageNet4096 | native-soft and beam-best mixture | fresh4'
    recipe.version='native-beam-bayes-mixture-v1';recipe.epochs=4;recipe.microbatch=128
    recipe.targets=dict(encoding='native-soft/beam-best complete-sequence mixture',native_weight=.25,
        native_temperature=.5,stochastic_prefixes=True,chunk_size=256,measured_entropy=float('nan'),
        full_vocabulary_native_component=True,finite_beam_support_limit=False,near_best_error_mask=False,
        calibration='pending train screen and validation gate')
    recipe.tokenizer.teacher_sampled_rfid=float('nan')
    recipe.lineage=dict(control_run='imnet-j16384-a4096-beam16-aligned-soft-scratch2-4a100-20261105',
        control_checkpoint_step=17250,source_checkpoint_loaded=False,
        earlier_native_run='imnet-j16384-a4096-rfid4512-sampler1024p095-cont100-2b300-20261031',
        earlier_native_epoch4_fid=54.1376692178589,current_beam_epoch4_fid=60.662829012099394,
        experiment='restore native teacher uncertainty while retaining beam reconstruction quality',
        comparison_epochs=[1,2,4])
    OmegaConf.save(recipe,BASE/'inputs/recipe.yaml')
    execution=json.loads((SOURCE/'inputs/execution.json').read_text())
    execution.update(fresh_stage2=True,fresh_optimizer=True,model_and_optimizer_restored=False,
        training_stop_epoch=4,target_step=2504,continuation_from_step=None,continuation_from_epoch=None,
        training_targets='full-support native-soft/beam-best sequence mixture',
        microbatch=128,accumulation_steps=4)
    for n in ['teacher_beam_width','beam_candidates_and_probabilities_bitwise_preserved']:
        execution.pop(n,None)
    (BASE/'inputs/execution.json').write_text(json.dumps(execution,indent=2)+'\n')
    shutil.copyfile(SOURCE/'train.py',BASE/'train.py')
    p=BASE/'train.py';p.write_text(p.read_text().replace("main('laser_aligned_beam',root)","main('laser_mixture',root)"))
    p=BASE/'source/src/training/rq_imagenet_shared.py';code=p.read_text()
    code=code.replace("    if kind == 'laser_aligned_beam':", "    if kind == 'laser_mixture':\n        from runtime.mixture_recipe import MixtureRecipe\n        PROFILE=MixtureRecipe(BASE)\n    elif kind == 'laser_aligned_beam':")
    marker='    parameters=sum(v.numel() for v in model.parameters())'
    replacement="""    if not args.resume and not args.smoke_updates:
        if rank==0:
            import hashlib
            fingerprint=hashlib.sha256()
            for name,value in model.state_dict().items():
                fingerprint.update(name.encode());fingerprint.update(value.detach().cpu().contiguous().numpy().tobytes())
            write_json(folder/'initial-model-sha256.json',dict(sha256=fingerprint.hexdigest(),seed=261001))
        dist.barrier()
"""+marker
    assert code.count(marker)==1;code=code.replace(marker,replacement)
    compile(code,str(p),'exec');p.write_text(code)
    p=BASE/'source/src/training/joint_runtime_support.py'
    p.write_text(p.read_text().replace('Beam451 aligned soft teacher 100 epochs 4A100 20261105',
                                      'Native beam mixture controlled four epoch pilot 20261107'))
    for n in ['status.json','launch.json','training.log','checkpoint.json','cloud-sync-status.json','evaluations','source-artifact.json','wandb.json']:
        (PUBLIC/n).symlink_to(ROOT/'production-nvme'/n,target_is_directory=n=='evaluations')
    (PUBLIC/'supervisor.log').symlink_to(ROOT/'supervisor.log')
    (PUBLIC/'quality').symlink_to(ROOT/'quality',target_is_directory=True)
    (PUBLIC/'live-nvme').symlink_to(ROOT/'production-nvme',target_is_directory=True)
    print(json.dumps(dict(base=str(BASE),run_id=RUN,pilot_epochs=4)),flush=True)


if __name__=='__main__':main()
