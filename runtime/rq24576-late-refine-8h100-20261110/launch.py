"""Select a verified late source, install, validate, then start production."""
import json
import math
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
BASE=Path('/tmp/laser-rq24576-late-refine-8h100-20261110')
RUN='imagenet-rq24576-late-lowLR-noise-ema-8h100-20261110'
OUT=ROOT/'outputs'/RUN
ORIGINAL='imagenet-rq24576-native-scratch100-8h100-20261108'
EPOCH100=ROOT/'outputs/imagenet-rq24576-late-select-epoch100-20261110/report.json'


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_suffix('.pending');pending.write_text(json.dumps(value,indent=2)+'\n');pending.replace(path)


def main():
    OUT.mkdir(exist_ok=True);BASE.mkdir(exist_ok=True)
    write(OUT/'launch-pipeline.json',dict(phase='waiting_for_epoch100_sampling',time=time.time()))
    deadline=time.monotonic()+1200
    while not EPOCH100.exists():
        assert time.monotonic()<deadline,'Epoch100 sampler selection did not finish'
        time.sleep(5)
    comparison=json.loads(EPOCH100.read_text());assert comparison['frozen_checkpoint_verified']
    old=json.loads((ROOT/'outputs/imagenet-rq24576-sampler-expanded-epoch94-images-20261110/report.json').read_text())
    strongest94=old['best_observed_single_seed']
    assert strongest94['source_epoch']==94 and strongest94['seed']==261001
    candidate100=comparison['best_fid']
    chosen=candidate100 if candidate100['fid']<strongest94['fid']-.05 else strongest94
    epoch=chosen['source_epoch'];step=epoch*626
    source=(Path('/tmp/laser-rq24576-late-select-epoch100-20261110/frozen-checkpoint.pt') if epoch==100 else
        Path('/tmp/laser-rq24576-sampler-search-epoch94-images-20261110/frozen-checkpoint.pt'))
    selection=dict(selected_epoch=epoch,selected_fid=chosen['fid'],epoch94=strongest94,
        epoch100_results=comparison['results'],epoch100_min_fid_advantage_required=.05,
        reason='Use epoch100 only if it improves FID by at least 0.05; epoch94 has independent seed confirmations.')
    write(OUT/'source-selection.json',selection)
    original_lr=.0005*.5*(1.+math.cos(math.pi*step/62600))
    plan=dict(run_id=RUN,name=f'ImageNet | RQ24576 late{epoch} lowLR noise anneal EMA | 8H100',
        parent_run_id=ORIGINAL,source_checkpoint=str(source),source_checkpoint_sha256=chosen['checkpoint_sha256'],
        source_epoch=epoch,source_step=step,source_selection_pending=False,target_epoch=epoch+30,
        base=str(BASE),output=str(OUT),world_size=8,microbatch=64,global_batch=2048,
        final_codewords=24576,tokenizer_changed=False,architecture_changed=False,
        source_stage1_sha256='dd28db9306d526bdc9fbf8016403e106c4f317fd8f5c04792f0953e53310fdab',
        source_codebook_sha256='8b3f295593b3193d28991554f7a16d629074ec6ce3072a477fb2772e484d86de',
        source_recipe_sha256='4bd51b2da94d481ad988995d48448148fc32e3dcb3495cfd385f49def4312747',
        lr=dict(kind='one_epoch_ramp_then_cosine',start_step=step,start_lr=original_lr,
            ramp_end_step=step+626,end_step=step+30*626,peak_lr=1e-5,min_lr=1e-6,
            original_horizon_steps=62600,preserve_optimizer_state=True,preserve_optimizer_age=True),
        teacher=dict(initial=.5,final=.25,start_step=step,anneal_steps=1252,stochastic_prefixes=True,full_vocabulary=True),
        ema_decay=.999,sampler=chosen['sampling'],baseline=dict(fid=chosen['fid'],
            inception_score=chosen['inception_score'],inception_score_std=chosen['inception_score_std'],sampler=chosen['sampling']),
        evaluation=dict(generated_images=50000,reference_images=1281167,reference_split='train',
            reference_sha256='3f9c92d15755e76ec312964a819e3e19b9cf3aadc618a7ae99f5e8aa96501260',
            seed=261001,is_shuffle_seed=261002,rejection_sampling=False,target_fid=11.56,
            ema_every_epochs=2,raw_every_epochs=4,raw_also_first_evaluation=True,log_images=True),created_at=time.time())
    write(HERE/'plan.json',plan)
    for script in ['install.py','check_refinement.py','validate.py']:
        write(OUT/'launch-pipeline.json',dict(phase=script,source_epoch=epoch,time=time.time()))
        subprocess.run([sys.executable,'-u',str(HERE/script)],check=True)
    # Validation children have exited; production starts from the original parent.
    with (OUT/'supervisor.log').open('ab',buffering=0) as stream:
        child=subprocess.Popen([sys.executable,'-u',str(HERE/'supervise.py')],stdin=subprocess.DEVNULL,
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write(OUT/'launch-pipeline.json',dict(phase='production_supervisor_launched',supervisor_pid=child.pid,
        source_epoch=epoch,target_epoch=epoch+30,time=time.time()))
    print(json.dumps(dict(production_supervisor_pid=child.pid,run_id=RUN,source_epoch=epoch)),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        write(OUT/'launch-pipeline.json',dict(phase='failed',error=str(error),error_type=type(error).__name__,time=time.time()))
        raise
