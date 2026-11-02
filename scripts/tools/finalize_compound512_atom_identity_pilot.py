#!/usr/bin/env python3
"""Publish the completed paired result without restarting either training run."""
import json
import os
from pathlib import Path
import shutil
import time

OUT=Path('/workspace/Projects/laser/outputs/imagenet-compound512-atom-identity-pilot-20261016')
RUNS={'control':'imagenet-compound512-identity-control128-4h100-20261016',
      'atom_identity':'imagenet-compound512-atom-identity128-4h100-20261016'}


def main():
    import wandb
    comparison=json.loads((OUT/'comparison.json').read_text())
    assert comparison['completed']
    gate=comparison['gate']
    os.environ['WANDB_API_KEY']=Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip()
    api=wandb.Api(timeout=60)
    updates={
        'train/epoch':1380/626,
        'pilot/checkpoint_global_step':1380,
        'pilot/checkpoint_epochs':1380/626,
        'trial/status':'completed_fid_evaluated' if comparison['fid_evaluated'] else 'completed_gate_rejected',
        'pilot/gate_passed':gate['qualifies_for_fid'],
        'pilot/matched_training_hashes':gate['matched_training_hashes'],
        'pilot/matched_initial_predictions':gate['matched_initial_predictions'],
        'pilot/heldout_images':gate['images'],
        'pilot/atom_ce_candidate_minus_control':gate['atom_candidate_minus_control']['mean'],
        'pilot/atom_ce_delta_ci95_low':gate['atom_candidate_minus_control']['interval95'][0],
        'pilot/atom_ce_delta_ci95_high':gate['atom_candidate_minus_control']['interval95'][1],
        'pilot/joint_pair_ce_candidate_minus_control':gate['joint_pair_candidate_minus_control']['mean'],
        'pilot/fid_evaluated':comparison['fid_evaluated'],
        'pilot/automatic_adoption':False,
    }
    branches={}
    for branch,run_id in RUNS.items():
        receipt=json.loads((OUT/branch/'durable-checkpoint.json').read_text())
        integrity=json.loads((OUT/branch/'checkpoint-integrity.json').read_text())
        assert integrity['passed'] and integrity['full_payload_read'] and integrity['sha256']==receipt['sha256']
        target=Path(receipt['durable_path'])
        assert receipt['durable'] and target.is_file() and target.stat().st_size==receipt['bytes']
        run=api.run('helloimlixin-rutgers/laser/'+run_id)
        assert run.state=='finished' and int(run.summary['train/global_step'])==1380
        metrics_path=Path(receipt['local_path']).parent/'metrics.jsonl'
        shutil.copyfile(metrics_path,OUT/branch/'training-metrics.jsonl')
        metrics=[json.loads(row) for row in metrics_path.read_text().splitlines()]
        seconds=metrics[-1]['time']-metrics[0]['time']
        updates_measured=metrics[-1]['train/global_step']-metrics[0]['train/global_step']
        throughput=updates_measured*2048/seconds
        branch_updates={**updates,'pilot/wall_interval_images_per_second':throughput,
                        'pilot/checkpoint_full_sha256_verified':True}
        if comparison['fid_evaluated']:
            fid=json.loads((OUT/branch/'fid50k.json').read_text())
            assert fid['generated_images']==50000 and fid['real_images']==1281167
            for attempt in range(12):
                history=run.history(samples=100,pandas=False)
                points=[row for row in history if row.get('eval/fid') is not None and row.get('train/global_step')==1380]
                if points:break
                time.sleep(5)
            assert points and abs(points[-1]['eval/fid']-fid['fid'])<1.e-10
            # Keep one FID field; min/last aliases and copied branch scores
            # duplicate the actual eval/fid history series in W&B.
            branch_updates['eval/fid']=fid['fid']
            for key in ['eval/fid.last','eval/fid.min','pilot/fid_control','pilot/fid_candidate']:
                if key in run.summary:del run.summary[key]
            if branch=='atom_identity':branch_updates['pilot/fid_candidate_minus_control']=comparison['fid']['delta']
            elif 'pilot/fid_candidate_minus_control' in run.summary:del run.summary['pilot/fid_candidate_minus_control']
        run.summary.update(branch_updates)
        branches[branch]=dict(url=run.url,state=run.state,global_step=1380,
            durable_checkpoint=str(target),checkpoint_sha256=receipt['sha256'],
            checkpoint_bytes=receipt['bytes'],checkpoint_full_sha256_verified=True,summary_updated=True,
            elapsed_update_intervals=updates_measured,elapsed_update_seconds=seconds,
            elapsed_images_per_second=throughput)
        if comparison['fid_evaluated']:
            branches[branch].update(fid=fid['fid'],fid_history_verified=True,
                fid_history_global_step=points[-1]['train/global_step'])
    result=dict(passed=True,completed=True,branches=branches,gate_passed=gate['qualifies_for_fid'],
                fid_evaluated=comparison['fid_evaluated'],time=time.time())
    probe_folder=OUT/'probe-targets';probe_folder.mkdir(exist_ok=True)
    for rank in range(4):
        shutil.copyfile(Path('/tmp/laser-compound512-atom-identity-pilot-20261016')/f'probe/rank{rank}.pt',
                        probe_folder/f'rank{rank}.pt')
    shutil.copyfile(Path(__file__),OUT/'reproducibility/finalize_compound512_atom_identity_pilot.py')
    (OUT/'final-online-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
