"""Complete valid scalar controls and the corrected historical pair cases."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path('/workspace/Projects/laser')
BASE=Path('/tmp/laser-pair-ablation-20261015')
RAW=Path('/tmp/laser-pair-ablation-rawid-20261015')
OUT=ROOT/'outputs/imagenet-laser-pair-ablation-20261015'
child=None
cancelled=False


def record(**status):
    temporary=OUT/'controller.tmp'
    temporary.write_text(json.dumps(dict(status,pid=os.getpid(),time=time.time(),revision='raw_atom_identity'),indent=2)+'\n')
    temporary.replace(OUT/'controller.json')


def stop(*_):
    global cancelled
    cancelled=True
    if child is not None and child.poll() is None:
        child.send_signal(signal.SIGINT)


def launch(label, coefficients, *, pair=False, fid=False):
    global child
    destination=OUT/label/('fid50k.json' if fid else 'result.json')
    if destination.exists():return
    if cancelled:raise KeyboardInterrupt
    folder=RAW if pair else BASE
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=4',
             str(folder/'trial.py'),'--decoder','pair' if pair else 'scalar',
             '--coefficients',coefficients,'--updates','128']
    if pair:command.extend(['--label',label])
    if fid:command.append('--fid-only')
    environment=dict(os.environ,LASER_PAIR_ABLATION_BASE=str(folder),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    log_path=folder/(label+('-fid50k' if fid else '')+'.log')
    with log_path.open('a') as log:
        child=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
        record(phase='fid50k' if fid else 'training',branch=label,child_pid=child.pid,log=str(log_path),command=command)
        code=child.wait()
    if code:
        record(phase='failed',branch=label,returncode=code,log=str(log_path))
        raise RuntimeError(str(log_path))
    assert destination.exists()


def main():
    lock=(RAW/'controller.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGINT,stop)
    signal.signal(signal.SIGTERM,stop)
    correction=json.loads((OUT/'prototype-identity-correction.json').read_text())
    active=correction['active_control']
    record(phase='waiting_for_active_control_fid',branch=active,active_child=correction['active_child'])
    while not (OUT/active/'fid50k.json').exists():
        if cancelled:return
        time.sleep(4)
    while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        if cancelled:return
        time.sleep(4)
    # The old queue was stopped without interrupting its live FID child.
    try:os.kill(correction['controller_paused'],signal.SIGKILL)
    except ProcessLookupError:pass
    for coefficients in ['hard','soft']:
        label='pair-raw-'+coefficients
        launch(label,coefficients,pair=True)
        launch(label,coefficients,pair=True,fid=True)
    launch('scalar-soft','soft',fid=True)
    selected=['scalar-hard','scalar-soft','pair-raw-hard','pair-raw-soft']
    cases={n:json.loads((OUT/n/'result.json').read_text()) for n in selected}
    assert all(c['same_data_order_sha256_by_rank']==cases['scalar-hard']['same_data_order_sha256_by_rank'] for c in cases.values())
    results={n:json.loads((OUT/n/'fid50k.json').read_text()) for n in selected}
    winner=min(results,key=lambda n:results[n]['fid'])
    gain=results['scalar-hard']['fid']-results[winner]['fid']
    summary=dict(passed=True,selected=selected,fid={n:r['fid'] for n,r in results.items()},best_branch=winner,
        improvement_over_matched_control=gain,qualifies_for_adoption=winner!='scalar-hard' and gain>=1.,
        generated_images_per_case=50000,reference_images=1281167,source_step=3756,tested_step=3884,
        matched_additional_updates=128,auxiliary_loss_terms=0,omp_trajectory_banks=0,
        corrected_pair_identity='raw_dictionary_atom',superseded_prototypes=['pair-hard','pair-soft'],production_replaced=False)
    (OUT/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
    subprocess.run([sys.executable,str(ROOT/'scripts/tools/analyze_laser_pair_ablation.py')],cwd=ROOT,check=True)
    record(phase='complete_awaiting_adoption',**summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
