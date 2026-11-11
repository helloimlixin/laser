"""Benchmark identical beam arithmetic and resume the 100-epoch run."""
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('speed_optimizer',HERE/'optimize.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
BASE=m.BASE;PROD=m.PROD;OUT=m.OUT;SUPERVISOR=m.SUPERVISOR


def resume(selected):
    code=SUPERVISOR.read_text()
    for key,value in [('LASER_TRAIN_SDPA',int(selected['sdpa'])),
        ('LASER_COMPILE_NATIVE_LOSS',int(selected['compiled_loss'])),
        ('LASER_SDPA_MIN_SEQ',selected['min_sequence_for_sdpa'])]:
        code,n=re.subn(key+r"='[0-9]+'",key+"='"+str(value)+"'",code);assert n==1
    flags="LASER_EXACT_BEAM_KERNELS='"+str(int(selected.get('exact_beam',False)))+"', LASER_FAST_BEAM_TOPK='"+str(int(selected.get('hierarchical_topk',False)))+"', "
    code=code.replace("LASER_FUSED_ADAMW='1',","LASER_FUSED_ADAMW='1', "+flags)
    compile(code,str(SUPERVISOR),'exec');SUPERVISOR.write_text(code)
    shutil.copyfile(SUPERVISOR,BASE/'runtime/supervise.py')
    ledger=m.read(PROD/'execution-benchmarks.json');ledger.append(dict(selected,
        verification=m.read(OUT/selected['name']/'verification.json')))
    m.write(PROD/'execution-benchmarks.json',ledger)
    execution=m.read(BASE/'inputs/execution.json')
    execution.update(exact_fused_beam_cost=selected.get('exact_beam',False),
        exact_hierarchical_beam_topk=selected.get('hierarchical_topk',False),
        beam_candidates_and_probabilities_bitwise_preserved=True)
    m.write(BASE/'inputs/execution.json',execution)
    for name in ['benchmarks.json','exact-beam-parity.json']:
        shutil.copyfile(OUT/name,BASE/'inputs'/('speed-'+name))
    for p in HERE.glob('*.py'):shutil.copyfile(p,BASE/'runtime/speed'/p.name)
    with (PROD/'supervisor.log').open('ab',buffering=0) as log:
        p=subprocess.Popen([sys.executable,'-u',str(SUPERVISOR)],stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    m.write(OUT/'teacher-resumed.json',dict(supervisor_pid=p.pid,selected=selected,time=time.time(),target_epochs=100))


def main():
    assert m.read(OUT/'exact-beam-parity.json')['passed']
    current=m.read(OUT/'resumed.json')['selected']
    checkpoint=m.read(PROD/'checkpoint.json');source=Path(checkpoint['path']);step=checkpoint['global_step']
    status=m.read(PROD/'status.json');pid=status['supervisor_pid']
    assert b'aligned-beam451-20261105/supervise.py' in Path(f'/proc/{pid}/cmdline').read_bytes()
    rank_pids=[m.read(PROD/f'train/startup-rank{rank}.json')['pid'] for rank in range(4)]
    os.kill(pid,signal.SIGTERM)
    for _ in range(60):
        alive=[]
        for rank_pid in rank_pids:
            p=Path(f'/proc/{rank_pid}/stat')
            if p.exists() and p.read_text().split()[2]!='Z':alive.append(rank_pid)
        if not alive:break
        time.sleep(1)
    assert not alive,'Previous training ranks did not stop'
    fallback=dict(current,exact_beam=False,hierarchical_topk=False)
    selected=fallback
    native_env=m.env
    extra=dict(LASER_EXACT_BEAM_KERNELS='0',LASER_FAST_BEAM_TOPK='0')
    def environment():
        value=native_env();value.update(extra);return value
    m.env=environment
    rows=[]
    def bench(name,exact,topk,profile=False):
        extra.update(LASER_EXACT_BEAM_KERNELS=str(int(exact)),LASER_FAST_BEAM_TOPK=str(int(topk)))
        row=m.benchmark(name,current['microbatch'],source,step,
            sdpa=current['sdpa'],compiled=current['compiled_loss'],
            min_seq=current['min_sequence_for_sdpa'],profile=profile)
        row.update(exact_beam=exact,hierarchical_topk=topk)
        ledger=m.read(OUT/'benchmarks.json');ledger[-1]=row;m.write(OUT/'benchmarks.json',ledger)
        rows.append(row);return row
    try:
        baseline=bench('before-exact-beam',False,False,profile=True)
        assert baseline['passed'];selected=baseline
        bench('fused-exact-beam-cost',True,False)
        bench('fused-exact-beam-topk',True,True)
        best=max([r for r in rows if r['passed']],key=lambda r:r['images_per_second'])
        if best['images_per_second']>=baseline['images_per_second']*1.03:selected=best
        m.write(OUT/'teacher-selection.json',dict(selected=selected,baseline=baseline,
            speedup_vs_previous=selected['images_per_second']/baseline['images_per_second'],
            speedup_vs_original=selected['images_per_second']/m.read(OUT/'benchmarks.json')[0]['images_per_second']))
    except Exception as error:
        m.write(OUT/'teacher-failure.json',dict(error=str(error),resuming_with=selected,time=time.time()))
    finally:resume(selected)


if __name__=='__main__':main()
