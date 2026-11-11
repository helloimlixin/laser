"""Confirm the modest sampler gain with another seed, then activate generation only."""
import json
from pathlib import Path
import sys
import time
from sweep import run_evaluation,resume_training,write,BASE,EXPERIMENT


def main():
    activation=EXPERIMENT/'activation'
    while not (activation/'pause.json').exists() or json.loads((activation/'pause.json').read_text())['phase']!='paused_for_fixed_checkpoint_calibration':
        time.sleep(10)
    initial=json.loads((EXPERIMENT/'report.json').read_text())
    base=next(r for r in initial['confirmation'] if r['policy']=='baseline')
    candidates=[r for r in initial['confirmation'] if r['policy']!='baseline' and r['inception_score']>=.95*base['inception_score']]
    candidate=min(candidates,key=lambda r:r['fid'])
    try:
        policies=[dict(name=r['policy'],sampling=r['sampling']) for r in [base,candidate]]
        repeated=run_evaluation('repeat-confirm',policies,50000,Path(base['checkpoint']),initial['checkpoint_sha256'],seed=26110873)
        repeat_base=next(r for r in repeated if r['policy']=='baseline')
        repeat_candidate=next(r for r in repeated if r['policy']==candidate['policy'])
        delta1=base['fid']-candidate['fid'];delta2=repeat_base['fid']-repeat_candidate['fid']
        passed=(delta1>0 and delta2>0 and (delta1+delta2)/2>=.3 and repeat_candidate['inception_score']>=.95*repeat_base['inception_score'])
        report=dict(initial,selected_policy=candidate['policy'] if passed else 'baseline',
            sampling=candidate['sampling'] if passed else base['sampling'],fid=candidate['fid'] if passed else base['fid'],
            inception_score=candidate['inception_score'] if passed else base['inception_score'],
            improved=passed,minimum_fid_improvement=0.,repeat_confirmation=repeated,
            fid_improvement_by_seed=[delta1,delta2],mean_fid_improvement=(delta1+delta2)/2,
            selection='Positive paired FID50k improvement on two independent confirmation seeds, mean gain >=0.3; retain >=95% baseline IS on each seed.')
        write(activation/'report.json',report)
        if passed:
            from install_override import install
            install(report)
        write(activation/'decision.json',dict(applied=passed,policy=report['selected_policy'],sampling=report['sampling'],fid_gains=[delta1,delta2]))
    except BaseException as error:
        write(activation/'failure.json',dict(error=str(error),time=time.time()));raise
    finally:
        pid=resume_training()
        write(activation/'training-resumed.json',dict(supervisor_pid=pid,time=time.time()))


if __name__=='__main__':main()
