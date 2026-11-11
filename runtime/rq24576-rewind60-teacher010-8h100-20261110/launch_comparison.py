"""Run and aggregate the matched 10k comparison without changing tokenizers."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
NEW=Path('/tmp/laser-rq24576-rewind60-teacher010-8h100-20261110')
BASE=Path('/tmp/laser-rq24576-native-8h100-20261108')
OUT=ROOT/'outputs/imagenet-rq24576-vs-original-latenterror-20261110'


def write(path,value):
    path=Path(path);pending=path.with_suffix('.pending');pending.write_text(json.dumps(value,indent=2)+'\n');pending.replace(path)


def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--validation',action='store_true');args=parser.parse_args()
    expected=50000 if args.validation else 10000
    if args.validation:OUT=ROOT/'outputs/imagenet-rq24576-vs-original-latenterror-val50k-20261110'
    matched=bool(os.environ.get('LASER_COMPARISON_TEMPERATURE'))
    if matched:OUT=ROOT/'outputs/imagenet-rq24576-noisematched-latenterror-val50k-20261110'
    OUT.mkdir(exist_ok=True)
    original=NEW/'original-rq/inputs';original.mkdir(parents=True,exist_ok=True)
    for name in ['stage1-config.yaml','stage1-model.pt']:
        target=original/name
        if not target.exists():target.symlink_to(NEW/'inputs'/('original-rq-'+name))
    children=[]
    for rank in range(8):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
        with (OUT/f'worker-{rank}.log').open('ab',buffering=0) as log:
            command=[sys.executable,'-u',str(HERE/'compare_latents.py'),'--rank',str(rank),'--output',str(OUT)]
            if args.validation:command.append('--validation')
            children.append(subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
    write(OUT/'launch.json',dict(pid=os.getpid(),workers=[c.pid for c in children],images=expected,time=time.time()))
    codes=[c.wait() for c in children];assert codes==[0]*8,codes
    for rank in range(8):assert json.loads((OUT/f'verification-rank-{rank}.json').read_text())['passed']
    sys.path[:0]=[str(BASE/'source'),str(BASE)];os.environ['RQ_UPSTREAM_ROOT']=str(BASE/'upstream')
    import numpy as np
    from src.rqvae_metrics import frechet_distance
    local=NEW/('latent-comparison-val50k-matched' if matched else ('latent-comparison-val50k' if args.validation else 'latent-comparison'))
    def collect(name):
        parts=[np.load(local/f'{name}-rank-{r}.npz') for r in range(8)]
        count=sum(int(p['count']) for p in parts);total=sum(p['total'] for p in parts);cross=sum(p['cross'] for p in parts)
        assert count==expected
        return parts,count,total/count,(cross-np.outer(total,total)/count)/(count-1)
    _,_,real_mu,real_cov=collect('real');results={};arrays={}
    keys=['original_rq_greedy','original_rq_teacher050','laser_rq24576_greedy','laser_rq24576_teacher050','laser_rq24576_teacher025','laser_rq24576_teacher010']
    if args.validation:keys=[k for k in keys if k not in ['laser_rq24576_teacher050','laser_rq24576_teacher025']]
    if matched:keys=[k.replace('teacher010','teacher_matched') for k in keys]
    rfid_key='validation_rfid50k' if args.validation else 'paired_rfid10k'
    for key in keys:
        parts,count,mu,cov=collect(key);sums=sum(p['sums'] for p in parts);ch=sum(p['channel_sum'] for p in parts)
        sites=count*64;centered_energy=sums[1]-np.square(ch).sum()/sites
        arrays[key]=np.concatenate([p['per_image'] for p in parts])
        results[key]=dict(images=count,latent_mse=float(sums[0]/sums[2]),latent_energy_per_element=float(sums[1]/sums[2]),
            error_over_latent_energy=float(sums[0]/sums[1]),error_over_centered_latent_variance=float(sums[0]/centered_energy),
            residual_mse_by_depth=(sums[7:11]/sums[2]).tolist(),entropy_by_depth=(sums[3:7]/sites).tolist(),
            trajectory_flip_fraction_vs_greedy_by_depth=(sums[11:15]/sites).tolist(),**{rfid_key:frechet_distance(real_mu,real_cov,mu,cov)})
    for key,row in results.items():
        base='original_rq_greedy' if key.startswith('original') else 'laser_rq24576_greedy'
        row['sampled_to_greedy_error_ratio']=row['latent_mse']/results[base]['latent_mse']
        row['extra_error_over_latent_energy']=(row['latent_mse']-results[base]['latent_mse'])/row['latent_energy_per_element']
    control=results['original_rq_teacher050'];current=results['laser_rq24576_teacher_matched' if matched else 'laser_rq24576_teacher010']
    report=dict(passed=True,images=expected,split='Official ImageNet validation' if args.validation else 'Matched heldout training images, disjoint from dictionary fitting',
        validation_used_for_calibration=False,requested_teacher_temperature=float(os.environ.get('LASER_COMPARISON_TEMPERATURE','.1')),
        original_rq_stage1_sha256='e8eb312e6c0dbd8e490ca95f69e3c3b417162625ff9320bbfff791cbf832b694',
        laser_codebook_sha256='8b3f295593b3193d28991554f7a16d629074ec6ce3072a477fb2772e484d86de',results=results,
        teacher010_comparison=dict(normalized_total_latent_error_no_worse=current['error_over_latent_energy']<=control['error_over_latent_energy'],
            centered_normalized_error_no_worse=current['error_over_centered_latent_variance']<=control['error_over_centered_latent_variance'],
            added_noise_over_latent_energy_no_worse=current['extra_error_over_latent_energy']<=control['extra_error_over_latent_energy'],
            reconstruction_rfid_no_worse=current[rfid_key]<=control[rfid_key]),
        limits='Raw MSE values use different learned latent coordinates. Normalization removes a global scale but does not equalize decoder sensitivity. Reconstruction FID is distinct from generation FID.')
    if args.validation:
        report['original_greedy_rfid_reproduced']=abs(results['original_rq_greedy'][rfid_key]-4.723498928977904)<.03
        report['laser_greedy_rfid_reproduced']=abs(results['laser_rq24576_greedy'][rfid_key]-4.561264923772001)<.03
        assert report['original_greedy_rfid_reproduced'] and report['laser_greedy_rfid_reproduced']
    if matched:report['matched_teacher_comparison']=report.pop('teacher010_comparison')
    write(OUT/'report.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
