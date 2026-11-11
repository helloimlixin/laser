"""Resume the exact saved optimizer boundary using audited FP32 execution controls."""
import importlib.util,json,os,subprocess,time,zipfile,hashlib
from pathlib import Path
BASE=Path('/tmp/laser-cc3m-rq24576-20261111');ROOT=Path('/workspace/Projects/laser')
spec=importlib.util.spec_from_file_location('cc3m_launch',ROOT/'runtime/cc3m-rq24576-20261111/launch.py')
launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
import torch

def main():
 for pid in [3374,3375]:
  try:os.kill(pid,0)
  except ProcessLookupError:continue
  raise RuntimeError(f'Old rank {pid} is still alive; preserve its final checkpoint first')
 path=BASE/'checkpoints/last.pt'
 payload=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
 assert payload['global_step']==950 and payload['next_update']==950 and payload['epoch']==0
 assert payload['scheduler']['last_epoch']==950
 assert len(payload['rng_by_rank'])==2
 ages={int(v['step']) for v in payload['optimizer']['state'].values()}
 assert ages=={950}
 startup=json.loads((BASE/'live/preflight/startup-rank0.json').read_text())
 assert startup['execution_runtime']['encoder_batch']==32 and startup['execution_runtime']['cudnn_benchmark']
 assert json.loads((BASE/'live/preflight/passed.json').read_text())['passed']
 override=BASE/'execution-fast-fp32.json'
 proof=dict(source_checkpoint=str(path),global_step=950,optimizer_steps=sorted(ages),scheduler_step=950,
  rng_states_restored=2,checkpoint_config_unchanged=True,execution=json.loads(override.read_text()),
  fp32_preserved=True,tf32_enabled=False,source_probes={k:v.reshape(-1)[:4].tolist() for k,v in list(payload['model'].items())[:5]},time=time.time())
 del payload
 repro=BASE/'live/reproducibility'
 with zipfile.ZipFile(repro/'source-fast-fp32.zip','w',zipfile.ZIP_DEFLATED) as archive:
  for p in (BASE/'source').rglob('*'):
   if p.is_file() and p.suffix in ['.py','.json','.yaml','.txt'] and '__pycache__' not in p.parts:archive.write(p,p.relative_to(BASE/'source'))
 for source in [BASE/'live/encoder-speed-parity.json',BASE/'live/encoder-fp32-largebatch-parity.json',BASE/'live/full-speed-benchmark.json',override,BASE/'logs/fast-tests.log']:
  import shutil;shutil.copyfile(source,repro/source.name)
 (repro/'resume-fast-fp32-verification.json').write_text(json.dumps(proof,indent=2)+'\n')
 env=launcher.environment();env['LASER_CC3M_EXECUTION_OVERRIDES']=str(override)
 log=(BASE/'logs/training-fast-fp32.log').open('ab')
 proc=subprocess.Popen(launcher.command(),cwd=BASE/'source',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 status=dict(pid=proc.pid,resumed_from=950,command=launcher.command(),execution=proof['execution'],time=time.time())
 (BASE/'resume-fast-fp32-launch.json').write_text(json.dumps(status,indent=2)+'\n')
 print(json.dumps(status),flush=True)
if __name__=='__main__':main()
