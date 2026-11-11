"""Wait for the final atomic checkpoint, then replace the stopped trainer."""
import json,os,signal,subprocess,time
from pathlib import Path
BASE=Path('/tmp/laser-cc3m-rq24576-20261111');ROOT=Path('/workspace/Projects/laser')
print('Waiting for final atomic step950 checkpoint; no optimizer progress will be discarded',flush=True)
while True:
 candidates=list((BASE/'checkpoints').glob('step-0000950-*.pt'))
 if candidates:break
 time.sleep(3)
import torch
payload=torch.load(candidates[0],map_location='cpu',mmap=True,weights_only=False)
assert payload['global_step']==950 and payload['next_update']==950
assert payload['scheduler']['last_epoch']==950 and len(payload['rng_by_rank'])==2
assert {int(s['step']) for s in payload['optimizer']['state'].values()}=={950}
assert (BASE/'checkpoints/last.pt').stat().st_ino==candidates[0].stat().st_ino
saved=BASE/'inputs/stage2-step950.pt'
if not saved.exists():saved.hardlink_to(candidates[0])
del payload
print('Complete step950 model, optimizer, scheduler and both rank RNG states preserved',flush=True)
try:os.killpg(3363,signal.SIGKILL)
except ProcessLookupError:pass
for pid in [3374,3375,3459]:
 try:os.kill(pid,signal.SIGKILL)
 except ProcessLookupError:pass
for _ in range(60):
 alive=[]
 for pid in [3374,3375]:
  try:os.kill(pid,0);alive.append(pid)
  except ProcessLookupError:pass
 if not alive:break
 time.sleep(.5)
subprocess.run(['/tmp/laser-cc3m-venv/bin/python',str(ROOT/'runtime/cc3m-rq24576-speed-20261111/resume.py')],check=True,cwd=ROOT)
