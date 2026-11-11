"""Replace the slow shared-storage extraction once the local archive is verified."""
import json,os,signal,subprocess,time
from pathlib import Path
BASE=Path('/tmp/laser-matched680-vector6144-20261111')
OUT=Path('/workspace/Projects/laser/outputs/imnet-s2-matched680-vector6144-var-d16-4h100-20261111')
ready=BASE/'data/ILSVRC2012_img_train.tar.verified.json'
while not ready.exists():time.sleep(5)
assert json.loads(ready.read_text())['md5']=='1d675b47d978889d74fa0da5fadfb00e'
print('Verified NVMe training archive ready; restarting incomplete extraction on local storage',flush=True)
try:os.kill(3777,signal.SIGTERM)
except ProcessLookupError:pass
for pid in json.loads(Path('/tmp/vector-cache-paused.json').read_text())['pids']:
 try:os.kill(pid,signal.SIGCONT)
 except ProcessLookupError:pass
for _ in range(120):
 try:os.kill(3768,0)
 except ProcessLookupError:break
 time.sleep(1)
else:raise RuntimeError('Previous controller did not release its lock')
with (OUT/'supervisor.log').open('ab',buffering=0) as log:
 p=subprocess.Popen(['python','train.py','--config','configs/stage2/imagenet-matched680-vector6144-classcond-4h100-20261111.yaml'],cwd='/workspace/Projects/laser',stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
(OUT/'supervisor-launch.json').write_text(json.dumps(dict(pid=p.pid,time=time.time(),reason='local NVMe archive now MD5 verified'),indent=2))
print('Controller restarted',p.pid,flush=True)
