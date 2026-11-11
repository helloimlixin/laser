import os,signal,json,time
from pathlib import Path
O=Path('/workspace/Projects/laser/outputs/imagenet-beam451-aligned-joint16k-scratch100-8h100-20261108')
NEW=Path('/workspace/Projects/laser/outputs/imagenet-compound128-successful-scratch100-8h100-20261108')
s=json.loads((O/'status.json').read_text());pid=s['supervisor_pid'];assert b'duplicate-beam451-8h100-20261108/supervise.py' in Path(f'/proc/{pid}/cmdline').read_bytes()
start=json.loads((O/'arms/cutoff05/checkpoint.json').read_text())['global_step']
while True:
 r=json.loads((O/'arms/cutoff05/checkpoint.json').read_text())
 if r['global_step']>start and r['persistent_state']=='COMPLETE':break
 time.sleep(5)
os.kill(pid,signal.SIGTERM)
(NEW/'previous-experiment-preserved.json').write_text(json.dumps(dict(stopped_supervisor=pid,checkpoint=r,reason='User requested fresh compound128 training on all GPUs',time=time.time()),indent=2)+'\n')
print('Previous experiment preserved at',r['global_step'],flush=True)
