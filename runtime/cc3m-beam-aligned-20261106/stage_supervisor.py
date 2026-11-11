"""Keep public, checksummed CC3M data staging alive independently of this session."""
import json,subprocess,sys,time
from pathlib import Path
BASE=Path('/tmp/laser-cc3m-aligned-20261106')
SCRIPT=Path(__file__).with_name('stage_data.py')
if (BASE/'all-shards-ready.json').exists():raise SystemExit(0)
for attempt in range(1,11):
 with (BASE/'staging.log').open('ab',buffering=0) as log:
  p=subprocess.Popen([sys.executable,'-u',str(SCRIPT)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  (BASE/'staging-worker.json').write_text(json.dumps(dict(pid=p.pid,supervisor_pid=__import__('os').getpid(),attempt=attempt,time=time.time()),indent=2)+'\n')
  code=p.wait()
 if code==0 and (BASE/'all-shards-ready.json').exists():raise SystemExit(0)
 (BASE/'staging-retry.json').write_text(json.dumps(dict(attempt=attempt,exit_code=code,time=time.time()),indent=2)+'\n')
 time.sleep(5)
raise RuntimeError('Data staging failed ten attempts; inspect staging.log')
