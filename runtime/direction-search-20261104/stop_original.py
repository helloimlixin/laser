"""Stop an old auto-restarting W&B run, with bounded repeated stop requests."""
import json
import os
from pathlib import Path
import time

os.environ['WANDB_API_KEY'] = Path('/root/.config/laser/wandb.key').read_text().strip()
os.environ['WANDB_BASE_URL'] = 'https://api.forge.coreweave.com'
import wandb

RUN = 'imnet-j16384-a4096-fid14574-lrquarter-trial64-1b200-20261104'
OUT = Path('/workspace/Projects/laser/outputs') / RUN
api = wandb.Api(timeout=30)
requests_sent = 0
started = time.time()
while time.time() - started < 900:
    try:
        status = json.loads((OUT / 'status.json').read_text())
        if status['phase'] in {'stopping_paid_instance', 'unrecoverable_training_failure',
                               'interrupted_by_user', 'billing_stop_deferred_other_gpu_work'}:
            print(json.dumps(dict(stopping=True, phase=status['phase'], time=time.time())), flush=True)
            break
        run = api.run('helloimlixin-rutgers/laser/' + RUN)
        q = 'query($e:String!,$p:String!,$r:String!){project(name:$p,entityName:$e){run(name:$r){stopped state}}}'
        record = run._service_api.execute_graphql(q, dict(e='helloimlixin-rutgers', p='laser', r=RUN))
        current = record['project']['run']
        if current['state'] != 'running' and time.time() - status['time'] > 90:
            print(json.dumps(dict(stopped=True, state=current['state'], time=time.time())), flush=True)
            break
        if not current['stopped']:
            result = run._service_api.execute_graphql(
                'mutation($id:ID!){stopRun(input:{id:$id}){success}}', dict(id=run.storage_id))
            assert result['stopRun']['success']
            requests_sent += 1
        progress = status.get('latest_progress') or {}
        report = dict(time=time.time(), requests_sent=requests_sent, phase=status['phase'],
                      retry_count=status.get('retry_count'), old_step=progress.get('train/global_step'),
                      last_remote_status_time=status['time'])
        (OUT / 'stop-watch.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)
    except Exception as error:
        print(json.dumps(dict(error_type=type(error).__name__, time=time.time())), flush=True)
    time.sleep(10)
else:
    print('Remote stop could not be confirmed within fifteen minutes', flush=True)
