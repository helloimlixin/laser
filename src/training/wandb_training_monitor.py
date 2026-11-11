"""Stream an existing training job's metrics to W&B without restarting it."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def complete_json_rows(path, position):
    """Leave partially appended lines for the next poll; handle file truncation."""
    path = Path(path)
    if not path.is_file():
        return [], position
    if path.stat().st_size < position:
        position = 0
    rows = []
    with path.open() as stream:
        stream.seek(position)
        while True:
            start = stream.tell()
            line = stream.readline()
            if not line or not line.endswith('\n'):
                return rows, start
            rows.append(json.loads(line))
            position = stream.tell()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--credentials', type=Path, required=True)
    parser.add_argument('--base-url', default='https://api.wandb.ai')
    parser.add_argument('--entity', default='helloimlixin-rutgers')
    parser.add_argument('--project', default='laser')
    args = parser.parse_args()
    output = args.output.resolve()
    lock = (output / 'wandb-monitor.lock').open('w')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.environ.update(WANDB_API_KEY=args.credentials.read_text().strip(), WANDB_BASE_URL=args.base_url,
                      WANDB_MODE='online', WANDB_CONSOLE='off', WANDB_DISABLE_GIT='true')
    import wandb
    import yaml
    recipe = yaml.safe_load((output / 'recipe.yaml').read_text())
    startup = read_json(output / 'train/startup-rank0.json', {})
    local = Path(recipe['runtime_base']) / 'wandb-monitor'
    local.mkdir(parents=True, exist_ok=True)
    config = dict(startup, recipe=recipe, wandb_id=recipe['run_id'],
        wandb_logging_enabled=True, logging_transport='external monitor',
        stage1_checkpoint_sha256=recipe['tokenizer']['sha256'])
    run = wandb.init(entity=args.entity, project=args.project, id=recipe['run_id'],
        name=recipe.get('name', recipe['run_id']), resume='allow', dir=str(local), config=config,
        settings=wandb.Settings(console='off', disable_git=True, x_disable_stats=True))
    run.define_metric('train/global_step')
    run.define_metric('train/*', step_metric='train/global_step')
    run.define_metric('eval/*', step_metric='train/epoch')
    run.define_metric('gpu/*', step_metric='train/global_step')
    run.define_metric('samples/global_step')
    run.define_metric('samples/*', step_metric='samples/global_step')
    run.summary.update({'training/status': 'running', 'logging/transport': 'external monitor',
        'training/stop_after_epochs': int(recipe['epochs']), 'evaluation/generated_images': 50000,
        'evaluation/reference_images': 1281167, 'evaluation/reference_split': 'train'})
    atomic_json(output / 'wandb.json', dict(url=run.url, id=run.id, entity=args.entity,
        project=args.project, monitor_pid=os.getpid(), base_url=args.base_url))
    print(json.dumps(dict(wandb_url=run.url, monitor_pid=os.getpid())), flush=True)
    last_step = int(run.summary.get('logging/last_training_step', -1))
    position = 0
    evaluations = set(run.summary.get('logging/uploaded_evaluations', []))
    images = set(run.summary.get('logging/uploaded_images', []))
    stopping = False

    def stop(sig, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    last_status = last_gpu = 0.
    while not stopping:
        rows, position = complete_json_rows(output / 'train/metrics.jsonl', position)
        for row in rows:
            step = int(row['train/global_step'])
            if step <= last_step:
                continue
            row['train/epoch_progress'] = step / int(startup.get('optimizer_updates_per_epoch', 626))
            run.log(row)
            last_step = step
            run.summary['logging/last_training_step'] = last_step
        for path in sorted((output / 'evaluations').glob('epoch*.json')):
            if path.name not in evaluations:
                row = read_json(path)
                if row:
                    run.log(row)
                    evaluations.add(path.name)
                    run.summary['logging/uploaded_evaluations'] = sorted(evaluations)
        for path in sorted((output / 'samples').glob('*.png')):
            if path.name not in images:
                match = re.match(r'step(\d+)-', path.name)
                key = ('tokenizer/reconstructions' if path.name.startswith('tokenizer-') else
                       'samples/fixed' if match else 'samples/generated')
                row = {key: wandb.Image(str(path), caption=path.stem),
                       'train/global_step': max(last_step, 0)}
                if match:
                    row['samples/global_step'] = int(match.group(1))
                run.log(row)
                images.add(path.name)
                run.summary['logging/uploaded_images'] = sorted(images)
        now = time.monotonic()
        progress = read_json(output / 'train/progress-rank0.json', {})
        if now - last_gpu >= 30 and progress:
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used',
                '--format=csv,noheader,nounits'], capture_output=True, text=True)
            metrics = {'train/global_step': int(progress['train/global_step'])}
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    index, utilization, memory = [v.strip() for v in line.split(',')]
                    metrics[f'gpu/{index}/utilization_percent'] = float(utilization)
                    metrics[f'gpu/{index}/memory_used_gib'] = float(memory) / 1024
                run.log(metrics)
            last_gpu = now
        if now - last_status >= 10:
            generation = read_json(output / 'generation-progress.json', {})
            state = read_json(output / 'status.json', {})
            evaluating = generation.get('time', 0) > progress.get('time', 0)
            phase = 'evaluating' if evaluating else state.get('phase', 'training')
            run.summary['training/phase'] = phase
            if evaluating:
                run.summary.update({'evaluation/current_epoch': generation.get('epoch'),
                    'evaluation/generated_on_rank': generation.get('generated_on_rank'),
                    'evaluation/samples_on_rank': generation.get('samples_on_rank')})
            checkpoint = read_json(output / 'checkpoint.json', {})
            if checkpoint:
                run.summary['training/checkpoint_step'] = checkpoint['global_step']
            if progress:
                run.summary.update({'training/live_step': progress['train/global_step'],
                    'training/latest_loss': progress['train/loss'], 'training/worker_pid': progress['pid']})
            atomic_json(output / 'wandb-monitor-status.json', dict(phase=phase, pid=os.getpid(),
                url=run.url, last_logged_training_step=last_step,
                live_training_step=progress.get('train/global_step'), time=time.time()))
            last_status = now
        if (output / 'completed.json').is_file():
            run.summary['training/status'] = 'completed'
            run.finish()
            return
        state = read_json(output / 'status.json', {})
        if state.get('phase') == 'failed':
            run.summary['training/status'] = 'failed'
            run.finish(exit_code=1)
            return
        time.sleep(2)
    run.finish()


if __name__ == '__main__':
    main()
