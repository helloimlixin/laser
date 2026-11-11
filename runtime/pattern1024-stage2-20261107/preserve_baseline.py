"""Preserve the complete baseline checkpoint before reallocating all four GPUs."""
import json
import os
from pathlib import Path
import signal
import time

ROOT=Path('/tmp/laser-pattern1024-stage2-20261107')
OLD=Path('/tmp/laser-compound128-20261107')


def main():
    import torch
    checkpoint=json.loads((OLD/'production-nvme/checkpoint.json').read_text())
    path=Path(checkpoint['path'])
    state=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    assert state['run_id']=='imagenet-laser-compound128-native-scratch100-4a100-20261107'
    step=int(state['global_step'])
    assert len(state['optimizer']['state'])==837 and all(int(v['step'])==step for v in state['optimizer']['state'].values())
    del state
    protected=ROOT/f'baseline-step-{step:07d}.pt'
    if not protected.exists():protected.hardlink_to(path)
    progress=json.loads((OLD/'production-nvme/train/progress-rank0.json').read_text())
    controller=85895
    try:
        command=Path(f'/proc/{controller}/cmdline').read_bytes().decode().replace('\x00',' ')
        assert 'compound128-20261107/supervise.py' in command
        os.kill(controller,signal.SIGTERM)
    except FileNotFoundError:
        pass
    receipt=dict(baseline_run=state_id(),checkpoint=str(path),protected_local_checkpoint=str(protected),
        checkpoint_global_step=step,last_observed_global_step=progress['train/global_step'],
        complete_optimizer_state_verified=True,checkpoint_bytes=path.stat().st_size,
        cloud_receipt=json.loads((OLD/'production-nvme/cloud-sync-status.json').read_text()),
        checkpoint_uploader_left_running=True,time=time.time(),reason='User requested fresh 1K-pattern training with full GPU capacity')
    (ROOT/'preserved-baseline.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
    os.environ.update(WANDB_API_KEY=Path('/root/.config/laser/wandb.key').read_text().strip(),WANDB_BASE_URL='https://api.forge.coreweave.com')
    import wandb
    run=wandb.Api().run('helloimlixin-rutgers/laser/'+state_id())
    run.summary.update({'training/status':'stopped_for_fresh_pattern1024_training',
        'training/replaced_by':'imagenet-laser-pattern1024-native-scratch100-4a100-20261107',
        'training/preserved_checkpoint_step':step})
    run.summary.update()


def state_id():
    return 'imagenet-laser-compound128-native-scratch100-4a100-20261107'


if __name__=='__main__':main()
