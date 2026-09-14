#!/usr/bin/env python3
"""Checkpoint the existing ImageNet prior and launch a verified fresh compact one."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.original_rq_training import atomic_json, file_sha256
from src.tokenizer_fidelity import require_tokenizer_fidelity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--previous', type=Path, default=ROOT/'outputs/imagenet-rfid421-rq8-refit-20260913')
    parser.add_argument('--base', type=Path, default=ROOT/'outputs/imagenet-compact-rq32k-stage2-20260913')
    parser.add_argument('--run-id', default='imagenet-compact-rq32k-scratch-20260913')
    parser.add_argument('--max-rfid-drift', type=float, default=.20)
    args = parser.parse_args()
    base, previous = args.base.resolve(), args.previous.resolve()
    assert not (base/'train').exists() and not (base/'launch.json').exists()
    proof = json.loads((base/'preflight-verified/verification.json').read_text())
    for flag in ('preflight_passed','strict_checkpoint_reload','saved_tensors_finite',
                 'class_conditioning_verified','released_sampler_decode_passed','tokenizer_frozen'):
        assert proof[flag], flag
    assert proof['world_size']==4 and proof['optimizer_updates']==3
    assert proof['coefficient_construction'] in ('atom-specific','depth-and-atom-specific')
    assert proof['vocabulary'] in (32769,65537)
    assert json.loads((base/'cache-smoke-verification.json').read_text())['passed']
    for name, expected in proof['source_hashes'].items():
        assert file_sha256(ROOT/name)==expected, name
    quality = require_tokenizer_fidelity(base/'fidelity-gate.json',proof['tokenizer_sha256'],
        proof['codebook_sha256'],args.max_rfid_drift)
    old_launch = json.loads((previous/'launch.json').read_text())
    status_path = previous/'train/status.json'
    old_status = json.loads(status_path.read_text())
    assert old_status['phase']=='training'
    pid = old_status['pid']
    command = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    assert str(previous/'train').encode() in command
    assert str(previous/'source-snapshot/scripts/tools/train_imagenet_scaled_stage2.py').encode() in command
    parent_command = Path(f'/proc/{old_launch["torchrun_pid"]}/cmdline').read_bytes()
    assert str(previous).encode() in parent_command
    atomic_json(base/'switch-request.json',dict(previous_run=old_launch['run_id'],previous_status=old_status,
        new_run=args.run_id,from_scratch=True,quality=quality,requested_unix=time.time()))
    requested = time.time()
    os.kill(pid,signal.SIGTERM)
    deadline = time.monotonic()+180
    while time.monotonic()<deadline:
        status = json.loads(status_path.read_text())
        proc = Path(f'/proc/{old_launch["torchrun_pid"]}/cmdline')
        active = proc.exists() and str(previous).encode() in proc.read_bytes()
        if status['phase']=='paused' and not active:
            break
        time.sleep(2)
    else:
        raise TimeoutError('Existing prior has not finished its checkpointed pause')
    saved_path = previous/'train/last.pt'
    assert saved_path.stat().st_mtime>=requested
    import torch
    torch.set_num_threads(8)
    saved = torch.load(saved_path,map_location='cpu',weights_only=False,mmap=True)
    assert saved['step']==status['optimizer_step'] and saved['world_size']==4
    assert len(saved['rng_states'])==4 and saved['optimizer']['state']
    for key in ('state_dict','optimizer','scheduler','scaler','rng_states','cache','config'):
        assert key in saved,key
    for value in saved['state_dict'].values():
        if value.is_floating_point():
            assert torch.isfinite(value).all()
    for state in saved['optimizer']['state'].values():
        for value in state.values():
            if torch.is_tensor(value) and value.is_floating_point():
                assert torch.isfinite(value).all()
    archive = base/'previous-run'
    archive.mkdir(exist_ok=False)
    os.link(saved_path,archive/'last.pt')
    record = dict(run_id=old_launch['run_id'],checkpoint=str(saved_path),
        preserved_checkpoint=str(archive/'last.pt'),checkpoint_sha256=file_sha256(saved_path),
        optimizer_step=saved['step'],epoch=saved['epoch'],batch_in_epoch=saved['batch_in_epoch'],
        full_training_state_preserved=True,all_model_optimizer_tensors_finite=True,
        rng_rank_states=len(saved['rng_states']),status=status,completed_unix=time.time())
    atomic_json(base/'previous-run-paused.json',record)
    print(json.dumps(dict(phase='previous_run_preserved',**record)),flush=True)
    del saved
    subprocess.run([sys.executable,str(ROOT/'scripts/tools/launch_imagenet_scaled_stage2.py'),
        '--base',str(base),'--run-id',args.run_id,'--max-rfid-drift',str(args.max_rfid_drift),
        '--sampler','original_t09'],cwd=ROOT,check=True)


if __name__=='__main__':
    main()
