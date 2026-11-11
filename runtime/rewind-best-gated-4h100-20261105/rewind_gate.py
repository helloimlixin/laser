"""A rewind must reproduce the best before any new optimizer updates are allowed."""
import json
from pathlib import Path
import time


def baseline_decision(reference, measured):
    fid_delta = measured['fid'] - reference['fid']
    is_delta = measured['inception_score'] - reference['inception_score']
    return dict(accepted=abs(fid_delta) <= 0.10 and abs(is_delta) <= 0.50,
                fid_delta=fid_delta, is_delta=is_delta,
                criterion='Reproduce historical four-GPU FID within 0.10 and IS within 0.50')


def pilot_decision(baseline, candidate):
    fid_delta = candidate['fid'] - baseline['fid']
    is_delta = candidate['inception_score'] - baseline['inception_score']
    return dict(accepted=fid_delta <= -0.05 and is_delta >= -0.20,
                fid_delta=fid_delta, is_delta=is_delta,
                criterion='Matched FID50k improves by at least 0.05 and IS falls by no more than 0.20')


def record_rewind(out, reason):
    out = Path(out)
    best_path = out / 'persistent-train/checkpoints/best.json'
    best = json.loads(best_path.read_text())
    assert best['state'] == 'COMMITTED'
    value = dict(phase='rewound_to_best', reason=reason, time=time.time(),
                 active_checkpoint=best, further_updates_allowed=False)
    path = out / 'rewound.json'
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)
    return value
