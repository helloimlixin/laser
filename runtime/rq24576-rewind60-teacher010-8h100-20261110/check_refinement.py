"""Check original cosine continuity and the precise epoch60 teacher boundary."""
import json
import math
from pathlib import Path
import torch
from refinement import teacher_temperature,ModelEMA


def main():
    plan=json.loads((Path(__file__).parent/'plan.json').read_text());p=plan['teacher']
    assert teacher_temperature(36308,p)==.5
    assert teacher_temperature(37559,p)==.5
    assert teacher_temperature(37560,p)==p['final']
    assert teacher_temperature(62600,p)==p['final']
    model=torch.nn.Linear(2,1,bias=False)
    with torch.no_grad():model.weight.fill_(2.)
    ema=ModelEMA(model,.75)
    with torch.no_grad():model.weight.fill_(6.)
    ema.update(model);torch.testing.assert_close(ema.model.weight,torch.full_like(model.weight,3.))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0005)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,62600)
    before=None
    for step in range(1,62601):
        optimizer.step();scheduler.step()
        expected=.0005*.5*(1.+math.cos(math.pi*step/62600))
        actual=optimizer.param_groups[0]['lr']
        assert math.isclose(actual,expected,rel_tol=1e-8,abs_tol=1e-12)
        if step==37560:before=actual
    assert optimizer.param_groups[0]['lr']==0.
    report=dict(passed=True,teacher_epoch60_boundary_verified=True,ema_math_verified=True,
        full_original_cosine_verified=True,lr_at_epoch60=before,final_lr=0.,lr_restart=False)
    (Path(plan['output'])/'policy-verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
