"""Exercise averaging, teacher boundaries, and restored LR beyond epoch 100."""
import json
import math
from pathlib import Path
import torch
from refinement import ModelEMA,LearningRateRefinement,teacher_temperature


def main():
    model=torch.nn.Linear(2,1,bias=False)
    with torch.no_grad():model.weight.fill_(2.)
    ema=ModelEMA(model,.75)
    with torch.no_grad():model.weight.fill_(6.)
    ema.update(model)
    torch.testing.assert_close(ema.model.weight,torch.full_like(model.weight,3.))
    with torch.no_grad():model.weight.fill_(10.)
    ema.update(model)
    torch.testing.assert_close(ema.model.weight,torch.full_like(model.weight,4.75))
    assert ema.updates==2 and all(not p.requires_grad for p in ema.model.parameters())
    plan=json.loads((Path(__file__).parent/'plan.json').read_text());policy=plan['lr'];teacher=plan['teacher']
    assert teacher_temperature(policy['start_step'],teacher)==.5
    assert teacher_temperature(policy['start_step']+626,teacher)==.375
    assert teacher_temperature(policy['start_step']+1252,teacher)==.25
    assert teacher_temperature(policy['end_step'],teacher)==.25
    # Mirror the native warmup wrapper's serialized structure and actual cosine.
    class Wrapper:
        def __init__(self,optimizer,step):
            self.after_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,62600)
            self.after_scheduler.last_epoch=step
        def step(self):self.after_scheduler.step()
        def get_last_lr(self):return self.after_scheduler.get_last_lr()
        def state_dict(self):return {'after_scheduler':self.after_scheduler.state_dict()}
        def load_state_dict(self,state):self.after_scheduler.load_state_dict(state['after_scheduler'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=policy['start_lr'])
    native=Wrapper(optimizer,policy['start_step'])
    schedule=LearningRateRefinement(native,optimizer,policy,policy['start_step'])
    rates=[]
    for step in range(policy['start_step']+1,policy['end_step']+1):
        optimizer.step();schedule.step();rates.append(optimizer.param_groups[0]['lr'])
        if step in [policy['ramp_end_step'],62601,policy['end_step']-20]:
            restored=LearningRateRefinement(native,optimizer,policy,step,saved_policy=policy)
            restored.load_state_dict(schedule.state_dict());schedule=restored
        assert math.isclose(optimizer.param_groups[0]['lr'],schedule.value_at_step(step),abs_tol=1e-13)
    assert math.isclose(schedule.value_at_step(policy['ramp_end_step']),1e-5)
    assert math.isclose(schedule.value_at_step(policy['start_step']),policy['start_lr'])
    assert math.isclose(rates[-1],1e-6)
    report=dict(passed=True,ema_closed_form=True,teacher_boundaries=True,
        lr_resume_beyond_epoch100=True,updates=len(rates),transition_start_lr=policy['start_lr'],cosine_start_lr=policy['peak_lr'],final_lr=rates[-1])
    (Path(plan['output'])/'policy-verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
