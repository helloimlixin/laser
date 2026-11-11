"""Execute the frozen VAR runner with verified recovery and separate preflight outputs."""
import sys,json,os,importlib.util,hashlib
from pathlib import Path
BASE=Path('/tmp/laser-var-resume-20261105')
HERE=Path(__file__).resolve().parent
OUT=Path('/workspace/Projects/laser/outputs/imnet-var-j4096-bestep1-continue200-4rtx6k-20261105')
sys.path[:0]=[str(BASE/'runtime')]
spec=importlib.util.spec_from_file_location('frozen_var_training',BASE/'runtime/train_joint4096_var.py')
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
config=json.loads(Path(sys.argv[sys.argv.index('--config')+1]).read_text())
preflight='--smoke' in sys.argv
if preflight:
 # Retain the real 1668-update epoch length and the original LR clock.
 m.SmokeImages=lambda _:m.ArchiveImages(config['train_archive'],config['train_index'],True,config['seed'])
Original=m.Experiment
class ResumedExperiment(Original):
 def __init__(self,cfg,smoke=False):
  super().__init__(cfg,smoke)
  assert self.progress['step']>=1668 and self.updates==1668
  ages={int(s['step']) for s in self.optimizer.state.values() if 'step' in s}
  assert ages=={self.progress['step']},ages
  assert not any(p.requires_grad for p in self.source.parameters())
  assert self.world==4 and self.global_batch==768
  if self.rank==0:
   m.atomic(self.out/'resume-verification.json',dict(passed=True,restored_step=self.progress['step'],restored_epoch=self.progress['epoch'],optimizer_ages=sorted(ages),schedule_reset=False,original_training_contract_preserved=True,tokenizer_frozen=True,world_size=self.world,global_batch=self.global_batch))
   if self.run:
    self.run.summary.update({'prior/fresh_initialization':False,'training/continued_from_step':1668,'training/source_best_fid50k':196.2973291315157,'training/initialization':'best_evaluated_epoch1_full_state','training/hardware':'4 NVIDIA RTX PRO 6000 Blackwell 96GB'})
    self.run.use_artifact('helloimlixin-rutgers/laser/imnet-var-j4096-a1024-l4-d16-scratch200-4h100-20261102-r2-prior:v2')
    self.run.use_artifact(cfg['tokenizer_artifact'])
    self.run.use_artifact(cfg['codebook_artifact'])
    import wandb
    a=wandb.Artifact(cfg['wandb']['id']+'-resume-runtime',type='code')
    for p in HERE.glob('*.py'):a.add_file(str(p),name=p.name)
    a.add_file(str(OUT/'source-verification.json'));a.add_file(str(BASE/'config.json'),name='resume-config.json')
    a=self.run.log_artifact(a);a.wait()
    m.atomic(self.out/'resume-runtime-artifact.json',dict(artifact=a.qualified_name,state=str(a.state)))
 def train_loop(self,max_steps=None):
  super().train_loop(max_steps)
  if self.smoke:self.preview()
m.Experiment=ResumedExperiment
if __name__ == '__main__':
 m.main()
