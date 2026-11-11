"""Exercise the actual recovery/continuation functions with CPU model states."""
import ast
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch


class RecoveryTest(unittest.TestCase):
    def test_regression_restores_both_adams_and_continues_without_replaying_data(self):
        source=ast.parse(Path(__file__).with_name('continuous.py').read_text())
        main=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        functions=[n for n in main.body if isinstance(n,ast.FunctionDef) and n.name in ['recover','continuous_loop']]
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);out=root/'out';durable=root/'durable';out.mkdir();durable.mkdir()
            model=torch.nn.Linear(2,2);critic=torch.nn.Linear(2,1)
            optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
            d_optimizer=torch.optim.Adam(critic.parameters(),lr=.01)
            model(torch.ones(1,2)).sum().backward();optimizer.step()
            critic(torch.ones(1,2)).sum().backward();d_optimizer.step()
            payload=copy.deepcopy(dict(model=model.state_dict(),discriminator=critic.state_dict(),
                optimizer=optimizer.state_dict(),discriminator_optimizer=d_optimizer.state_dict(),
                progress=dict(epoch=0,batch=2,step=1),rng=[{'sentinel':True}],
                learning_dynamics=dict(lr_factor=1.,nonimproving_checks=0,evaluations=1,reductions=0)))
            torch.save(payload,out/'tokenizer-best-rfid.pt')
            (out/'best-reconstruction.json').write_text(json.dumps(dict(rfid=2.465)))
            with torch.no_grad():model.weight.add_(10);critic.weight.add_(10)
            logs=[];restored_rng=[];calls=[]
            def log(phase,**values):
                logs.append(dict(phase=phase,time=0,**values))
            experiment=SimpleNamespace(out=out,model=model,discriminator=critic,
                optimizer=optimizer,d_optimizer=d_optimizer,progress=dict(epoch=2,batch=154,step=77),
                learning_dynamics=dict(lr_factor=1.,nonimproving_checks=1,evaluations=2,reductions=0),
                stop=False,quality_stop=None,run=SimpleNamespace(summary={}),log=log)
            def loop(self,max_steps=None):
                calls.append(1)
                if len(calls)==1:self.quality_stop='stop_regression';self.stop=True
                else:
                    self.assertions_passed=True
                    assert self.quality_stop is None and not self.stop
                    assert self.learning_dynamics['lr_factor']==.5
            namespace=dict(torch=torch,json=json,Path=Path,time=SimpleNamespace(time=lambda:0),
                cfg={'continuation':{'durable':str(durable)}},original_loop=loop,
                trainer=SimpleNamespace(restore_rng=lambda r:restored_rng.append(r),
                    atomic=lambda p,v:p.write_text(json.dumps(v))))
            exec(compile(ast.fix_missing_locations(ast.Module(body=functions,type_ignores=[])),
                         'continuous.py','exec'),namespace)
            namespace['continuous_loop'](experiment)
            self.assertEqual(len(calls),2)
            self.assertTrue(experiment.assertions_passed)
            self.assertEqual(experiment.progress,dict(epoch=2,batch=154,step=77))
            self.assertTrue(torch.equal(model.weight,payload['model']['weight']))
            self.assertTrue(torch.equal(critic.weight,payload['discriminator']['weight']))
            self.assertEqual(restored_rng,[{'sentinel':True}])
            self.assertEqual(experiment.run.summary['training/status'],'running')
            self.assertEqual(logs[0]['restored_weight_step'],1)
            self.assertEqual(logs[0]['stream_progress']['step'],77)
            self.assertEqual(logs[0]['optimizer_age'],[1])


if __name__=='__main__':unittest.main()
