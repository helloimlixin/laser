"""Independently reproduce the best snapshot's full 50k FID without training."""
import json
from pathlib import Path
import shutil
import sys

import torch
import torch.distributed as dist

OLD=Path('/tmp/laser-s1-best-b200')
OUT=Path('/tmp/laser-s1-best2486-b200/baseline-control')
sys.path.insert(0,str(OLD/'frozen/runtime'))
import train_matched_stage1 as trainer


def main():
    torch.set_num_threads(4)
    trainer.var_dist.initialize(backend='nccl',timeout=120)
    try:
        OUT.mkdir(exist_ok=True)
        cfg=json.loads((OLD/'run-output/config.json').read_text())
        best=json.loads((OLD/'run-output/best-reconstruction.json').read_text())
        source=OLD/'run-output/tokenizer-best-rfid.pt'
        assert trainer.sha(source)==best['checkpoint']['checkpoint_sha256']
        state=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
        assert state['progress']['step']==42536
        experiment=object.__new__(trainer.Experiment)
        experiment.cfg=cfg
        experiment.device=torch.device('cuda',0)
        experiment.rank=0;experiment.world=1;experiment.out=OUT;experiment.run=None
        experiment.inception=None;experiment.progress=dict(state['progress'])
        experiment.model=trainer.FourScaleJointVAE()
        experiment.model.load_state_dict(state['model'],strict=True)
        experiment.model.to(experiment.device).eval()
        experiment.val=trainer.ArchiveImages(cfg['val_archive'],cfg['val_index'],False,cfg['seed'],labels=cfg['val_labels'])
        trainer.atomic(OUT/'checkpoint.json',best['checkpoint'])
        experiment.evaluate()
        result=json.loads((OUT/'reconstruction-epoch004.json').read_text())
        verification=dict(passed=abs(result['rfid']-best['rfid'])<.01,
            expected_rfid=best['rfid'],measured_rfid=result['rfid'],count=result['count'],
            checkpoint_sha256=best['checkpoint']['checkpoint_sha256'],no_training_updates=True)
        trainer.atomic(OUT/'verification.json',verification)
        print(json.dumps(verification),flush=True)
        assert verification['passed'],verification
    finally:
        dist.destroy_process_group()


if __name__=='__main__':main()
