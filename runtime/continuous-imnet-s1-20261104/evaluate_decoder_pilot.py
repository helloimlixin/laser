"""Evaluate the completed private pilot, without rerunning any training updates."""
import json
from pathlib import Path
import sys
import time

import torch

BASE=Path('/tmp/laser-decoder-only-pilot-20261105')
sys.path.insert(0,str(BASE/'runtime'))
import train_matched_stage1 as trainer


def main():
    torch.set_num_threads(2)
    trainer.var_dist.initialize(backend='nccl',timeout=120)
    torch.cuda.set_per_process_memory_fraction(.20,0)
    try:
        cfg=json.loads((BASE/'run-output/config.json').read_text())
        cfg['workers']=4
        state=torch.load(BASE/'run-output/tokenizer-last.pt',map_location='cpu',weights_only=False,mmap=True)
        proof=json.loads((BASE/'training-verification.json').read_text())
        assert proof['passed'] and state['progress']['step']==cfg['decoder_only_pilot']['source_step']+30
        experiment=object.__new__(trainer.Experiment)
        experiment.cfg=cfg;experiment.rank=0;experiment.world=1
        experiment.out=BASE/'run-output';experiment.run=None
        experiment.device=torch.device('cuda',0);experiment.inception=None
        experiment.progress=dict(state['progress'])
        experiment.model=trainer.FourScaleJointVAE().eval().requires_grad_(False)
        experiment.model.load_state_dict(state['model'],strict=True)
        experiment.model.cuda()
        experiment.val=trainer.ArchiveImages(cfg['val_archive'],cfg['val_index'],False,cfg['seed'],labels=cfg['val_labels'])
        started=time.monotonic();experiment.evaluate()
        result=json.loads((experiment.out/f"reconstruction-epoch{experiment.progress['epoch']:03d}.json").read_text())
        source=cfg['decoder_only_pilot']
        report=dict(source_rfid=source['source_rfid'],candidate_rfid=result['rfid'],
            full_validation_images=result['count'],source_step=source['source_step'],
            candidate_step=state['progress']['step'],improved=result['rfid']<source['source_rfid'],
            fid_protocol_unchanged=True,training_proof=proof,
            checkpoint=str(BASE/'run-output/tokenizer-last.pt'),
            evaluation_seconds=time.monotonic()-started,evaluation_loader_workers=4,
            no_training_updates_rerun=True)
        (BASE/'pilot-result.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)
    finally:
        trainer.td.destroy_process_group()


if __name__=='__main__':main()
