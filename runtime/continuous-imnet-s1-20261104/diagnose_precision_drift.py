"""Read-only paired reconstruction/token drift probe alongside the live trainer."""
import contextlib
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

BASE=Path('/tmp/laser-s1-lr1e9-b200-20261105')
RUNTIME=BASE/'runtime'
sys.path.insert(0,str(RUNTIME))
from train_joint4096_var import ArchiveImages
from src.models.trainable_joint_var_tokenizer import FourScaleJointVAE


def main():
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    cfg=json.loads((BASE/'run-output/config.json').read_text())
    best=torch.load(BASE/'run-output/tokenizer-best-rfid.pt',map_location='cpu',weights_only=False,mmap=True)
    latest=torch.load(BASE/'run-output/tokenizer-last.pt',map_location='cpu',weights_only=False,mmap=True)
    dataset=ArchiveImages(cfg['val_archive'],cfg['val_index'],False,cfg['seed'],labels=cfg['val_labels'])
    indices=np.linspace(0,49999,256,dtype=int).tolist()
    images=torch.stack([dataset[i][0] for i in indices])
    model=FourScaleJointVAE().eval().requires_grad_(False).cuda()
    outputs={};codes={};timings={}
    def evaluate(key,state,bf16):
        model.load_state_dict(state['model'],strict=True)
        decoded=[];tokens=[];started=time.monotonic()
        with torch.inference_mode():
            for batch in images.split(8):
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=bf16):
                    recon=model(batch.cuda())[0]
                decoded.append(recon.float().clamp(-1,1).cpu())
                tokens.append(model.quantize.last_atom_ids.cpu())
        torch.cuda.synchronize()
        outputs[key]=torch.cat(decoded);codes[key]=torch.cat(tokens)
        timings[key]=time.monotonic()-started
        print(key,'complete',timings[key],flush=True)
    evaluate('best_bf16',best,True)
    evaluate('latest_bf16',latest,True)
    ablation='--ablation' in sys.argv
    if ablation:
        hybrid=dict(latest)
        hybrid['model']={key:(best['model'][key] if key.startswith(
            ('encoder.','quant_conv.','quantize.')) else value)
            for key,value in latest['model'].items()}
        evaluate('fixed_encoder_quantizer_latest_decoder',hybrid,True)
    else:
        evaluate('best_bf16_repeat',best,True)
        evaluate('best_fp32',best,False)
        evaluate('latest_fp32',latest,False)
    def compare(a,b):
        delta=outputs[a]-outputs[b]
        return dict(pixel_rmse=float(delta.square().mean().sqrt()),
            pixel_max_abs=float(delta.abs().max()),
            changed_token_fraction=float((codes[a]!=codes[b]).float().mean()),
            images_with_changed_tokens=int((codes[a]!=codes[b]).reshape(len(images),-1).any(1).sum()))
    report=dict(read_only=True,training_unchanged=True,images=256,microbatch=8,
        best_progress=best['progress'],latest_progress=latest['progress'],
        weight_drift_bf16=compare('best_bf16','latest_bf16'),
        timings=timings,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        no_fid_protocol_change=True,no_subset_fid_reported=True)
    if ablation:
        report['fixed_encoder_quantizer_latest_decoder']=compare('best_bf16','fixed_encoder_quantizer_latest_decoder')
    else:
        report.update(best_bf16_repeat=compare('best_bf16','best_bf16_repeat'),
            weight_drift_fp32=compare('best_fp32','latest_fp32'),
            precision_difference_at_best=compare('best_bf16','best_fp32'))
    name='assignment-ablation-report.json' if ablation else 'precision-drift-report.json'
    (BASE/name).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
