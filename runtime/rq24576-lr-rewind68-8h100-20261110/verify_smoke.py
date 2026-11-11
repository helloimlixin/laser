"""Verify exact source state and independent RNG restore before the long branch."""
import hashlib
import json
from pathlib import Path
import torch

HERE=Path(__file__).resolve().parent


def main():
    plan=json.loads((HERE/'plan.json').read_text());base=Path(plan['base']);out=Path(plan['output'])
    proof=json.loads((base/'smoke/verification.json').read_text())
    assert proof['passed'] and proof['global_step']==plan['source_step']+2
    assert proof['optimizer_state_count']==780 and proof['all_adam_steps']==plan['source_step']+2
    assert proof['frozen_tokenizer_unchanged']
    parent=torch.load(plan['source_checkpoint'],map_location='cpu',mmap=True,weights_only=False)
    ranks=[]
    for rank in range(8):
        startup=json.loads((base/f'smoke/startup-rank{rank}.json').read_text())
        rng=json.loads((base/f'smoke/rng-startup-rank{rank}.json').read_text())
        interface=json.loads((base/f'smoke/generation-interface-rank{rank}.json').read_text())
        expected=hashlib.sha256(parent['rng_state_by_rank'][rank]['cuda'].numpy().tobytes()).hexdigest()
        assert startup['global_step']==plan['source_step'] and startup['restored_optimizer_age']==plan['source_step']
        assert startup['optimizer_state_count_at_start']==780 and not startup['fresh_stage2'] and not startup['fresh_optimizer']
        assert startup['world_size']==8 and startup['global_batch']==2048 and startup['microbatch']==64
        assert startup['training_autocast_dtype']=='bfloat16' and startup['source_CNN_unchanged']
        assert rng['rng_state_sha256']==expected and rng['training_rng_restored_from_checkpoint'] and rng['training_rng_unchanged_by_audit']
        assert interface['passed'] and interface['vocabulary']==24576 and interface['fixed_codeword_decoding']
        ranks.append(dict(rank=rank,source_optimizer_age=startup['restored_optimizer_age'],
            cuda_rng_restored_exactly=True,source_cuda_rng_sha256=expected,generation_verified=True))
    assert len({r['source_cuda_rng_sha256'] for r in ranks})==8
    result=dict(passed=True,source_epoch=plan['source_epoch'],source_step=plan['source_step'],
        model_and_optimizer_restored=True,optimizer_states=780,all_adam_steps=plan['source_step']+2,
        all_eight_saved_cuda_rng_states_restored_exactly=True,tokenizer_unchanged=True,
        original_global_batch_and_hardware_preserved=True,ranks=ranks,
        production_will_restart_from_parent_not_validation_updates=True)
    (out/'resume-preflight-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
