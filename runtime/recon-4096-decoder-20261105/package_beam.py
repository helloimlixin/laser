"""Package a sub4.73 compact tokenizer without changing its native soft AR teacher."""
from pathlib import Path
import json
import shutil
import sys
import torch

ROOT=Path('/workspace/Projects/laser')
sys.path.insert(0,str(ROOT/'runtime/recon-4096-quality-20261105'))
from evaluate import SOURCE,book,module_sha,write,sha,TarImageNet,load_config,create_transforms
from src.models.laser_joint_tokenizer import FrozenJointLaserTokenizer
from paired_tokenizer import PairedReconstructionTokenizer

BASE=Path('/tmp/laser-recon-4096-beam-20261105')
PUBLIC=ROOT/'outputs/imagenet-native4096-below473-20261105'


def main():
    PUBLIC.mkdir(exist_ok=True)
    report=json.loads((BASE/'full-results.json').read_text())
    selected=report['selected'];result=report['results'][selected]
    assert result['rfid']<4.73 and result['images']==50000
    source=Path(result['codebook_path'])
    spec=torch.load(source,map_location='cpu',weights_only=True)
    assert spec['dictionary'].shape==(256,4096) and spec['coefficients'].shape==(4,16385)
    spec.update(encoding_beam_width=1,ar_training_target_mode='native_stochastic_soft',
        reconstruction_policy_file='tokenizer-policy.json')
    checkpoint=PUBLIC/'joint-codebook.pt';torch.save(spec,checkpoint)
    for key in ['dictionary','atom_ids','coefficients']:
        assert torch.equal(spec[key],torch.load(source,map_location='cpu',weights_only=True)[key])
    temperatures=[0.15819901962280272,0.13549085464477534,0.08794634933471679,0.055382978057861326]
    policy=dict(atoms=4096,vocabulary=16385,residual_depth=4,coefficient_levels_per_atom=4,
        source_checkpoint_artifact='helloimlixin-rutgers/laser/imga16384k4altbn64-b128-b300-20260830000755-stage1-checkpoints:v9',
        source_checkpoint_file='best_rfid_slot1_model.pt',source_checkpoint_sha256=report['source_checkpoint_sha256'],
        joint_codebook_sha256=sha(checkpoint),
        reconstruction=dict(encoding='beam search with greedy anchor; fixed physical integer codewords',
            beam_width=result['reconstruction_beam_width'],rfid=result['rfid'],images=50000,
            per_image_coefficient_refitting=False,decoder_weights='unchanged source'),
        ar_teacher=dict(mode='native_stochastic_soft',encoding_beam_width=1,
            temperatures=temperatures,stochastic_prefixes=True,hard_beam_targets=False),
        note='Reconstruction search is separate from training targets. The codebook encoding_beam_width must stay1.',
        encoder_decoder_weights_unchanged=True,transformer_training_started=False)
    write(PUBLIC/'tokenizer-policy.json',policy)
    for n in ['full-results.json','screen-results.json','candidates.json']:
        shutil.copyfile(BASE/n,PUBLIC/n)
    shutil.copyfile(Path(__file__).with_name('paired_tokenizer.py'),PUBLIC/'paired_tokenizer.py')
    shutil.copyfile(ROOT/'src/models/laser_joint_search.py',PUBLIC/'beam_search.py')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda',0)
    plain=FrozenJointLaserTokenizer(SOURCE/'inputs/stage1-model.pt',source).to(device).eval().requires_grad_(False)
    paired=PairedReconstructionTokenizer(SOURCE/'inputs/stage1-model.pt',checkpoint,
        reconstruction_beam_width=policy['reconstruction']['beam_width']).to(device).eval().requires_grad_(False)
    config=load_config(SOURCE/'inputs/stage2-training-config.yaml')
    data=TarImageNet(BASE/'data','train',create_transforms(config.dataset,split='val'))
    indices=json.loads((BASE/'candidates.json').read_text())['screen_indices'][:2]
    images=torch.stack([data[i][0] for i in indices]).to(device)
    with torch.inference_mode():
        latent=plain.encode(images);torch.testing.assert_close(latent,paired.encode(images),rtol=0,atol=0)
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(261105473)
            p0,c0=plain.quantizer.get_soft_codes(latent,temperatures,stochastic=True,chunk_size=128)
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(261105473)
            p1,c1=paired.quantizer.get_soft_codes(latent,temperatures,stochastic=True,chunk_size=128)
        assert torch.equal(p0,p1) and torch.equal(c0,c1)
        entropy=float(-(p1*p1.clamp_min(1e-30).log()).sum(-1).mean());assert entropy>0
        greedy=plain.quantizer.quantize(latent,chunk_size=1024)
        encoded=paired.quantize_reconstruction(latent)
        error=(latent-encoded['quantized']).square().sum(-1)
        greedy_error=(latent-greedy['quantized']).square().sum(-1)
        assert bool((error<=greedy_error+1e-6).all())
        torch.testing.assert_close(plain.decode_code(encoded['codes']),paired.decode_code(encoded['codes']),rtol=0,atol=0)
        assert module_sha(plain.backbone)==module_sha(paired.backbone)
    proof=dict(passed=True,encoder_decoder_bitwise_identical=True,
        fixed_physical_codewords_bitwise_identical=True,native_soft_targets_bitwise_equal=True,
        native_stochastic_codes_bitwise_equal=True,mean_target_entropy=entropy,
        ar_teacher_is_not_hard_beam=True,reconstruction_beam_width=policy['reconstruction']['beam_width'],
        every_checked_latent_no_worse_than_greedy=True,codebook_encoding_beam_width=1,
        full_validation_rfid=result['rfid'],beats_original_rq=True,validation_images=50000,
        original_rq_reference=4.73,transformer_training_started=False)
    write(PUBLIC/'paired-teacher-verification.json',proof)
    print(json.dumps(dict(result=result,policy=policy,proof=proof)),flush=True)


if __name__=='__main__':main()
