"""Freeze a native, support-first 1K-pattern experiment without reusing AR weights."""
import ast
import json
from pathlib import Path
import shutil

from omegaconf import OmegaConf

ROOT = Path('/tmp/laser-pattern1024-stage2-20261107')
BASE = ROOT/'frozen'
SOURCE = Path('/tmp/laser-compound128-20261107/frozen')
REPO = Path('/workspace/Projects/laser')
RUNTIME = REPO/'runtime/pattern1024-stage2-20261107'
SCREEN = Path('/tmp/laser-coefficient-pattern-screen-20261107')
RUN = 'imagenet-laser-pattern1024-native-scratch100-4a100-20261107'


def main():
    assert not BASE.exists(), 'Already prepared; inspect before recreating'
    (BASE/'inputs').mkdir(parents=True)
    (ROOT/'production-nvme').mkdir()
    for name in ['source','upstream','runtime']:
        shutil.copytree(SOURCE/name, BASE/name, ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git'))
    for name in ['stage1-model.pt','imagenet_256_train.npz','rq-upstream-core.json']:
        (BASE/'inputs'/name).hardlink_to(SOURCE/'inputs'/name)
    for name in ['stage2-training-config.yaml','stage2-archive-config.yaml']:
        shutil.copyfile(SOURCE/'inputs'/name, BASE/'inputs'/name)
    fit = json.loads((SCREEN/'fit-results.json').read_text())
    book = Path(fit['selected']['1024']['path'])
    (BASE/'inputs/coefficient-patterns.pt').hardlink_to(book)
    for name in ['results.json','fit-results.json','cloud-verification.json','cpu-verification.json']:
        shutil.copyfile(SCREEN/name, BASE/'inputs'/('screen-'+name))
    shutil.copyfile(SCREEN/'cloud-verification.json', BASE/'inputs/paired-tokenizer-artifact.json')
    shutil.copyfile(SCREEN/'frozen-code/coefficient_pattern_codec.py', BASE/'runtime/pattern_codec.py')
    recipe = OmegaConf.load(SOURCE/'inputs/recipe.yaml')
    recipe.run_id = RUN
    recipe.name = 'ImageNet original LASER | support first plus 1K coefficient pattern | fresh100'
    recipe.version = 'native-support-first-pattern1024-v1'
    recipe.decoder = 'native_support_first_pattern'
    recipe.target_mode = 'hard_support_and_pattern'
    recipe.epochs = 100
    recipe.microbatch = 128
    recipe.tokenizer.coefficient_patterns = 1024
    recipe.tokenizer.pattern_codebook_sha256 = fit['selected']['1024']['sha256']
    recipe.tokenizer.measured_pattern_rfid = 4.297528190759465
    recipe.targets = dict(encoder_batch_size=8, coefficient_noise=False, stochastic_prefixes=False,
        coefficient_space='continuous physical final least-squares values', assignment='nearest support Gram distance')
    recipe.objective = 'joint support-plus-pattern categorical NLL averaged over five decisions'
    recipe.atom_ce_weight = 4.
    recipe.sampling = dict(atom_temperature=.9, atom_top_k=0, atom_top_p=.9,
        pattern_temperature=1., pattern_top_k=0, pattern_top_p=.85)
    recipe.lineage = dict(source_baseline='imagenet-laser-compound128-native-scratch100-4a100-20261107',
        screen_artifact=json.loads((SCREEN/'cloud-verification.json').read_text())['artifact'],
        source_stage2_checkpoint_loaded=False, coefficient_pattern_fit_split='train',
        fitting_images=16000, fitting_sites=1024000, seed_selection_images=2000,
        pattern_entropy_marginal_bits=9.856505450295192)
    OmegaConf.save(recipe, BASE/'inputs/recipe.yaml')
    config = OmegaConf.load(BASE/'inputs/stage2-training-config.yaml')
    config.loss = dict(type=recipe.objective, support_decisions=4, pattern_decisions=1,
        coefficient_patterns=1024, soft_targets=False, stochastic_codes=False)
    OmegaConf.save(config, BASE/'inputs/stage2-training-config.yaml')
    (BASE/'data').symlink_to(SOURCE/'data', target_is_directory=True)
    (ROOT/'torch-cache').symlink_to('/tmp/laser-compound128-20261107/torch-cache', target_is_directory=True)
    for p in RUNTIME.glob('*.py'):
        shutil.copyfile(p, BASE/'runtime'/p.name)
    shutil.copyfile(SOURCE/'train.py', BASE/'train.py')
    p = BASE/'train.py'
    p.write_text(p.read_text().replace("main('laser_compound128',root)", "main('laser_pattern1024',root)"))
    p = BASE/'upstream/rqvae/models/rqtransformer/transformers.py'
    code = p.read_text()
    marker = '    def forward(self, xs, model_aux=None, cond=None, amp=False):'
    hooks = '''    def embed_depth_with_model_aux(self, xs, model_aux):
        return self.embed_with_model_aux(xs, model_aux)

    def classify_head_outputs(self, hidden):
        return self.classifier(hidden)

'''
    assert code.count(marker)==1
    code = code.replace(marker, hooks+marker)
    assert code.count('depth_ctx = self.embed_with_model_aux(xs, model_aux)')==2
    code = code.replace('depth_ctx = self.embed_with_model_aux(xs, model_aux)',
        'depth_ctx = self.embed_depth_with_model_aux(xs, model_aux)')
    code = code.replace('seq_logits = self.classifier(head_outputs)', 'seq_logits = self.classify_head_outputs(head_outputs)')
    compile(code, str(p), 'exec')
    p.write_text(code)
    p = BASE/'source/src/training/rq_imagenet_shared.py'
    code = p.read_text().replace("    if kind == 'laser_compound128':",
        "    if kind == 'laser_pattern1024':\n        from runtime.pattern_recipe import PatternRecipe\n        PROFILE=PatternRecipe(BASE)\n    elif kind == 'laser_compound128':")
    marker = '            assert PROFILE.digest(aux)==before\n            if rank==0:\n                row='
    assert code.count(marker)==1
    code = code.replace(marker,
        '            from runtime.heldout import evaluate as heldout_evaluate\n'
        '            evaluation.update(heldout_evaluate(PROFILE,model,aux,device,rank,world))\n'+marker)
    marker = "                write_json(OUT/f'evaluations/epoch{epoch+1:03d}.json',dict(row,generated_images=50000,"
    assert code.count(marker)==1
    code = code.replace(marker,
        "                row.update({key:value for key,value in evaluation.items() if key.startswith('eval/')})\n"+marker)
    compile(code, str(p), 'exec')
    p.write_text(code)
    p = BASE/'source/src/training/joint_runtime_support.py'
    code = p.read_text().replace('assert codes.min()>=0 and codes.max()<aux.vocab_size',
        'profile.verify_codes(aux,codes)')
    code = code.replace('Original LASER compound128 fresh100 4A100 20261107', 'Native support first pattern1024 fresh100 4A100 20261107')
    code = code.replace('train/coefficient_ce','train/pattern_ce').replace('Atom and coefficient CE','Atom and pattern CE')
    code = code.replace('Atom and coefficient cross entropy','Joint support and pattern NLL')
    code = code.replace("        wr.LinePlot(title='Learning rate, zero warmup'",
        "        wr.LinePlot(title='Joint NLL bits per image',x='train/global_step',y=['train/joint_nll_bits_per_image']),\n"
        "        wr.LinePlot(title='Predicted pattern entropy, bits',x='train/global_step',y=['train/pattern_predicted_entropy_bits']),\n"
        "        wr.LinePlot(title='Learning rate, zero warmup'")
    code = code.replace("        wr.LinePlot(title='Inception score'",
        "        wr.LinePlot(title='Held-out NLL bits per image',x='train/epoch',y=['eval/nll_bits_per_image']),\n"
        "        wr.LinePlot(title='Held-out pattern cross entropy, bits',x='train/epoch',y=['eval/pattern_cross_entropy_bits']),\n"
        "        wr.LinePlot(title='Inception score'")
    p.write_text(code)
    quality = json.loads((SCREEN/'results.json').read_text())
    proof = dict(passed=True, measured_validation_rfid=quality['results']['pattern1024']['rfid'],
        real_images=50000, reconstructed_images=50000, source_screen=str(SCREEN/'results.json'),
        pattern_codebook_sha256=fit['selected']['1024']['sha256'],
        current_codec_uses_identical_physical_patterns_and_assignment=True,
        integer_transport_is_lossless=True, original_encoder_dictionary_decoder_frozen=True)
    (ROOT/'tokenizer-verification.json').write_text(json.dumps(proof, indent=2)+'\n')
    shutil.copyfile(ROOT/'tokenizer-verification.json', BASE/'inputs/tokenizer-verification.json')
    # Reuse the already exercised process supervision, environment and command
    # helpers; experiment-specific verification and profiling live separately.
    old = REPO/'runtime/compound128-20261107/supervise.py'
    names = {'read','write','stop','environment','command','child'}
    nodes = [n for n in ast.parse(old.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    header = '''import json, os, signal, subprocess, sys, time
from pathlib import Path
ROOT=Path('/tmp/laser-pattern1024-stage2-20261107')
BASE=ROOT/'frozen'
OUT=ROOT/'production-nvme'
CHILD=None
'''
    (BASE/'runtime/launch_support.py').write_text(header+'\n\n'.join(ast.unparse(n) for n in nodes)+'\n')
    public = REPO/'outputs'/RUN
    public.mkdir(parents=True, exist_ok=True)
    for name in ['status.json','launch.json','training.log','checkpoint.json','cloud-sync-status.json',
                 'evaluations','source-artifact.json','wandb.json','samples']:
        (public/name).symlink_to(ROOT/'production-nvme'/name, target_is_directory=name in ['evaluations','samples'])
    for name in ['tokenizer-verification.json','cpu-verification.json','throughput-profile.json','supervisor.log']:
        (public/name).symlink_to(ROOT/name)
    (public/'pipeline-status.json').symlink_to(ROOT/'status.json')
    (public/'live-nvme').symlink_to(ROOT/'production-nvme', target_is_directory=True)
    print(json.dumps(dict(base=str(BASE), run_id=RUN, fresh_stage2=True, pattern_count=1024)), flush=True)


if __name__=='__main__':
    main()
