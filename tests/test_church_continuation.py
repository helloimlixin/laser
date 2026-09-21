import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from src.training.church_continuation import (
    ContinuationLearningRate, ranked_candidates, upload_checkpoints,
)


PROTOCOL = dict(sampling=dict(temperature=1., top_k=1400, top_p=1.),
                reference_sha256='reference', generated_samples=50000)


def state():
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    return optimizer, scheduler


def test_lr_is_half_cosine_without_repeated_decay_and_survives_restart():
    opt, sched = state()
    control_opt, control_sched = state()
    lr = ContinuationLearningRate(opt, sched)
    for _ in range(8):
        opt.step(); lr.step()
        control_opt.step(); control_sched.step()
        assert opt.param_groups[0]['lr'] == pytest.approx(control_opt.param_groups[0]['lr'] / 2)
    saved = copy.deepcopy((opt.state_dict(), sched.state_dict(), lr.state_dict()))
    resumed_opt, resumed_sched = state()
    resumed_opt.load_state_dict(saved[0]); resumed_sched.load_state_dict(saved[1])
    resumed_lr = ContinuationLearningRate(resumed_opt, resumed_sched, state=saved[2])
    for _ in range(8):
        opt.step(); lr.step()
        resumed_opt.step(); resumed_lr.step()
        assert resumed_opt.param_groups[0]['lr'] == opt.param_groups[0]['lr']


def test_fid_reduction_survives_resume_and_protocol_change_resets_comparison():
    opt, sched = state()
    lr = ContinuationLearningRate(opt, sched)
    assert lr.observe(1, 12., PROTOCOL)['decision'] == 'baseline'
    assert lr.observe(2, 12.1, PROTOCOL)['decision'] == 'watch'
    saved = lr.state_dict()
    lr = ContinuationLearningRate(opt, sched, state=saved)
    assert lr.observe(3, 12.2, PROTOCOL)['decision'] == 'reduced'
    assert opt.param_groups[0]['lr'] == .000125
    assert lr.observe(4, 15., dict(PROTOCOL, reference_sha256='other'))['decision'] == 'baseline'
    for _ in range(30):
        opt.step(); lr.step()
    assert opt.param_groups[0]['lr'] == 1e-6


def test_ranking_retains_three_distinct_best_evaluated_steps():
    ranked = []
    for step, fid in enumerate([15., 12., 13., 16., 11.], 1):
        ranked = ranked_candidates(ranked, fid=fid, epoch=step*10, step=step)
    assert [row['step'] for row in ranked] == [5, 2, 3]
    assert ranked_candidates(ranked, fid=11., epoch=50, step=5) == ranked
    with pytest.raises(ValueError):
        ranked_candidates(ranked, fid=float('nan'), epoch=60, step=6)


@pytest.mark.parametrize('fail', [False, True])
def test_upload_pins_files_until_commit_and_never_claims_failed_upload(tmp_path, monkeypatch, fail):
    (tmp_path/'last.pt').write_bytes(b'original')
    ranked = ranked_candidates([], fid=12., epoch=10, step=620)
    (tmp_path/ranked[0]['path']).write_bytes(b'best')
    uploaded = {}

    class Artifact:
        qualified_name = 'entity/project/checkpoints:v0'

        def __init__(self, *args, **kwargs):
            self.files = {}

        def add_file(self, path, *, name):
            self.files[name] = Path(path)

        def wait(self):
            assert self.files['last.pt'].read_bytes() == b'original'
            assert self.files['best-fid-01.pt'].read_bytes() == b'best'
            if fail:
                raise RuntimeError('Upload interrupted')
            uploaded.update({name: path.read_bytes() for name, path in self.files.items()})

    class Run:
        id = 'test-run'
        summary = {}

        def log_artifact(self, artifact, **kwargs):
            replacement = tmp_path/'replacement.pt'
            replacement.write_bytes(b'next epoch')
            replacement.replace(tmp_path/'last.pt')
            return artifact

    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(Artifact=Artifact))
    if fail:
        with pytest.raises(RuntimeError, match='interrupted'):
            upload_checkpoints(Run(),tmp_path,ranked,epoch=10,step=620,protocol=PROTOCOL)
        assert not (tmp_path/'checkpoint-upload.json').exists()
    else:
        upload_checkpoints(Run(),tmp_path,ranked,epoch=10,step=620,protocol=PROTOCOL)
        assert uploaded['last.pt'] == b'original'
        assert json.loads((tmp_path/'checkpoint-upload.json').read_text())['optimizer_step'] == 620
    assert (tmp_path/ranked[0]['path']).read_bytes() == b'best'


def test_preview_uses_100_images_and_preserves_training_rng_on_gpu(tmp_path, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required for the production preview RNG path')
    import ast
    import random
    import time
    import numpy as np
    from torchvision.utils import save_image
    from PIL import Image

    source = Path(__file__).resolve().parents[1]/'scripts/tools/continue_church_stage2.py'
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='preview_samples')
    namespace = dict(torch=torch,random=random,np=np,time=time,json=json,save_image=save_image,
                     dist=SimpleNamespace(barrier=lambda:None),
                     atomic_json=lambda path,data:path.write_text(json.dumps(data)))
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)

    class Model(torch.nn.Module):
        def sample(self, codes, **kwargs):
            assert len(codes)==100
            torch.rand(3,device=codes.device)
            return codes

    class Tokenizer:
        def decode_code(self, codes):
            return torch.rand(len(codes),3,256,256,device=codes.device)*2-1

    device=torch.device('cuda',0)
    cpu_rng=torch.get_rng_state().clone()
    gpu_rng=torch.cuda.get_rng_state(device).clone()
    model=Model().train()
    namespace['preview_samples'](model,Tokenizer(),200,12400,tmp_path,device,0,None,PROTOCOL['sampling'])
    assert model.training
    assert torch.equal(cpu_rng,torch.get_rng_state())
    assert torch.equal(gpu_rng,torch.cuda.get_rng_state(device))
    receipt=json.loads((tmp_path/'preview-latest.json').read_text())
    assert receipt['samples']==100
    assert Image.open(receipt['path']).size==(2582,2582)


def test_actual_training_loop_preview_cadence_counts_successful_steps_not_epochs():
    import ast
    source=Path(__file__).resolve().parents[1]/'scripts/tools/continue_church_stage2.py'
    main=next(n for n in ast.parse(source.read_text()).body
              if isinstance(n,ast.FunctionDef) and n.name=='main')
    epoch_loop=next(n for n in main.body
                    if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='epoch_index')
    update_loop=next(n for n in epoch_loop.body
                     if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='update_index')
    preview=next(n for n in update_loop.body if isinstance(n,ast.If) and
                 any(isinstance(c,ast.Call) and isinstance(c.func,ast.Name) and
                     c.func.id=='preview_samples' for c in ast.walk(n)))
    calls=[]
    env=dict(args=SimpleNamespace(preview_every_steps=200,decode_batch_size=8),
             preview_samples=lambda *a:calls.append(a[3]),
             model=None,tokenizer=None,out=None,device=None,rank=0,run=None,sampling={})
    code=compile(ast.Module(body=[preview],type_ignores=[]),str(source),'exec')
    for step,epoch,updated in [(199,3,True),(200,3,True),(200,3,False),
                               (201,200,True),(400,7,True)]:
        env.update(step=step,epoch_fraction=epoch,metrics={'optimizer_updated':updated})
        exec(code,env)
    assert calls==[200,400]
