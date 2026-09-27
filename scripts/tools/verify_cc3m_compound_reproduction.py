#!/usr/bin/env python3
"""CPU contract checks for the CC3M compound launch; run in its frozen runtime."""
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

base = Path(os.environ['CC3M_BASE'])
spec = importlib.util.spec_from_file_location('cc3m_driver', base/'runtime/train_cc3m_compound.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
torch = m.torch
torch.set_num_threads(2)
checks = []

aux = m.auxiliary('cpu')
assert aux.dictionary.shape == (256, 16384)
assert not any(p.requires_grad for p in aux.parameters())
with torch.device('meta'):
    net = m.model()
assert list(net.block_size) == [8, 8, 4]
assert net.block_size_cond == 32 and net.vocab_size_cond == 16384
assert len(net.body_transformer.blocks) == 26 and len(net.head_transformer.blocks) == 4
assert len(net.coeff_classifier) == 4 and len(net.coeff_micro_transformer.blocks) == 2
parameters = sum(p.numel() for p in net.parameters())
checks.append('exact tokenizer hash, frozen stage one, architecture and four compound heads')
del net

tok = m.tokenizer(0.1)
draws = [tok.encode('A multicolored butterfly flying through a beautiful garden').ids for _ in range(100)]
assert all(len(x) == 32 and min(x) >= 0 and max(x) < 16384 for x in draws)
assert len({tuple(x) for x in draws}) > 1
fixed = m.tokenizer()
assert fixed.encode('A red car').ids == m.ref.encode_cc3m_prompts(['A red car'])[0].tolist()
data = m.TarImages(m.DATA/'wds/cc3m-train-0000.tar', training=True, limit=2)
first, caption = data[0]
again, _ = data[0]
assert first.shape == (2,3,256,256) and torch.equal(first, again) and caption
checks.append('official BPE dropout and deterministic two-crop image cache')

coeffs = torch.tensor([0.1, -0.2, 0.3, -0.4]).reshape(1,1,1,4)
_, probs = aux.compound_coeff_ids(coeffs, stochastic=False, temp=0.5)
expected = (-(coeffs[...,None]-aux.coeff_bins)**2 / 0.5).softmax(-1)
assert torch.equal(probs, expected)
torch.testing.assert_close(probs.sum(-1), torch.ones_like(coeffs))
checks.append('FFHQ normalized coefficient soft targets')

small_cfg = m.ref.RQTransformerConfig.create(m.OmegaConf.create(dict(
    type='rq-transformer', block_size=[8,8,4], embed_dim=64, input_embed_dim=256,
    shared_tok_emb=True, shared_cls_emb=True, input_emb_vqvae=True,
    head_emb_vqvae=True, cumsum_depth_ctx=True, vocab_size=16384,
    vocab_size_cond=16384, block_size_cond=32,
    body=dict(n_layer=1,block=dict(n_head=4)), head=dict(n_layer=1,block=dict(n_head=4)))))
small = m.ref.CompoundLaserRQTransformer(small_cfg,16384,2048,
    micro_transformer_layers=2,depth_specific_coeff_heads=True).eval()
atoms = torch.randint(0,16384,(1,8,8,4))
coefficients = torch.rand(1,8,8,4)*2-1
text = m.ref.encode_cc3m_prompts(['A red car'])
loss, _ = m.loss_for(small,aux,(atoms,coefficients,text),torch.device('cpu'),5.)
loss.backward()
assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in small.parameters())
with torch.no_grad():
    ids,_ = aux.compound_coeff_ids(coefficients,stochastic=False)
    packed = atoms*2048+ids
    output,_ = small(packed,model_aux=aux,cond=text)
    changed = packed.clone().reshape(1,-1)
    changed[:,37:] = (changed[:,37:]+2048) % (16384*2048)
    perturbed,_ = small(changed.reshape_as(packed),model_aux=aux,cond=text)
    torch.testing.assert_close(output['atom_logits'].reshape(1,256,-1)[:,:37],
                               perturbed['atom_logits'].reshape(1,256,-1)[:,:37],rtol=0,atol=0)
    small.init_cache()
    for w in range(2):
        for depth in range(4):
            hidden = small.cached_head_output(packed,aux,text,(0,w,depth),amp=False)
            atom_logits = small.classifier(hidden)
            coeff_logits = small.coefficient_logits(hidden,aux.dictionary.t()[atoms[:,0,w,depth]],depth_index=depth)
            torch.testing.assert_close(atom_logits,output['atom_logits'][:,0,w,depth],atol=2e-5,rtol=1e-4)
            torch.testing.assert_close(coeff_logits,output['coeff_logits'][:,0,w,depth],atol=2e-5,rtol=1e-4)
checks.append('text-conditioned compound gradients, causal prefix and cached/full logits')
del small
del aux

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    meta = dict(identity=m.cache_identity(), passed=True, lengths=[2,3],
                shards=['a.tar','b.tar'], coeff_scales=[2.,3.,4.,5.])
    m.write_json(root/'ready.json', meta)
    for filename, size, offset in [('a',2,0),('b',3,2)]:
        atoms = torch.zeros(size,2,8,8,4,dtype=torch.int16)
        for row in range(size):
            for view in range(2): atoms[row,view].fill_((offset+row)*10+view)
        text = torch.arange(100,dtype=torch.int16).reshape(1,100,1).expand(size,100,32)
        torch.save(dict(atoms=atoms, coeffs=torch.ones(size,2,8,8,4), text_tokens=text), root/(filename+'.pt'))
    for epoch in [0,1,99]:
        cache = m.CachedPairs(root, epoch)
        for index in range(5):
            atoms, coeff, text = cache[index]
            assert atoms[0,0,0].item() == index*10+(epoch+index)%2
            assert text[0].item() == epoch
            torch.testing.assert_close(coeff[0,0], 1/torch.tensor([2.,3.,4.,5.]))
checks.append('shard boundaries, crop alternation, text views and coefficient scaling')

for world in [4,8,16]:
    batch, accumulation = 16, 2048//(world*16)
    permutation = torch.randperm(4*2048, generator=torch.Generator().manual_seed(5))
    rank_rows = [permutation.reshape(-1,accumulation,world,batch)[:,:,r] for r in range(world)]
    recovered = torch.stack(rank_rows,dim=2).reshape(-1)
    assert torch.equal(recovered, permutation)
    for offset in [1,3]:
        resumed = torch.stack([x[offset:] for x in rank_rows],dim=2).reshape(-1)
        assert torch.equal(resumed, permutation[offset*2048:])
checks.append('exact global batch and resume cursor on 4, 8 and 16 GPUs')

with tempfile.TemporaryDirectory() as tmp:
    local = Path(tmp)
    tiny = torch.nn.Linear(2,2)
    opt = torch.optim.AdamW(tiny.parameters(),lr=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=100)
    opt.zero_grad(); tiny(torch.ones(1,2)).sum().backward(); opt.step(); scheduler.step()
    best = dict(fid=float('inf'),clip_score=-float('inf'))
    uploads = []
    def upload(wb, paths, **kwargs):
        uploads.append(dict(paths=[p.name for p in paths], **kwargs))
    with patch.object(m,'BASE',local), patch.object(m.ref,'upload_checkpoints',upload), \
         patch.object(torch.cuda,'get_rng_state',torch.get_rng_state):
        m.checkpoint(tiny,opt,scheduler,1,1,{},local,object(),best,dict(fid=20.,clip_score=.20))
        assert set(uploads[-1]['aliases']) >= {'last','latest','best-fid','best-clip'}
        m.checkpoint(tiny,opt,scheduler,2,1,{},local,object(),best,dict(fid=25.,clip_score=.25))
        assert 'best-clip' in uploads[-1]['aliases'] and 'best-fid' not in uploads[-1]['aliases']
        m.checkpoint(tiny,opt,scheduler,3,1,{},local,object(),best)
        assert set(uploads[-1]['paths']) == {'last.pt','best-fid.pt','best-clip.pt'}
        assert torch.load(local/'best-fid.pt',weights_only=False)['step'] == 1
        assert torch.load(local/'best-clip.pt',weights_only=False)['step'] == 2
        last = torch.load(local/'last.pt',weights_only=False)
        assert last['step'] == 3 and last['best']['evaluated_step'] == 2
        assert last['optimizer']['state'] and last['rng']
        # A rejected/failed upload must never advance the durable receipt.
        with patch.object(m.ref,'upload_checkpoints',side_effect=RuntimeError('upload failed')):
            try: m.checkpoint(tiny,opt,scheduler,4,1,{},local,object(),best)
            except RuntimeError: pass
            else: raise AssertionError('Upload failure was swallowed')
        assert json.loads((local/'last-upload.json').read_text())['step'] == 3
checks.append('best FID/CLIP retention, resumable last state and confirmed-upload receipts')

report = dict(passed=True, checks=checks, parameters=parameters, tokenizer_sha256=m.TOKENIZER_SHA,
              gpu_training_verified=False)
m.write_json(base/'cpu-preflight.json',report)
print(json.dumps(report,indent=2))
