import copy
import json
import os
from pathlib import Path

import pytest
import torch
from torch import nn
from omegaconf import OmegaConf

from src import rq_upstream
from src.models.laser_packed_integer_tokenizer import (
    FrozenPackedIntegerLaserTokenizer, pack_pair_ids, unpack_pair_ids,
)
from src.models.laser_tokenizer_adapter import FrozenLaserTokenizer
from src.models.laser_pair_id_head import ProjectedPairIDLinear
from src.training.laser_packed_integer_recipe import PackedIntegerLaserRecipe
from src.training.packed_pair_loss import sparse_pair_cross_entropy


def tokenizer():
    aux = FrozenPackedIntegerLaserTokenizer.__new__(FrozenPackedIntegerLaserTokenizer)
    nn.Module.__init__(aux)
    aux.num_atoms, aux.coeff_vocab_size, aux.sparsity_level = 5, 9, 4
    aux.vocab_size = 45
    aux.register_buffer('dictionary', torch.nn.functional.normalize(torch.randn(7, 5), dim=0))
    aux.register_buffer('coeff_bins', torch.linspace(-3, 3, 9))
    aux.register_buffer('coeff_scales', torch.tensor([3., 2., 1., .5]))
    aux.encoder = aux.decoder = aux.quant_conv = aux.post_quant_conv = nn.Identity()
    return aux.requires_grad_(False).eval()


def profile(tmp_path, *, projected=False, mode='soft_coefficient'):
    inputs = tmp_path/'inputs'
    inputs.mkdir(exist_ok=True)
    recipe = dict(decoder='native_packed_pair_integer', auxiliary_loss_terms=0,
        run_id='toy', name='toy', microbatch=2, epochs=2, lr_schedule_epochs=100,
        tokenizer=dict(num_atoms=5, coefficient_bins=9, vocab_size=45, depth=4),
        evaluation=dict(batch_size=2), targets=dict(mode=mode, temperature=.01125,
            encoder_batch_size=2, loss_row_chunk_size=3),
        sampling=dict(temperature=.9, top_k=7, top_p=.95),
        execution=dict(classifier='projected_global_pair_ids' if projected else 'native_dense',
            projection_dim=6))
    OmegaConf.save(OmegaConf.create(recipe), inputs/'recipe.yaml')
    return PackedIntegerLaserRecipe(tmp_path)


def architecture():
    from rqvae.utils.config import augment_arch_defaults
    return augment_arch_defaults(OmegaConf.create(dict(type='rq-transformer',
        block_size=[2, 2, 4], embed_dim=12, input_embed_dim=7,
        shared_tok_emb=True, shared_cls_emb=True, input_emb_vqvae=True,
        head_emb_vqvae=True, cumsum_depth_ctx=True, vocab_size=5,
        vocab_size_cond=1000, block_size_cond=1,
        body=dict(n_layer=1, block=dict(n_head=3, resid_pdrop=0.)),
        head=dict(n_layer=1, block=dict(n_head=3, resid_pdrop=0.)))))


def test_every_atom_and_bin_gets_a_distinct_integer_and_round_trips():
    atoms = torch.arange(5).repeat_interleave(512)
    bins = torch.arange(512).repeat(5)
    codes = pack_pair_ids(atoms, bins, 5, 512)
    assert torch.equal(codes, torch.arange(5*512))
    a, b = unpack_pair_ids(codes, 5, 512)
    assert torch.equal(a, atoms) and torch.equal(b, bins)
    codes = pack_pair_ids(torch.tensor([0, 16383]), torch.tensor([0, 511]), 16384, 512)
    assert codes.tolist() == [0, 8388607]
    for bad in [-1, 8388608]:
        with pytest.raises(ValueError): unpack_pair_ids(torch.tensor([bad]), 16384, 512)
    with pytest.raises(ValueError): pack_pair_ids(torch.tensor([5]), torch.tensor([0]), 5, 512)
    with pytest.raises(ValueError): pack_pair_ids(torch.tensor([0]), torch.tensor([512]), 5, 512)


def test_packing_preserves_physical_reconstruction_and_causal_prefix_embeddings():
    torch.manual_seed(737)
    aux = tokenizer()
    codes = torch.randint(45, (2, 2, 2, 4))
    atoms, bins = unpack_pair_ids(codes, 5, 9)
    scalar = torch.empty(2, 2, 2, 8, dtype=torch.long)
    scalar[..., 0::2], scalar[..., 1::2] = atoms, bins + 5
    expected = FrozenLaserTokenizer.decode_tokens(aux, scalar)
    torch.testing.assert_close(aux.decode_code(codes), expected, rtol=0, atol=0)
    full = aux.get_code_emb_with_depth(codes)[0]
    for depth in range(1, 5):
        torch.testing.assert_close(aux.get_code_emb_with_depth(codes[..., :depth])[0],
                                  full[..., :depth, :], rtol=0, atol=0)


@pytest.mark.parametrize('row_chunk', [1, 3, 128])
def test_sparse_single_softmax_loss_and_gradient_equal_native_dense_target_ce(row_chunk):
    torch.manual_seed(7371)
    logits = torch.randn(2, 2, 2, 4, 45, requires_grad=True)
    other = logits.detach().clone().requires_grad_()
    atoms = torch.randint(5, logits.shape[:-1])
    p = torch.randn(*atoms.shape, 9).softmax(-1)
    # Also verify the exact general weighted-target formula.
    p *= torch.rand(*atoms.shape, 1) + .5
    ids = atoms[..., None] * 9 + torch.arange(9)
    dense = torch.zeros_like(logits).scatter(-1, ids, p)
    expected = -(other.log_softmax(-1) * dense).sum(-1).mean()
    actual = sparse_pair_cross_entropy(logits, atoms, p, row_chunk)
    torch.testing.assert_close(actual, expected, rtol=2.e-6, atol=2.e-6)
    (actual*.37).backward()
    (expected*.37).backward()
    torch.testing.assert_close(logits.grad, other.grad, rtol=3.e-6, atol=1.e-7)


def test_one_joint_normalizer_includes_other_atoms_and_all_coefficient_bins():
    logits = torch.zeros(1, 15, requires_grad=True)
    atoms, p = torch.tensor([1]), torch.tensor([[.1, .2, .4, .2, .1]])
    loss = sparse_pair_cross_entropy(logits, atoms, p)
    loss.backward()
    assert (logits.grad[0, :5] > 0).all() and (logits.grad[0, 10:] > 0).all()
    changed = logits.detach().clone()
    changed[:, :5] += 5
    assert sparse_pair_cross_entropy(changed, atoms, p) > loss.detach()


@pytest.mark.parametrize('projected', [False, True])
@torch.no_grad()
def test_actual_upstream_forward_cache_and_sampler_use_four_pair_ids(tmp_path, projected):
    torch.manual_seed(7372)
    recipe, aux = profile(tmp_path, projected=projected), tokenizer()
    arch = architecture()
    model = recipe.model(arch).eval()
    assert type(model) is rq_upstream.RQTransformer
    assert arch.vocab_size == 5 and model.block_size == torch.Size([2, 2, 4])
    assert model.config.vocab_size == model.vocab_size == [45]*4
    assert not any('coeff' in name or 'adapter' in name for name, _ in model.named_parameters())
    codes, labels = torch.randint(45, (2, 2, 2, 4)), torch.tensor([0, 999])
    expected = model(codes, aux, labels)
    assert expected.shape == (2, 2, 2, 4, 45)
    prefix = torch.zeros_like(codes)
    model.init_cache()
    for h in range(2):
        for w in range(2):
            for d in range(4):
                actual = model.cached_forward(prefix[:, :h+1], aux, labels, sample_loc=(h,w,d))
                torch.testing.assert_close(actual, expected[:,h,w,d], rtol=1.e-5, atol=2.e-6)
                prefix[:,h,w,d] = codes[:,h,w,d]
    torch.manual_seed(7373)
    cached = model.sample(torch.zeros_like(codes), aux, labels, temperature=.9, top_k=7, top_p=.95)
    torch.manual_seed(7373)
    uncached = model.sample(torch.zeros_like(codes), aux, labels, temperature=.9, top_k=7, top_p=.95, cached=False)
    assert torch.equal(cached, uncached)
    # A future pair cannot change the current prediction.
    changed = codes.clone()
    changed[:,0,0,2:] = (changed[:,0,0,2:] + 11) % 45
    torch.testing.assert_close(model(changed,aux,labels)[:,0,0,:3], expected[:,0,0,:3], rtol=0,atol=0)


@pytest.mark.parametrize('projected', [False, True])
@pytest.mark.parametrize('mode', ['hard', 'soft_coefficient'])
def test_native_training_backward_matches_full_joint_target_and_all_parameter_gradients(tmp_path, projected, mode):
    from src.training.rq_imagenet_shared import training_backward
    torch.manual_seed(7374)
    recipe, aux = profile(tmp_path, projected=projected, mode=mode), tokenizer()
    recipe.shape = (2, 2, 4)
    model = recipe.model(architecture()).eval()
    reference = copy.deepcopy(model)
    images, labels = torch.randn(2,7,2,2), torch.tensor([0,999])
    targets,codes = recipe.targets(aux,images)
    recipe.verify_targets(aux,targets,codes)
    recipe.targets = lambda aux,images:(targets,codes)
    actual = training_backward(recipe,model,aux,images,labels,
        torch.amp.GradScaler('cuda',enabled=False),.37,amp=False)
    if mode == 'soft_coefficient':
        ids = targets['atoms'][...,None]*9+torch.arange(9)
        dense = torch.zeros(*codes.shape,45).scatter(-1,ids,targets['coefficient_probabilities'])
        expected = reference.compute_loss(reference(codes,aux,labels),dense,use_soft_target=True)
    else:
        expected = reference.compute_loss(reference(codes,aux,labels),codes,use_soft_target=False)
    (expected*.37).backward()
    torch.testing.assert_close(actual,expected,rtol=2.e-6,atol=2.e-6)
    for p,q in zip(model.parameters(),reference.parameters()):
        assert p.grad is not None and torch.isfinite(p.grad).all()
        torch.testing.assert_close(p.grad,q.grad,rtol=5.e-5,atol=1.e-7)


def test_projected_head_is_a_single_global_categorical_matrix_with_independent_pair_rows():
    torch.manual_seed(7375)
    head = ProjectedPairIDLinear(12,45,6)
    x = torch.randn(7,12,requires_grad=True)
    expected = torch.nn.functional.linear(x,head.classes.weight @ head.projection.weight,head.classes.bias)
    torch.testing.assert_close(head(x),expected,rtol=1.e-6,atol=1.e-6)
    assert head.classes.weight.shape == (45,6)
    with pytest.raises(ValueError): ProjectedPairIDLinear(12,45,13)


@torch.no_grad()
def test_skipping_vocabulary_wide_topk_preserves_native_codes_and_rng(tmp_path):
    import sys
    from src.models.laser_pair_id_head import skip_redundant_native_topk
    recipe,aux=profile(tmp_path,projected=True),tokenizer()
    model=recipe.model(architecture()).eval()
    codes=torch.zeros(2,2,2,4,dtype=torch.long)
    labels=torch.tensor([0,999])
    module=sys.modules[type(model).__module__]
    native=module.sample_from_logits
    torch.manual_seed(7386)
    expected=model.sample(codes,aux,labels,top_k=45,top_p=.95)
    expected_rng=torch.get_rng_state()
    torch.manual_seed(7386)
    with skip_redundant_native_topk(model):
        actual=model.sample(codes,aux,labels,top_k=45,top_p=.95)
    assert module.sample_from_logits is native
    assert torch.equal(expected,actual) and torch.equal(expected_rng,torch.get_rng_state())


def test_projected_pair_model_starts_with_the_same_fresh_native_backbone(tmp_path):
    from src.training.rq_imagenet_recipes import RQRecipe
    recipe=profile(tmp_path,projected=True)
    recipe.recipe.initialization=dict(paired_fresh_backbone=True,reference_vocab_size=13)
    arch=architecture();arch.vocab_size=13
    torch.manual_seed(7387)
    reference=RQRecipe.model(recipe,arch)
    torch.manual_seed(7387)
    actual=recipe.model(arch)
    for name,value in actual.state_dict().items():
        if not name.startswith('classifier.linear.'):
            torch.testing.assert_close(value,reference.state_dict()[name],rtol=0,atol=0)


def test_full_cartesian_dense_model_is_meta_only_until_sharded_execution_exists(tmp_path):
    recipe = profile(tmp_path)
    recipe.recipe.tokenizer.num_atoms = 16384
    recipe.recipe.tokenizer.coefficient_bins = 512
    recipe.recipe.tokenizer.vocab_size = recipe.vocabulary = 8388608
    with pytest.raises(RuntimeError, match='sharded execution backend'):
        recipe.model(architecture())
    with torch.device('meta'):
        model = recipe.model(architecture())
    assert model.classifier.linear.weight.shape == (8388608,12)
    assert all(p.device.type == 'meta' for p in model.parameters())


def test_per_depth_sampling_metadata_and_native_pair_caption_are_json_serializable():
    from src.training.rq_sample_panels import sampling_caption, sampling_metadata
    s = OmegaConf.create(dict(joint_temperature=.85,joint_temperature_by_depth=[.8,.85,.9,.9]))
    assert json.loads(json.dumps(sampling_metadata(s)))['joint_temperature_by_depth'] == [.8,.85,.9,.9]
    assert 'Whole-pair T=[0.8, 0.85, 0.9, 0.9]' in sampling_caption(s)
    assert 'Pair-token' in sampling_caption(OmegaConf.create(dict(temperature=1.,top_k=0,top_p=.95)))


@pytest.mark.parametrize('row_chunk,class_chunk', [(1,1),(3,7),(128,65536)])
@pytest.mark.parametrize('has_bias', [False,True])
def test_chunked_global_linear_loss_matches_dense_loss_and_every_gradient(row_chunk,class_chunk,has_bias):
    from src.training.pair_id_linear_loss import chunked_pair_linear_cross_entropy
    torch.manual_seed(7381)
    features = torch.randn(2,2,4,6,requires_grad=True)
    weight = torch.randn(45,6,requires_grad=True)
    bias = torch.randn(45,requires_grad=True) if has_bias else None
    others = [x.detach().clone().requires_grad_() if x is not None else None
              for x in [features,weight,bias]]
    atoms = torch.randint(5,features.shape[:-1])
    ids = atoms[...,None]*9+torch.arange(9)
    p = torch.randn(*ids.shape).softmax(-1)
    p *= torch.rand(*atoms.shape,1)+.5
    logits = torch.nn.functional.linear(*others)
    dense = torch.zeros_like(logits).scatter(-1,ids,p)
    expected = -(logits.log_softmax(-1)*dense).sum(-1).mean()
    actual = chunked_pair_linear_cross_entropy(features,weight,bias,ids,p,row_chunk,class_chunk)
    torch.testing.assert_close(actual,expected,rtol=3.e-6,atol=3.e-6)
    (actual*.37).backward(); (expected*.37).backward()
    for x,y in zip([features,weight,bias],others):
        if x is not None:
            torch.testing.assert_close(x.grad,y.grad,rtol=5.e-5,atol=1.e-6)


@pytest.mark.parametrize('mode',['hard','soft_coefficient'])
def test_chunked_training_on_native_rq_has_same_loss_and_gradients_and_dense_eval(tmp_path,mode):
    from src.training.rq_imagenet_shared import training_backward
    torch.manual_seed(7382)
    recipe,aux = profile(tmp_path,projected=True,mode=mode),tokenizer()
    recipe.shape = (2,2,4)
    recipe.recipe.execution.update(chunked_training_loss=True,loss_row_chunk_size=3,loss_class_chunk_size=7)
    model = recipe.model(architecture()).train()
    reference = copy.deepcopy(model)
    reference.classifier.linear.chunked_training_loss = False
    images,labels = torch.randn(2,7,2,2),torch.tensor([0,999])
    targets,codes = recipe.targets(aux,images)
    recipe.targets = lambda aux,images:(targets,codes)
    actual = training_backward(recipe,model,aux,images,labels,
        torch.amp.GradScaler('cuda',enabled=False),.37,amp=False)
    logits = reference(codes,aux,labels)
    if mode == 'soft_coefficient':
        ids = targets['atoms'][...,None]*9+torch.arange(9)
        dense = torch.zeros_like(logits).scatter(-1,ids,targets['coefficient_probabilities'])
        expected = reference.compute_loss(logits,dense,use_soft_target=True)
    else:
        expected = reference.compute_loss(logits,codes,use_soft_target=False)
    (expected*.37).backward()
    torch.testing.assert_close(actual,expected,rtol=3.e-6,atol=3.e-6)
    for p,q in zip(model.parameters(),reference.parameters()):
        assert p.grad is not None and torch.isfinite(p.grad).all()
        torch.testing.assert_close(p.grad,q.grad,rtol=7.e-5,atol=2.e-7)
    model.eval(); reference.eval()
    with torch.no_grad():
        assert model(codes,aux,labels).shape[-1] == 45
        torch.testing.assert_close(model(codes,aux,labels),reference(codes,aux,labels),rtol=0,atol=0)


def _distributed_pair_id_worker(rank,world,directory,init_file):
    import torch.distributed as dist
    from contextlib import nullcontext
    from torch.nn.parallel import DistributedDataParallel as DDP
    torch.set_num_threads(1)
    dist.init_process_group('gloo',rank=rank,world_size=world,init_method='file://'+init_file)
    torch.manual_seed(7383)
    rank_dir=Path(directory)/str(rank);rank_dir.mkdir()
    recipe,aux=profile(rank_dir,projected=True),tokenizer()
    recipe.recipe.execution.update(chunked_training_loss=True,loss_row_chunk_size=3,loss_class_chunk_size=7)
    model=recipe.model(architecture()).train()
    ddp=DDP(model)
    batches=[]
    for micro in range(2):
        torch.manual_seed(7384+rank*10+micro)
        images,labels=torch.randn(2,7,2,2),torch.tensor([rank,999-rank])
        targets,codes=recipe.targets(aux,images)
        batches.append((targets,codes,labels))
        with ddp.no_sync() if micro==0 else nullcontext():
            loss=recipe.loss(model,ddp(codes,aux,labels),targets)
            (loss*.5).backward()
    gathered=[None]*world
    dist.all_gather_object(gathered,batches)
    reference=copy.deepcopy(model)
    reference.zero_grad(set_to_none=True)
    reference.classifier.linear.chunked_training_loss=False
    all_batches=[batch for rank_batches in gathered for batch in rank_batches]
    codes=torch.cat([b[1] for b in all_batches]);labels=torch.cat([b[2] for b in all_batches])
    atoms=torch.cat([b[0]['atoms'] for b in all_batches])
    p=torch.cat([b[0]['coefficient_probabilities'] for b in all_batches])
    logits=reference(codes,aux,labels)
    ids=atoms[...,None]*9+torch.arange(9)
    dense=torch.zeros_like(logits).scatter(-1,ids,p)
    reference.compute_loss(logits,dense,use_soft_target=True).backward()
    for a,b in zip(model.parameters(),reference.parameters()):
        torch.testing.assert_close(a.grad,b.grad,rtol=1.e-4,atol=3.e-7)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0005,betas=(.9,.95),weight_decay=.0001)
    optimizer.step()
    assert all(int(state['step'])==1 for state in optimizer.state.values())
    assert len(optimizer.state)==len(list(model.parameters()))
    (rank_dir/'ddp-verification.json').write_text(json.dumps(dict(passed=True,
        accumulated_microbatches=2,global_batch=8,all_adam_steps=1)))
    dist.destroy_process_group()


def test_distributed_chunked_head_gradients_and_accumulation_match_dense_global_batch(tmp_path):
    torch.multiprocessing.spawn(_distributed_pair_id_worker,
        args=(2,str(tmp_path),str(tmp_path/'gloo-init')),nprocs=2,join=True)
    assert all(json.loads((tmp_path/str(rank)/'ddp-verification.json').read_text())['passed']
               for rank in range(2))


@pytest.mark.skipif(os.environ.get('LASER_CUDA_TESTS')!='1',reason='Explicit isolated GPU verification')
def test_gpu_amp_master_weight_gradients_match_dense_native_global_loss(tmp_path):
    torch.set_num_threads(4);torch.cuda.set_device(0)
    torch.manual_seed(7385)
    recipe,aux=profile(tmp_path,projected=True),tokenizer().cuda()
    recipe.recipe.execution.update(chunked_training_loss=True,loss_row_chunk_size=3,loss_class_chunk_size=7)
    model=recipe.model(architecture()).cuda().train()
    reference=copy.deepcopy(model)
    reference.classifier.linear.chunked_training_loss=False
    images,labels=torch.randn(2,7,2,2,device='cuda'),torch.tensor([0,999],device='cuda')
    targets,codes=recipe.targets(aux,images)
    logits=model(codes,aux,labels,amp=True)
    actual=recipe.loss(model,logits,targets)
    other=reference(codes,aux,labels,amp=True)
    ids=targets['atoms'][...,None]*9+torch.arange(9,device='cuda')
    dense=torch.zeros_like(other,dtype=torch.float32).scatter(-1,ids,targets['coefficient_probabilities'])
    expected=reference.compute_loss(other.float(),dense,use_soft_target=True)
    (actual*128).backward();(expected*128).backward()
    torch.testing.assert_close(actual,expected,rtol=3.e-4,atol=3.e-4)
    for a,b in zip(model.parameters(),reference.parameters()):
        assert a.dtype==torch.float32 and a.grad is not None and torch.isfinite(a.grad).all()
        torch.testing.assert_close(a.grad/128,b.grad/128,rtol=2.e-2,atol=3.e-4)
