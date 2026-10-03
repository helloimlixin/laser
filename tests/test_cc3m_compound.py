import torch
from src.training import rqtransformer as rq
from src.training.cc3m_compound import evaluation_indices
from scripts.tools.build_cc3m_compound_cache import text_tokenizer
from tests.test_compound_pair_autoregressive import tiny_config, tiny_aux


def test_production_recipe_composes_at_global_scope():
    from src.training.cli import load_config
    config=load_config(rq.ROOT/'configs/stage2/cc3m-rfid421-compound-650m-6h100.yaml')
    assert config.stage=='stage2' and config.backend=='cc3m_compound'
    assert config.options.batch_size*config.options.accumulation*6==config.options.total_batch_size
    assert config.options.latent_shape==[8,8,4]


def test_text_conditioned_pair_causality_and_cached_generation():
    torch.manual_seed(79)
    config = tiny_config(4)
    config.block_size = [2,2,4]
    config.block_size_cond = 3
    config.vocab_size_cond = 13
    model = rq.CompoundLaserRQTransformer(config,7,5,micro_transformer_layers=2,
        depth_specific_coeff_heads=True,pair_autoregressive=True,mask_seen_atoms_training=False).eval()
    aux = tiny_aux(4)
    packed = torch.arange(16).reshape(1,2,2,4).remainder(7)*5+2
    text = torch.tensor([[1,2,3]])
    expected, text_logits = model(packed,model_aux=aux,cond=text)
    other, _ = model(packed,model_aux=aux,cond=torch.tensor([[4,2,3]]))
    for key in expected:
        assert not torch.equal(other[key],expected[key])
    changed = packed.clone(); changed[0,0,0,0] += 1
    modified, _ = model(changed,model_aux=aux,cond=text)
    for key in expected:
        torch.testing.assert_close(modified[key][:,0,0,0],expected[key][:,0,0,0],rtol=0,atol=0)
        assert not torch.equal(modified[key][:,0,0,1],expected[key][:,0,0,1])
        assert not torch.equal(modified[key][:,1,1,3],expected[key][:,1,1,3])
    # Change every future target; cached logits must match teacher forcing.
    generated = torch.zeros_like(packed)
    model.init_cache()
    for h in range(2):
        for w in range(2):
            for d in range(4):
                hidden = model.cached_head_output(generated,aux,text,(h,w,d),amp=False)
                atom_logits = model.classifier(hidden)
                atom = packed[:,h,w,d]//5
                coeff_logits = model.coefficient_logits(hidden,aux.dictionary.t()[atom],depth_index=d)
                torch.testing.assert_close(atom_logits,expected['atom_logits'][:,h,w,d],atol=2e-6,rtol=2e-5)
                torch.testing.assert_close(coeff_logits,expected['coeff_logits'][:,h,w,d],atol=2e-6,rtol=2e-5)
                generated[:,h,w,d] = packed[:,h,w,d]
    model.init_cache()
    # Text LM can see only earlier text, and has trainable gradients.
    assert text_logits.shape == (1,2,13)
    loss = torch.nn.functional.cross_entropy(text_logits.reshape(-1,13),text[:,1:].reshape(-1))
    loss.backward()
    assert model.cond_classifier.linear.weight.grad.abs().sum()>0


def test_validation_partition_has_no_repeats_or_dropped_captions():
    parts = [evaluation_indices(13443,r,6) for r in range(6)]
    assert sorted(sum(parts,[])) == list(range(13443))


def test_official_text_tokenizer_and_length():
    tokenizer = text_tokenizer()
    result = tokenizer.encode_batch(['a red car', 'a dog on a beach'])
    assert tokenizer.get_vocab_size()==16384
    assert all(len(x.ids)==32 for x in result)
    assert result[0].ids != result[1].ids


def test_best_clip_upload_ranks_highest_and_replaces(tmp_path):
    class Run:
        def __init__(self): self.saved=[]
        def save(self,path,**kwargs): self.saved.append(path)
    paths=[tmp_path/name for name in ['last.pt','low.pt','high.pt']]
    for path in paths: path.write_bytes(path.name.encode())
    wb=Run()
    rq.upload_selected_checkpoint_files(wb,last_checkpoint=paths[0],best_fid=[],
        best_clip=[(.2,paths[1]),(.3,paths[2])],upload_dir=tmp_path/'uploads')
    assert (tmp_path/'uploads/best-clip-01.pt').read_bytes()==b'high.pt'
    rq.upload_selected_checkpoint_files(wb,last_checkpoint=paths[0],best_fid=[],
        best_clip=[(.4,paths[1])],upload_dir=tmp_path/'uploads')
    assert (tmp_path/'uploads/best-clip-01.pt').read_bytes()==b'low.pt'
    assert not (tmp_path/'uploads/best-clip-02.pt').exists()


def test_cache_merge_preserves_physical_pairs_and_caption_alignment(tmp_path):
    import json
    from scripts.tools.run_cc3m_compound import merge_cache
    from scripts.tools.build_cc3m_compound_cache import STAGE1_SHA
    from src.training.cc3m_compound import load_cache
    base=tmp_path/'run';(base/'cache').mkdir(parents=True)
    for split,index,value in [('train',0,3.),('train',1,6.),('validation',0,9.)]:
        path=base/'cache'/f'cc3m-{split}-{index:04d}.pt'
        atoms=torch.arange(4,dtype=torch.int16).reshape(1,1,1,4).expand(1,8,8,4).contiguous()
        coeffs=torch.full((1,8,8,4),value)
        meta=dict(stage1_sha256=STAGE1_SHA,clip_coefficients=False,items=1)
        data=dict(atoms=atoms,coeffs=coeffs,captions=[f'{split}-{index}'],text_ids=torch.full((1,32),index,dtype=torch.int16),meta=meta)
        torch.save(data,path);path.with_suffix('.json').write_text(json.dumps(meta))
    options=dict(train_items=2,validation_items=1,token_cache=str(tmp_path/'local/train.pt'),
        validation_cache=str(tmp_path/'local/validation.pt'),dataset_revision='test')
    merge_cache(base,options,expected_shards={'train':2,'validation':1})
    training=load_cache(options['token_cache']);validation=load_cache(options['validation_cache'])
    assert training['captions']==['train-0','train-1']
    assert training['meta']['coeff_scales']==[2.]*4
    assert training['coeffs'][0].unique().tolist()==[1.5]
    assert training['coeffs'][1].unique().tolist()==[3.]
    assert validation['coeffs'].unique().tolist()==[4.5]  # Validation is never clipped.
