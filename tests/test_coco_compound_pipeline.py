from types import SimpleNamespace
import torch
import pytest

from scripts.tools.build_coco_compound_cache import caption_cache
from src.training.cc3m_compound import load_cache
from src.training.cli import load_config
from src.training.paths import ROOT


def test_coco_cache_expands_captions_without_mixing_images_or_clipping(tmp_path):
    dataset = SimpleNamespace(samples=[(0,'a red car'),(0,'a parked car'),(1,'a dog')],
        images=[dict(id=123),dict(id=987)], split='train',caption_mode='all')
    atoms = torch.stack([torch.zeros(8,8,4,dtype=torch.int16),torch.ones(8,8,4,dtype=torch.int16)])
    coeffs = torch.stack([torch.full((8,8,4),6.),torch.full((8,8,4),12.)])
    cache = caption_cache(dataset,atoms,coeffs,torch.tensor([1.,2.,3.,4.]),'coco-test')
    assert cache['image_ids'].tolist() == [123,123,987]
    assert cache['captions'] == ['a red car','a parked car','a dog']
    torch.testing.assert_close(cache['coeffs'][0],cache['coeffs'][1])
    assert cache['coeffs'][2,0,0].tolist() == [12.,6.,4.,3.]
    path=tmp_path/'cache.pt';torch.save(cache,path)
    assert len(load_cache(path,'coco-test')['atoms']) == 3
    with pytest.raises(ValueError,match='stage-1'):
        load_cache(path,'unrelated-tokenizer')


def test_coco_launch_retains_ffhq_objective_and_caption_split():
    s1=load_config(ROOT/'configs/stage1/coco2014-laser-k4-4a100.yaml')
    s2=load_config(ROOT/'configs/stage2/coco2014-ffhq-compound-k4-4a100.yaml').options
    assert s1.model.sparsity_level == s2.sparsity_level == 4
    assert s1.model.force_quant_conv and s1.checkpoint.monitor == 'val/rfid'
    assert s1.data.batch_size*4 == 128
    assert s2.batch_size*s2.accumulation*4 == 128
    assert s2.train_items == 414113 and s2.validation_items == 40504
    assert s2.model_preset == 'ffhq-350m-text' and s2.epochs == 200
    assert s2.coeff_vocab_size == 2048 and s2.coeff_target_temperature == .5
    assert s2.atom_loss_weight == 1.5 and s2.geometry_loss_weight == .05


def test_latest_step_callback_publishes_first_then_scheduled_steps():
    from src.training.common import _make_selected_checkpoint_file_callback
    Callback=_make_selected_checkpoint_file_callback(object)
    callback=Callback(SimpleNamespace(save_last=False),upload_dir='/tmp/unused',every_n_train_steps=200)
    uploads=[]
    callback._upload=lambda trainer:uploads.append(trainer.global_step)
    trainer=SimpleNamespace(global_step=0)
    for step in (0,2,2,4,198,200,202,400):
        trainer.global_step=step
        callback.on_train_batch_end(trainer,None,None,None,0)
    assert uploads == [2,200,400]
