import copy

import pytest
import torch
from torchmetrics.image.fid import FrechetInceptionDistance

from src.training.fid_reference import (
    SCHEMA, load_torchmetrics_reference, seed_torchmetrics_reference,
    compute_reference_fids, fixed_evaluation_rng, fid_log_values,
)


class Features(torch.nn.Module):
    num_features = 3

    def forward(self, images):
        return images.mean(dim=(-2, -1))


def reference(tmp_path):
    real = torch.rand(37, 3, 4, 4, generator=torch.Generator().manual_seed(19))
    metric = FrechetInceptionDistance(feature=Features())
    metric.update(real, real=True)
    payload = dict(schema=SCHEMA, feature_dim=3, metadata={'samples':37},
        **{key:getattr(metric,key) for key in ('real_features_sum',
           'real_features_cov_sum','real_features_num_samples')})
    path = tmp_path/'real.pt'
    torch.save(payload,path)
    return path,real,payload


def test_cached_entire_real_corpus_matches_direct_fid_with_fewer_fake_images(tmp_path):
    path,real,_ = reference(tmp_path)
    fake = torch.rand(11,3,4,4,generator=torch.Generator().manual_seed(29))*.7
    direct = FrechetInceptionDistance(feature=Features())
    direct.update(real,real=True)
    direct.update(fake,real=False)
    cached = FrechetInceptionDistance(feature=Features())
    seed_torchmetrics_reference(cached,path)
    cached.update(fake,real=False)
    assert int(cached.real_features_num_samples)==37
    assert int(cached.fake_features_num_samples)==11
    torch.testing.assert_close(cached.compute(),direct.compute(),rtol=0,atol=0)


def test_reference_is_not_multiplied_by_world_size(tmp_path):
    path,_,payload = reference(tmp_path)
    ranks=[FrechetInceptionDistance(feature=Features()) for _ in range(6)]
    for rank,metric in enumerate(ranks):
        seed_torchmetrics_reference(metric,path,rank=rank)
    for key in ('real_features_sum','real_features_cov_sum','real_features_num_samples'):
        torch.testing.assert_close(sum(getattr(metric,key) for metric in ranks),payload[key],rtol=0,atol=0)
    shards=[list(range(rank,1281167,6)) for rank in range(6)]
    assert sum(map(len,shards))==1281167
    assert sorted(index for shard in shards for index in shard)==list(range(1281167))


@pytest.mark.parametrize('failure',['schema','dimensions','dtype','count','nonfinite','incomplete'])
def test_invalid_reference_is_rejected(tmp_path,failure):
    path,_,payload = reference(tmp_path)
    payload=copy.deepcopy(payload)
    if failure=='schema':payload['schema']='other-network'
    if failure=='dimensions':payload['real_features_sum']=torch.zeros(4,dtype=torch.float64)
    if failure=='dtype':payload['real_features_sum']=payload['real_features_sum'].float()
    if failure=='count':payload['metadata']['samples']=38
    if failure=='nonfinite':payload['real_features_cov_sum'][0,0]=float('nan')
    torch.save(payload,path)
    with pytest.raises(ValueError):
        load_torchmetrics_reference(path,feature_dim=3,expected_samples=1281167 if failure=='incomplete' else 37)


def test_distinct_real_references_reuse_the_exact_same_fake_moments(tmp_path):
    training_path, real, payload = reference(tmp_path)
    validation = real * .5 + .3
    val_metric = FrechetInceptionDistance(feature=Features())
    val_metric.update(validation, real=True)
    val_path = tmp_path / 'validation.pt'
    torch.save({**payload, 'metadata': {'samples': 37, 'real_split': 'val'},
                **{name: getattr(val_metric, name) for name in
                   ('real_features_sum', 'real_features_cov_sum', 'real_features_num_samples')}}, val_path)
    fake = torch.rand(11, 3, 4, 4, generator=torch.Generator().manual_seed(29)) * .7
    metric = FrechetInceptionDistance(feature=Features())
    metric.update(fake, real=False)
    before = metric.fake_features_cov_sum.clone()
    scores = compute_reference_fids(metric, {'train_full': training_path, 'val_full': val_path},
                                    expected_generated_samples=11)
    for name, images in [('train_full', real), ('val_full', validation)]:
        direct = FrechetInceptionDistance(feature=Features())
        direct.update(images, real=True)
        direct.update(fake, real=False)
        assert scores[name]['fid'] == float(direct.compute())
        assert scores[name]['real_images'] == 37
        assert scores[name]['generated_images'] == 11
    assert scores['train_full']['fid'] != scores['val_full']['fid']
    torch.testing.assert_close(metric.fake_features_cov_sum, before, rtol=0, atol=0)
    values = fid_log_values(scores, dataset='imagenet')
    assert set(values) == {'eval/fid_imagenet_train_full', 'eval/fid_imagenet_val_full'}
    with pytest.raises(ValueError, match='generated sample count'):
        compute_reference_fids(metric, {'train_full': training_path}, expected_generated_samples=12)


@pytest.mark.parametrize('fail', [False, True])
def test_fixed_evaluation_rng_is_repeatable_and_preserves_training_stream_on_error(fail):
    state = torch.get_rng_state().clone()
    samples = []
    for _ in range(2):
        try:
            with fixed_evaluation_rng(261056, 'cpu', rank=3):
                samples.append(torch.rand(20))
                if fail:
                    raise RuntimeError('interrupted evaluation')
        except RuntimeError:
            pass
        assert torch.equal(torch.get_rng_state(), state)
    assert torch.equal(samples[0], samples[1])


@pytest.mark.parametrize('enabled', [False, True])
def test_resume_evaluation_flag_accepts_yaml_boolean_options(enabled):
    from src.training.options import options_to_argv
    from src.training.rqtransformer import build_parser
    args = build_parser().parse_args(options_to_argv({
        'checkpoint': 'tokenizer.pt', 'data': 'imagenet', 'output': 'output',
        'fid_on_resume': enabled, 'fid_seed': 261056,
    }))
    assert args.fid_on_resume is enabled
    assert args.fid_seed == 261056
