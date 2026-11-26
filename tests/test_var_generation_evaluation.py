from types import SimpleNamespace
import torch
from torch import nn
from src.training import var_generation_evaluation as evaluation


def test_fid_and_is_share_images_preserve_rng_and_training_mode(tmp_path,monkeypatch):
    class Model(nn.Module):
        num_classes=2
        def sample(self,labels,**kwargs):
            torch.rand(3)  # Exercise the RNG guard.
            return labels[:,None,None,None].float()
    class Metric:
        seen=[]
        def __init__(self,device,compute_inception_score,**kwargs):
            assert compute_inception_score
            self.fake_count=torch.tensor(0)
        def update(self,images,real):
            assert not real
            self.fake_count+=len(images);self.seen.append(images.clone())
        def compute(self,shuffle_for_inception,shuffle_seed):
            assert shuffle_for_inception
            torch.randperm(int(self.fake_count))
            return 12.,23.,.4
    monkeypatch.setattr(evaluation,'PairedGenerationMetrics',Metric)
    events=[];model=Model();model.train()
    experiment=SimpleNamespace(cfg=dict(fid_samples=20,generation_batch_size=7,fid_reference='unused'),
        model=model,source=SimpleNamespace(decoder=nn.Identity(),post_quant_conv=nn.Identity()),
        rank=0,world=1,device=torch.device('cpu'),inception=nn.Identity(),
        progress=dict(epoch=5,step=8340),out=tmp_path,log=lambda phase,**kw:events.append((phase,kw)))
    state=torch.get_rng_state().clone();result=evaluation.evaluate_generation(experiment)
    assert torch.equal(state,torch.get_rng_state()) and model.training
    assert result['count']==20 and result['inception_score']==23. and result['inception_score_std']==.4
    assert result['validation_fid50k']==12. and result['inception_score_splits']==10
    assert sum(len(batch) for batch in Metric.seen)==20
    assert events[0][0]=='generation' and (tmp_path/'fid-epoch005.json').exists()


def test_numpy_is_handles_zeros_and_is_reproducible():
    import numpy as np
    uniform=np.full((100,4),.25)
    mean,std=evaluation.numpy_inception_score(uniform,seed=7)
    assert mean==1. and std==0.
    certain=np.repeat(np.eye(4),100,axis=0)
    first=evaluation.numpy_inception_score(certain,seed=7)
    assert first==evaluation.numpy_inception_score(certain,seed=7)
    assert 3.8<first[0]<=4. and first[1]>=0.


def test_paired_cpu_aggregation_matches_fid_and_keeps_probabilities(tmp_path):
    import numpy as np
    class Tiny(evaluation.PairedGenerationMetrics):
        feature_dim=2
        num_logits=4
    rng=np.random.default_rng(3);features=rng.normal(size=(100,2))
    reference=tmp_path/'reference.npz';np.savez(reference,mu=features.mean(0),sigma=np.cov(features,rowvar=False))
    metric=Tiny(torch.device('cpu'),compute_inception_score=True,reference_stats_path=reference,
                inception=nn.Identity(),statistics_directory=tmp_path/'stats')
    metric.fake_sum.copy_(torch.from_numpy(features.sum(0)))
    metric.fake_cross.copy_(torch.from_numpy(features.T@features));metric.fake_count.fill_(100)
    metric.fake_probabilities=[torch.full((100,4),.25)]
    fid,mean,std=metric.compute(shuffle_seed=7)
    assert abs(fid)<1e-8 and mean==1. and std==0.
    assert np.load(tmp_path/'stats/probabilities-rank0.npy').shape==(100,4)
