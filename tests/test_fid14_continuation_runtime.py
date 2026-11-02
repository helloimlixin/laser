from types import ModuleType, SimpleNamespace
import sys

import torch

from scripts.tools.fid14_continuation_runtime import install


def test_fulltrain_fid_and_historical_lr_use_same_fakes_and_distinct_references(monkeypatch, tmp_path):
    """A changed main reference must not feed an incomparable FID to saved LR."""
    from src import rqvae_metrics as metrics
    tar=ModuleType('imagenet_tar');tar.TarImageNet=type('TarImageNet',(),{})
    monkeypatch.setitem(sys.modules,'imagenet_tar',tar)
    logs=[];observations=[];recorded=[];metric_instances=[]

    class Run:
        id='test'
        url='test'
        def define_metric(self,*args,**kwargs):pass
        def log(self,value,*args,**kwargs):logs.append(value)

    class Metric:
        def __init__(self,device,**kwargs):
            assert kwargs['reference_stats_path']==tmp_path/'inputs/imagenet_256_train.npz'
            self.fake_count=torch.tensor(0)
            self.fake_sum=torch.zeros(2);self.fake_cross=torch.zeros(2,2)
            metric_instances.append(self)
        def update(self,images,*,real):
            assert not real and len(images)<=4
            self.fake_count+=len(images)
        def compute(self,**kwargs):
            assert int(self.fake_count)==50000
            return 11.25,77.5,.4

    monkeypatch.setattr(metrics,'DistributedOriginalRQVAEMetrics',Metric)
    monkeypatch.setattr(metrics,'_mean_covariance',lambda *args:(torch.zeros(2),torch.eye(2)))
    def reference(path):
        assert path==tmp_path/'inputs/imagenet_256_val.npz'
        return torch.ones(2),torch.eye(2)
    monkeypatch.setattr(metrics,'load_reference_statistics',reference)
    monkeypatch.setattr(metrics,'frechet_distance',lambda *args:14.9)
    monkeypatch.setattr(torch.distributed,'get_rank',lambda:0)
    monkeypatch.setattr(torch.distributed,'get_world_size',lambda:1)
    monkeypatch.setattr(torch.distributed,'broadcast',lambda *args,**kwargs:None)
    actual_fork=torch.random.fork_rng
    monkeypatch.setattr(torch.random,'fork_rng',lambda **kwargs:actual_fork(devices=[]))
    monkeypatch.setattr(torch.cuda,'manual_seed',lambda *args:None)

    def observe(fid):
        observations.append(fid)
        return dict(reductions=0,lr_after=3.e-6)
    training=SimpleNamespace()
    state=dict(training=training,BASE=tmp_path,EVIDENCE=tmp_path,
        original_loader=lambda *args,**kwargs:None,original_init=lambda *args,**kwargs:Run(),
        ARGS=SimpleNamespace(wandb_mode='disabled'),record=lambda path,value:recorded.append((path,value)),
        INITIAL_RECOVERY=dict(global_step=48747),UPDATES=10,
        observe_official_fid=observe,VERIFY=tmp_path/'verification')
    install(state)
    run=state['original_init'](config={});state['WB']=run
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.parameter=torch.nn.Parameter(torch.zeros(1));self.samples=0
        def sample_sparse(self,count,aux,cond,amp):
            self.samples+=count
            return torch.zeros(count,1,1,8,dtype=torch.long)
    model=Model()
    aux=SimpleNamespace(decode_tokens=lambda tokens:torch.zeros(len(tokens),3,1,1))
    result=training.evaluate_generation_metrics(model,aux,None,50000,128,metric_backend='original-rqvae')
    assert result==(11.25,77.5,.4) and observations==[14.9]
    assert model.samples==50000 and len(metric_instances)==1 and model.training
    run.log({'val/fid':result[0],'val/fid_rqvae_imagenet_train':result[0],
        'val/inception_score':result[1],'val/inception_score_std':result[2],'train/global_step':48757})
    assert logs[-1]['eval/fid']==11.25
    assert not any(key.startswith('val/fid') for key in logs[-1])
    assert logs[0]['diagnostics/fid_validation']==14.9
    assert state['OFFICIAL_METRICS']['real_images']==1281167
    assert state['OFFICIAL_METRICS']['generated_images']==50000
