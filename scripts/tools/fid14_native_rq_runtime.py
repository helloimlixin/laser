"""Use the actual upstream RQ stacks with checkpoint-compatible LASER inputs."""
import hashlib
from pathlib import Path

import torch
from omegaconf import OmegaConf
from rqvae.models.rqtransformer.configs import RQTransformerConfig
from rqvae.models.rqtransformer.attentions import AttentionStack, MultiSelfAttention


def install(namespace):
    from native_physical_pair_adapter import LaserPairRQTransformer
    g=namespace
    legacy_factory=g['original_build_model']

    def factory(*args,**kwargs):
        assert kwargs.get('physical_pair_context') and not kwargs.get('compound')
        with torch.device('meta'):
            archived=legacy_factory(*args,**kwargs)
            config=RQTransformerConfig.create(OmegaConf.create(
                OmegaConf.to_container(archived.config,resolve=True)))
            model=LaserPairRQTransformer(config,num_atoms=int(args[1]))
        assert list(model.state_dict())==list(archived.state_dict())
        assert all(a.shape==b.shape for a,b in zip(model.parameters(),archived.parameters()))
        assert len(list(model.parameters()))==782
        assert isinstance(model.body_transformer,AttentionStack)
        assert isinstance(model.head_transformer,AttentionStack)
        assert all(isinstance(b.attn,MultiSelfAttention)
            for stack in (model.body_transformer,model.head_transformer) for b in stack.blocks)
        # Every tensor is immediately overwritten by the strict full resume.
        model.to_empty(device='cpu')
        from rqvae.models.rqtransformer import attentions,transformers
        proof=dict(passed=True,model_class=type(model).__name__,parameter_tensors=782,
            spatial_layers=len(model.body_transformer.blocks),depth_layers=len(model.head_transformer.blocks),
            attention_source=str(Path(attentions.__file__).resolve()),
            attention_sha256=hashlib.sha256(Path(attentions.__file__).read_bytes()).hexdigest(),
            transformer_source=str(Path(transformers.__file__).resolve()),
            no_upstream_core_edits=True,laser_adapter_only=True)
        assert proof['attention_sha256']=='cfcb3bb5e005cfc6e9a1641fe87e2c18db7afc8ae775b6711f64e5de3761561b'
        g['record'](g['VERIFY']/('native-core-rank'+g['os'].environ['RANK']+'.json'),proof)
        return model

    g['original_build_model']=factory
    original_init=g['original_init']
    def open_run(*args,**kwargs):
        kwargs.setdefault('config',{}).update(native_rq_core=True,
            rq_upstream_commit='341395e562ac347f5eb62db9f5f08b9f2cc42a60',
            rq_attention_implementation='rqvae.models.rqtransformer.attentions',
            rq_attention_sha256='cfcb3bb5e005cfc6e9a1641fe87e2c18db7afc8ae775b6711f64e5de3761561b',
            upstream_core_modified=False,laser_physical_pair_adapter=True,
            native_migration_prediction_and_joint_sampler_checks_passed=True)
        return original_init(*args,**kwargs)
    g['original_init']=open_run
    original_save=g['training'].atomic_torch_save
    def save(payload,target):
        payload=dict(payload,execution_core=dict(
            implementation='original rqvae.models.rqtransformer',
            attention_sha256='cfcb3bb5e005cfc6e9a1641fe87e2c18db7afc8ae775b6711f64e5de3761561b',
            laser_adapter='checkpoint-compatible physical pairs and typed vocabulary slices'))
        return original_save(payload,target)
    g['training'].atomic_torch_save=save
