#!/usr/bin/env python3
"""Require strict checkpoint reload and production-size sampling before launch."""
import json
import os
from pathlib import Path
import sys

BASE=Path(os.environ.get('LASER_RQ_BASE','/tmp/laser-rq-minimal-imagenet-20261014'))
os.environ['RQ_UPSTREAM_ROOT']=str(BASE/'upstream')
sys.path[:0]=[str(BASE/'source'),str(BASE/'runtime'),str(BASE/'upstream')]
import torch
from omegaconf import OmegaConf
from src.training import laser_rq_minimal as laser
from rqvae.utils.config import load_config,augment_arch_defaults,augment_optimizer_defaults
from rqvae.optimizer.optimizer import create_resnet_optimizer
from rqvae.optimizer.scheduler import create_scheduler
from src.training.rq_imagenet_runtime import write_json
from src.training import k4_checkpoint_io as checkpoint_io


def verify_hard_encoding(profile, aux, config, device, out):
    """Check real-image codes against the independent stage-1 implementation."""
    import time
    from src.training import laser_hard_ar as hard
    from src.training.rq_imagenet_runtime import TarImageNet
    from rqvae.img_datasets.transforms import create_transforms
    sys.path.insert(0,str(BASE/'validation'))
    from stage1_reference.dictionary_learner import DictionaryLearning
    for name in ('src.stochastic_compound','src.training.stochastic_image_pairs',
                 'src.training.stochastic_omp_graph','src.training.omp_joint_targets'):
        assert name not in sys.modules, name
    data=TarImageNet(BASE/'data','train',create_transforms(config.dataset,split='train'),seed=261001)
    images=torch.stack([data[index][0] for index in range(16)]).to(device)
    before_rng=torch.cuda.get_rng_state(device).clone()
    before_cpu_rng=torch.get_rng_state().clone()
    tokens,_=profile.targets(aux,images)
    hard.verify_tokens(aux,tokens)
    torch.testing.assert_close(torch.cuda.get_rng_state(device),before_rng,rtol=0,atol=0)
    torch.testing.assert_close(torch.get_rng_state(),before_cpu_rng,rtol=0,atol=0)
    replay,_=profile.targets(aux,images)
    torch.testing.assert_close(replay,tokens,rtol=0,atol=0)
    latent=aux.quant_conv(aux.encoder(images)).permute(0,2,3,1).float()
    atoms,values=hard.sparse_encoding(latent,aux.dictionary,aux.sparsity_level,gram=aux._hard_ar_gram)
    reference=DictionaryLearning(num_embeddings=aux.num_atoms,embedding_dim=aux.dictionary.shape[0],
                                sparsity_level=aux.sparsity_level).to(device)
    expected_atoms,expected_values=reference.batch_omp_with_support(
        latent.reshape(-1,latent.shape[-1]).T,aux.dictionary)
    expected_atoms=expected_atoms.reshape_as(atoms);expected_values=expected_values.reshape_as(values)
    torch.testing.assert_close(atoms,expected_atoms,rtol=0,atol=0)
    torch.testing.assert_close(values,expected_values,rtol=1e-5,atol=1e-5)
    factors=profile.recipe.get('ar_factors',{})
    coordinates=factors.get('coordinates','physical');bins=int(factors.get('coefficient_bins',2048))
    expected_target=expected_values
    if coordinates=='orthogonal':
        from src.orthogonal_sparse_codec import dictionary_to_orthogonal_coefficients
        expected_target,_=dictionary_to_orthogonal_coefficients(aux.dictionary.T[atoms],expected_values)
    grid=aux.coeff_bins if bins==2048 else torch.linspace(-3,3,bins,device=device)
    group=2048//bins
    expected_ids=hard.nearest_coefficient_ids(expected_target/aux.coeff_scales,grid)*group+group//2
    torch.testing.assert_close(tokens[...,0::2],expected_atoms,rtol=0,atol=0)
    torch.testing.assert_close(tokens[...,1::2]-aux.num_atoms,expected_ids,rtol=0,atol=0)
    del reference
    timings=[]
    for _ in range(5):
        torch.cuda.synchronize(device);start=time.monotonic()
        profile.targets(aux,images)
        torch.cuda.synchronize(device);timings.append((time.monotonic()-start)*1000)
    normalized=expected_target/aux.coeff_scales
    write_json(out/'hard-token-gpu-verification.json',dict(passed=True,real_images=len(images),
        deterministic=True,rng_unchanged=True,stage1_reference_support_exact=True,coefficient_coordinates=coordinates,
        stage1_reference_coefficient_bins_exact=True,
        max_physical_coefficient_error=float((values-expected_values).abs().max()),
        endpoint_clipped_fraction=float(((normalized<aux.coeff_bins[0])|(normalized>aux.coeff_bins[-1])).float().mean()),
        hard_encoding_ms_per_16_images=timings,sampled_trajectories=0,teacher_modules_imported=False))


def main():
    torch.set_num_threads(4)
    recipe=OmegaConf.load(BASE/'inputs/recipe.yaml')
    out=Path('/workspace/Projects/laser/outputs')/recipe.run_id
    verification=json.loads((out/'smoke/verification.json').read_text())
    assert verification['passed'] and verification['global_step']==2
    checkpoint=checkpoint_io._checkpoint_upload_source((out/'smoke/checkpoints/last.pt').resolve())
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    assert saved['recipe']==OmegaConf.to_container(recipe,resolve=True)
    assert saved['global_step']==2 and saved['runtime']['fresh_stage2'] and saved['runtime']['fresh_optimizer']
    config=load_config(BASE/'inputs/stage2-training-config.yaml')
    arch=augment_arch_defaults(config.arch);arch.vocab_size=config.dataset.vocab_size
    with torch.device('meta'):
        model=laser.build_model(arch,num_atoms=recipe.tokenizer.num_atoms,
            coefficient_bins=recipe.tokenizer.coefficient_bins,depth=recipe.tokenizer.depth)
    model.load_state_dict(saved['state_dict'],strict=True,assign=True)
    optimizer=create_resnet_optimizer(model,augment_optimizer_defaults(config.optimizer))
    scheduler=create_scheduler(optimizer,config.optimizer.warmup,626,int(recipe.lr_schedule_epochs))
    optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
    assert scheduler.after_scheduler.last_epoch==2 and scheduler.after_scheduler.T_max==62600
    assert len(optimizer.state)==len(list(model.parameters()))
    assert all(int(state['step'])==2 for state in optimizer.state.values())
    for state in optimizer.state.values():
        assert torch.isfinite(state['exp_avg']).all() and torch.isfinite(state['exp_avg_sq']).all()
    assert len(saved['rng_state_by_rank'])==saved['runtime']['world_size']
    scaler=saved['scaler'];assert scaler['scale']>0
    # Exercise the actual cached sampler and decoder at the FID batch size.
    # This private checkpoint is still discarded before fresh production.
    from src.training.rq_imagenet_recipes import LaserRecipe
    torch.cuda.set_device(0)
    device=torch.device('cuda:0')
    profile=LaserRecipe(BASE)
    aux,_=profile.tokenizer(device)
    model.eval().to(device)
    profile.adapt(model,aux)
    count=int(recipe.evaluation.batch_size)
    with torch.inference_mode(),torch.random.fork_rng(devices=[0]):
        torch.manual_seed(261014);torch.cuda.manual_seed(261014)
        if recipe.get('target_mode')=='hard_sparse_tokens':
            verify_hard_encoding(profile,aux,config,device,out)
        tokens=profile.sample(model,aux,count,torch.arange(count,device=device)%1000)
        assert tokens.shape==(count,8,8,2*int(recipe.tokenizer.depth))
        atoms=tokens[...,0::2];coefficients=tokens[...,1::2]-aux.num_atoms
        assert ((atoms>=0)&(atoms<aux.num_atoms)).all()
        assert ((coefficients>=0)&(coefficients<aux.coeff_vocab_size)).all()
        assert (atoms.sort(-1).values.diff(dim=-1)>0).all()
        for chunk in tokens.split(8):
            images=profile.decode(aux,chunk)
            assert images.shape==(len(chunk),3,256,256) and torch.isfinite(images).all()
    write_json(out/'gpu-generation-verification.json',dict(passed=True,
        generated_images=count,production_sampler=True,production_generation_batch=count,
        token_shape=list(tokens.shape),unique_atoms=True,valid_coefficient_bins=True,
        finite_decoded_images=True,private_preflight=True,
        peak_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3))
    write_json(out/'smoke-reload-verification.json',dict(passed=True,global_step=2,
        strict_model_reload=True,optimizer_state_count=len(optimizer.state),all_adam_steps=2,
        scheduler_horizon=62600,scheduler_step=2,rng_streams=len(saved['rng_state_by_rank']),
        private_smoke_discarded_for_fresh_production=True,checkpoint_bytes=checkpoint.stat().st_size,
        production_size_gpu_sampling_verified=True))
    print('LASER checkpoint reload, production-size GPU sampler and decoder verified',flush=True)


if __name__=='__main__':main()
