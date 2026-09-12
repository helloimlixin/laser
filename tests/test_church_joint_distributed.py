from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from src.church_joint_distributed import JointObjective, backward_batch, local_chunks, generation_range
from src.church_joint_geometry import make_prior
from tests.test_church_ffhq_archived import tiny


def _gradient_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://'+rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(29)
        old, aux, packed=tiny()
        prior=make_prior(config=old.config,num_atoms=7,coeff_vocab_size=8)
        serial=make_prior(config=old.config,num_atoms=7,coeff_vocab_size=8)
        serial.load_state_dict(prior.state_dict())
        data={'atoms':(packed//8).repeat(4,1,1,1),
            'coefficients':(aux.coeff_bins[packed%8]*aux.coeff_scales).repeat(4,1,1,1)}
        wrapped=DDP(JointObjective(prior,aux),broadcast_buffers=False,find_unused_parameters=True)
        reference=JointObjective(serial,aux)
        optimizer=torch.optim.AdamW(prior.parameters(),lr=5e-5,betas=(.9,.95),weight_decay=1e-4)
        reference_optimizer=torch.optim.AdamW(serial.parameters(),lr=5e-5,betas=(.9,.95),weight_decay=1e-4)
        for step in (4931,4932):
            optimizer.zero_grad(set_to_none=True);reference_optimizer.zero_grad(set_to_none=True)
            metrics=backward_batch(wrapped,torch.arange(8),data,.05,step,0,2,rank,2)
            expected=backward_batch(reference,torch.arange(8),data,.05,step,0,2,0,1)
            for key in metrics:assert abs(metrics[key]-expected[key])<2e-6
            for actual,ref in zip(prior.parameters(),serial.parameters()):
                if actual.grad is None:assert ref.grad is None
                else:torch.testing.assert_close(actual.grad,ref.grad,rtol=2e-4,atol=2e-6)
            torch.nn.utils.clip_grad_norm_(prior.parameters(),1.)
            torch.nn.utils.clip_grad_norm_(serial.parameters(),1.)
            optimizer.step();reference_optimizer.step()
            for actual,ref in zip(prior.parameters(),serial.parameters()):
                torch.testing.assert_close(actual,ref,rtol=2e-4,atol=2e-6)
    finally:dist.destroy_process_group()


def test_real_ddp_joint_geometry_matches_serial_microbatches(tmp_path):
    mp.spawn(_gradient_worker,args=(str(tmp_path/'rendezvous'),),nprocs=2,join=True)


def test_training_shards_preserve_all_indices_and_global_seeds():
    indices=torch.randperm(256)
    chunks=[c for r in range(2) for c in local_chunks(indices,32,r,2)]
    assert [i for i,_ in chunks]==list(range(8))
    assert torch.equal(torch.cat([v for _,v in chunks]),indices)


def test_generation_shards_cover_exactly_requested_population():
    for count in (4096,50000,513):
        ranges=[generation_range(count,r,2) for r in range(2)]
        assert ranges[0][0]==0 and ranges[0][1]==ranges[1][0] and ranges[1][1]==count
