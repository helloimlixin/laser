import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from src.models.dictionary_learner import DictionaryLearning


def _alternating_update_worker(rank, init_method):
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=2,
    )
    try:
        learner = DictionaryLearning(
            num_embeddings=4,
            embedding_dim=2,
            sparsity_level=1,
            dictionary_update_mode="alternating_residual",
            dictionary_update_relaxation=0.5,
            dictionary_update_max_atoms_per_step=4,
            dictionary_update_min_usage=1,
        )
        with torch.no_grad():
            learner.dictionary.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 2**-0.5, 2**-0.5],
                        [0.0, 1.0, 2**-0.5, -(2**-0.5)],
                    ]
                )
            )
        if rank == 0:
            z = torch.tensor(
                [[[[1.0, 0.9, 0.3, 0.4]], [[0.2, 0.3, 1.0, 0.9]]]]
            )
        else:
            z = torch.tensor(
                [[[[0.8, 1.0, 0.4, 0.2]], [[0.4, 0.1, 0.8, 1.0]]]]
            )

        learner(z)
        cached = learner._last_dictionary_update_batch

        def local_error(dictionary):
            support = cached["support"]
            values = cached["values"]
            atoms = dictionary.t()[support]
            reconstruction = (atoms * values.unsqueeze(-1)).sum(dim=1).t()
            return (cached["signals"] - reconstruction).square().sum()

        before = local_error(learner.dictionary.detach())
        updated = learner.alternating_dictionary_update_after_step_()
        after = local_error(learner.dictionary.detach())
        errors = torch.stack([before, after])
        dist.all_reduce(errors, op=dist.ReduceOp.SUM)

        dictionaries = [torch.empty_like(learner.dictionary) for _ in range(2)]
        dist.all_gather(dictionaries, learner.dictionary.detach())

        assert updated > 0
        assert errors[1] <= errors[0] + 1e-6
        assert torch.allclose(dictionaries[0], dictionaries[1], atol=1e-6)
        assert torch.allclose(
            learner.dictionary.norm(dim=0),
            torch.ones(4),
            atol=1e-6,
        )
    finally:
        dist.destroy_process_group()


def test_distributed_alternating_update_stays_synchronized(tmp_path):
    init_method = f"file://{tmp_path / 'dictionary-update-init'}"
    mp.spawn(
        _alternating_update_worker,
        args=(init_method,),
        nprocs=2,
        join=True,
    )


def _missing_cache_worker(rank, init_method):
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=2,
    )
    try:
        learner = DictionaryLearning(
            num_embeddings=4,
            embedding_dim=2,
            sparsity_level=1,
            dictionary_update_mode="alternating_residual",
            dictionary_update_min_usage=1,
        )
        if rank == 0:
            learner(torch.tensor([[[[1.0]], [[0.5]]]]))

        before = learner.dictionary.detach().clone()
        updated = learner.alternating_dictionary_update_after_step_()
        assert updated == 0
        assert torch.equal(learner.dictionary, before)

        # Reaching this collective proves the cache-validity decision returned
        # on every rank instead of abandoning peers inside the updater.
        reached = torch.ones((), dtype=torch.long)
        dist.all_reduce(reached, op=dist.ReduceOp.SUM)
        assert int(reached.item()) == 2
    finally:
        dist.destroy_process_group()


def test_distributed_alternating_update_skips_globally_for_missing_rank_cache(tmp_path):
    init_method = f"file://{tmp_path / 'dictionary-update-missing-cache-init'}"
    mp.spawn(
        _missing_cache_worker,
        args=(init_method,),
        nprocs=2,
        join=True,
    )
