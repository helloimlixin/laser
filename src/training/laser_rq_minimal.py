"""The RQ recipe adapted to LASER, with one token cross-entropy objective.

The released spatial/depth transformer sizes are retained. Four RQ codes
become four atom/coefficient pairs, so the local decoder has eight events.
Physical pair inputs and support uniqueness are representation adaptations;
there are no additional attention blocks, prediction heads, or loss terms.
"""
import hashlib

import torch
from omegaconf import OmegaConf

from src.models.laser_rq_adapter import LaserPairRQTransformer
from src.rq_upstream import RQTransformerConfig


def build_model(arch, *, num_atoms, coefficient_bins, depth):
    arch = OmegaConf.create(OmegaConf.to_container(arch, resolve=True))
    arch.block_size = [*arch.block_size[:2], 2 * depth]
    arch.vocab_size = num_atoms + coefficient_bins
    config = RQTransformerConfig.create(arch)
    model = LaserPairRQTransformer(config, num_atoms)
    model.apply(model._init_weights)
    return model


@torch.no_grad()
def image_targets(aux, images, teacher, *, backend="eager"):
    from src.training.stochastic_image_pairs import stochastic_soft_image_targets
    # Bound the frozen FP32 encoder's activation memory. The chunk size and
    # bank size are recorded as part of the reproducible teacher RNG policy.
    chunks = []
    with torch.autocast(device_type=images.device.type, enabled=False):
        for chunk in images.split(int(teacher.encoder_batch_size)):
            chunks.append(stochastic_soft_image_targets(
                aux, chunk.float(), atom_temperature=float(teacher.atom_temperature),
                coefficient_temperature=float(teacher.coefficient_temperature),
                variants=int(teacher.variants),
                site_chunk_size=int(teacher.site_chunk_size), backend=backend))
    return {key: torch.cat([chunk[key] for chunk in chunks]) for key in chunks[0]}


def token_cross_entropy(logits, targets):
    """Mean CE per scalar token: (atom CE + coefficient CE) / 2.

    Both terms are the chain rule for the same pair likelihood. Atom labels
    condition on the observed earlier pairs; coefficient labels additionally
    condition on the current sampled atom. No auxiliary objective is called.
    """
    from src.training.omp_joint_targets import OMPJointTargets, omp_joint_cross_entropy
    joint = OMPJointTargets(targets["atom_ids"], targets["atom_weights"],
                            targets["coefficient_probabilities"], None)
    return omp_joint_cross_entropy(logits["atom_logits"], logits["coeff_logits"], joint) / 2


@torch.no_grad()
def verify_targets(aux, targets):
    tokens = targets["tokens"]
    atoms = tokens[..., 0::2]
    coefficients = tokens[..., 1::2] - aux.num_atoms
    assert tokens.shape[-1] == 2 * aux.sparsity_level
    assert ((atoms >= 0) & (atoms < aux.num_atoms)).all()
    assert ((coefficients >= 0) & (coefficients < aux.coeff_vocab_size)).all()
    assert (atoms.sort(-1).values.diff(dim=-1) > 0).all()
    for key in ("atom_weights", "coefficient_probabilities"):
        values = targets[key]
        assert values.dtype == torch.float32 and torch.isfinite(values).all()
        assert (values >= 0).all()
        torch.testing.assert_close(values.sum(-1), torch.ones_like(atoms, dtype=torch.float32),
                                   rtol=2e-5, atol=2e-5)
    for depth in range(1, aux.sparsity_level):
        reused = (targets["atom_ids"][..., depth, :, None] == atoms[..., None, :depth]).any(-1)
        assert (targets["atom_weights"][..., depth, :][reused] == 0).all()


def frozen_digest(aux):
    digest = hashlib.sha256()
    for name, value in aux.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
