"""Random initialization for the original Church compound architecture."""
import hashlib
import torch

from scripts.train_official_rqtransformer_laser_stage2 import build_model
from src.church_epoch50_loop import GatedDepthLoop
from src.coefficient_history_training import EpochStream


def scratch_prior(variant='looped'):
    """Build fresh stage-2 tensors. No checkpoint or initialization path exists."""
    if variant not in {'looped', 'control'}:
        raise ValueError('Expected looped or control')
    model = build_model(18432, 16384, compound=True, coeff_vocab_size=2048,
        compound_micro_transformer_layers=2, compound_depth_specific_coeff_heads=True,
        compound_pair_autoregressive=True, sparsity_level=4, model_preset='lsun-church-350m')
    if variant == 'looped':
        model.head_transformer = GatedDepthLoop(list(model.head_transformer.blocks))
    return model


def initialization_audit(model, seed):
    digest = hashlib.sha256()
    tensors = 0
    for name, value in sorted(model.state_dict().items()):
        if name.endswith('loop_gates'):
            continue
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        tensors += 1
    return {'kind': 'random', 'seed': seed, 'stage2_checkpoint_loaded': None,
            'common_tensor_sha256': digest.hexdigest(), 'common_tensor_count': tensors,
            'initial_optimizer_step': 0}


class FullBatchEpochStream(EpochStream):
    """Match the original effective batches, dropping each epoch's remainder."""
    def next(self, batch_size):
        if self.size < batch_size:
            raise ValueError('Population must contain at least one full batch')
        if self.position == self.size or self.size - self.position < batch_size:
            self.position = self.size
            # Parent starts the next epoch and draws its next permutation.
        indices, progress, _ = super().next(batch_size)
        end = self.size - self.position < batch_size
        if end:
            self.position = self.size
            progress = float(self.epoch + 1)
        return indices, progress, end
