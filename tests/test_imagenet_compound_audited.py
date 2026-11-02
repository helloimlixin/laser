import io
from pathlib import Path

from omegaconf import OmegaConf
import pytest
import torch
from PIL import Image

from src.training import rqtransformer as training
from src.training.cli import load_config
from src.training.rq_imagenet_runtime import TarImageNet
from src.training.options import options_to_argv
from src.training.warmup_cosine_schedule import WarmupCosineSchedule
from src.training.imagenet_compound_runtime import ImageNetCompoundAux


RECIPE = Path(__file__).resolve().parents[1]/'configs/stage2/imagenet-compound512-audited-8a100-20261026.yaml'


def test_recipe_builds_exact_zero_based_compound_geometry_and_real_warmup():
    cfg = load_config(RECIPE)
    args = training.build_parser().parse_args(options_to_argv(OmegaConf.to_container(cfg.options)))
    assert args.lr_schedule == 'warmup-cosine' and args.warmup_epochs == .5
    with torch.device('meta'):
        model = training.build_model(16896, 16384, coeff_vocab_size=512, sparsity_level=4,
            physical_pair_context=True, efficient_pair_backbone=True,
            compound_micro_transformer_layers=2, compound_depth_specific_coeff_heads=True,
            coefficient_history_layers=2, coefficient_history_width=512,
            coefficient_history_heads=8, coefficient_history_normalize_fields=True)
    assert sum(p.numel() for p in model.parameters()) == 1456551936
    assert model.token_vocabulary_size == 16896
    assert tuple(model.token_block_size) == (8, 8, 8)
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(()))], lr=args.lr)
    scheduler = WarmupCosineSchedule(optimizer, initial_lr=args.lr, min_lr=0.,
                                    total_steps=62600, warmup_steps=313)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(args.lr/313)
    for _ in range(313):
        scheduler.step()
    assert optimizer.param_groups[0]['lr'] == args.lr


def test_online_teacher_disables_autocast_reuses_gram_and_restores_precision(monkeypatch):
    aux = ImageNetCompoundAux.__new__(ImageNetCompoundAux)
    torch.nn.Module.__init__(aux)
    aux.register_buffer('dictionary', torch.randn(4, 7))
    calls = []
    def encode(self, images, **kwargs):
        calls.append(kwargs['dictionary_gram'])
        assert images.dtype == torch.float32 and not torch.is_autocast_enabled('cpu')
        assert not torch.backends.cuda.matmul.allow_tf32
        assert not torch.backends.cudnn.allow_tf32
        return images[:, :1], images[:, :1]
    monkeypatch.setattr(training.LaserAux, 'encode_sparse_components', encode)
    monkeypatch.setenv('LASER_ENCODER_BATCH', '2')
    before = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        result = aux.encode_sparse_components(torch.randn(5, 4).bfloat16())
        aux.encode_sparse_components(torch.randn(1, 4))
    assert result[0].shape == (5, 1)
    assert len(calls) == 4 and all(g is calls[0] for g in calls)
    assert before == (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)


def test_tar_augmentation_does_not_seed_cuda_and_preserves_cpu_rng(monkeypatch, tmp_path):
    pixels = io.BytesIO()
    Image.new('RGB', (300, 256)).save(pixels, format='JPEG')
    path = tmp_path/'images.bin'
    path.write_bytes(pixels.getvalue())
    dataset = TarImageNet.__new__(TarImageNet)
    dataset.archive, dataset.descriptor = str(path), None
    dataset.offsets, dataset.sizes, dataset.labels = [0], [len(pixels.getvalue())], [3]
    dataset.split, dataset.seed = 'train', 13
    dataset.epoch = torch.tensor(0)
    dataset.transform = lambda image: torch.rand(4)
    def forbidden(*args, **kwargs):
        raise AssertionError('Augmentation must not reset CUDA RNG streams')
    monkeypatch.setattr(torch, 'manual_seed', forbidden)
    before = torch.get_rng_state().clone()
    a, label = dataset[0]
    assert label == 3 and torch.equal(before, torch.get_rng_state())
    assert torch.equal(a, dataset[0][0])
    dataset.set_epoch(1)
    assert not torch.equal(a, dataset[0][0])
