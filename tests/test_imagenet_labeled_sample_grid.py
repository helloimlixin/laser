import numpy as np
import torch
from PIL import Image

from scripts.train_official_rqtransformer_laser_stage2 import (
    save_class_labeled_grid,
    save_unlabeled_grid,
)


def test_class_labeled_grid_has_eight_edge_to_edge_samples_per_row(tmp_path):
    rows = 2
    samples_per_class = 8
    tile_height = 4
    tile_width = 5
    images = torch.empty(rows * samples_per_class, 3, tile_height, tile_width)
    expected_colors = []
    for index in range(len(images)):
        color = torch.tensor(
            [(index + 1) / 20.0, (index + 2) / 20.0, (index + 3) / 20.0]
        )
        images[index] = color[:, None, None]
        expected_colors.append(color.mul(255).round().to(torch.uint8).numpy())

    target = tmp_path / "labeled.png"
    save_class_labeled_grid(
        images,
        torch.tensor([3, 7]),
        [f"class name {index}" for index in range(10)],
        target,
        samples_per_class=samples_per_class,
        label_width=64,
    )

    rendered = np.asarray(Image.open(target).convert("RGB"))
    assert rendered.shape == (
        rows * tile_height,
        64 + samples_per_class * tile_width,
        3,
    )
    # Every image cell, including pixels directly on all internal boundaries,
    # contains only its source tile color. Any padding or gutter would fail.
    for row in range(rows):
        for column in range(samples_per_class):
            index = row * samples_per_class + column
            tile = rendered[
                row * tile_height:(row + 1) * tile_height,
                64 + column * tile_width:64 + (column + 1) * tile_width,
            ]
            assert np.all(tile == expected_colors[index])


def test_unconditional_grid_is_exactly_eight_by_eight_without_spacing(tmp_path):
    grid_size = 8
    tile_height = 4
    tile_width = 5
    images = torch.empty(grid_size * grid_size, 3, tile_height, tile_width)
    expected_colors = []
    for index in range(len(images)):
        color = torch.tensor(
            [(index + 1) / 70.0, (index + 2) / 70.0, (index + 3) / 70.0]
        )
        images[index] = color[:, None, None]
        expected_colors.append(color.mul(255).round().to(torch.uint8).numpy())

    target = tmp_path / "unconditional.png"
    save_unlabeled_grid(images, target, nrow=grid_size)

    rendered = np.asarray(Image.open(target).convert("RGB"))
    assert rendered.shape == (
        grid_size * tile_height,
        grid_size * tile_width,
        3,
    )
    for row in range(grid_size):
        for column in range(grid_size):
            index = row * grid_size + column
            tile = rendered[
                row * tile_height:(row + 1) * tile_height,
                column * tile_width:(column + 1) * tile_width,
            ]
            assert np.all(tile == expected_colors[index])
