import pytest
import torch

from src.training.image_view_cache import sample_image_views, validate_image_view_cache


def test_selection_preserves_whole_image_and_paired_coefficients_and_replays():
    atoms = torch.arange(12 * 3 * 2 * 2 * 4).reshape(12, 3, 2, 2, 4)
    coefficients = atoms.float() * .25 - 2
    generator = torch.Generator().manual_seed(17)
    state = generator.get_state()
    selected, coeffs, choices = sample_image_views(atoms, coefficients, generator=generator)
    for image, view in enumerate(choices):
        assert torch.equal(selected[image], atoms[image, view])
    torch.testing.assert_close(coeffs, selected.float() * .25 - 2)
    second = sample_image_views(atoms, coefficients, generator=generator)
    assert not torch.equal(choices, second[2])
    generator.set_state(state)
    for first, replay in zip((selected, coeffs, choices),
                             sample_image_views(atoms, coefficients, generator=generator)):
        assert torch.equal(first, replay)


def test_rejects_sitewise_choices_and_unidentified_or_misordered_caches():
    atoms = torch.zeros(2, 3, 2, 2, 4, dtype=torch.int16)
    coeffs = atoms.float()
    labels = torch.zeros(2, dtype=torch.long)
    meta = dict(views_per_image=3, shape=[2, 2, 4], image_view_cache_identity="test")
    validate_image_view_cache(atoms, coeffs, labels, meta)
    with pytest.raises(ValueError, match="one view"):
        sample_image_views(atoms, coeffs, choices=torch.zeros(2, 2, 2, dtype=torch.long))
    with pytest.raises(ValueError, match="provenance"):
        validate_image_view_cache(atoms, coeffs, labels, {**meta, "image_view_cache_identity": None})
    with pytest.raises(ValueError, match="shape"):
        validate_image_view_cache(atoms, coeffs, labels, {**meta, "shape": [3, 2, 4]})


def test_training_cache_loader_roundtrip_and_prefix_rejection(tmp_path):
    from src.training.image_view_cache import IMAGE_VIEW_FORMAT
    from src.training.rqtransformer import SparseTokenCacheDataset

    atoms = torch.arange(2 * 3 * 2 * 2 * 4, dtype=torch.int16).reshape(2, 3, 2, 2, 4)
    payload = dict(atoms=atoms, coeffs=atoms.float() / 100,
                   labels=torch.tensor([14, 91]),
                   meta=dict(format=IMAGE_VIEW_FORMAT, views_per_image=3,
                             shape=[2, 2, 4], image_view_cache_identity="roundtrip"))
    path = tmp_path / "views.pt"
    torch.save(payload, path)
    for in_memory in (False, True):
        dataset = SparseTokenCacheDataset(path, in_memory=in_memory)
        assert len(dataset) == 2
        a, c, label = dataset[1]
        assert torch.equal(a, atoms[1]) and label == 91
        torch.testing.assert_close(c, payload["coeffs"][1])
    with pytest.raises(ValueError, match="causal prefix"):
        SparseTokenCacheDataset(path, include_prefix_coeffs=True)
