import torch

import src.data.imagenet_labels as imagenet_labels
from cache import _attach_class_label_metadata
from src.stage2_preview import _class_label_texts, _write_class_label_manifest


def test_imagenet_synset_cache_displays_readable_class_names(monkeypatch):
    monkeypatch.setattr(
        imagenet_labels,
        "imagenet1k_categories",
        lambda: ("tench", "goldfish", "great white shark"),
    )
    cache = {
        "meta": {
            "dataset": "imagenet",
            "class_names": ["n01440764", "n01443537", "n01484850"],
        }
    }

    assert _class_label_texts(cache, torch.tensor([0, 2])) == [
        "tench (class 0)",
        "great white shark (class 2)",
    ]


def test_imagenet_manifest_keeps_class_name_and_synset(monkeypatch, tmp_path):
    monkeypatch.setattr(
        imagenet_labels,
        "imagenet1k_categories",
        lambda: ("tench", "goldfish", "great white shark"),
    )
    cache = {
        "meta": {
            "dataset": "imagenet",
            "class_names": ["n01440764", "n01443537", "n01484850"],
        }
    }

    out = tmp_path / "sample.png"
    _write_class_label_manifest(out, torch.tensor([1]), cache)

    text = out.with_suffix(".class_labels.txt").read_text(encoding="utf-8")
    tsv = out.with_suffix(".class_labels.tsv").read_text(encoding="utf-8")
    assert text == "goldfish (class 1)\n"
    assert "class_name\tclass_synset" in tsv
    assert "goldfish\tn01443537\tgoldfish (class 1)" in tsv


def test_cache_metadata_stores_imagenet_display_names_and_synsets(monkeypatch):
    monkeypatch.setattr(
        imagenet_labels,
        "imagenet1k_categories",
        lambda: ("tench", "goldfish", "great white shark"),
    )

    class Dataset:
        class_to_idx = {
            "n01440764": 0,
            "n01443537": 1,
            "n01484850": 2,
        }

    payload = {
        "tokens_flat": torch.zeros(3, 4, dtype=torch.int32),
        "meta": {"dataset": "imagenet"},
    }

    _attach_class_label_metadata(
        payload,
        torch.tensor([0, 1, 2]),
        dataset_name="imagenet",
        dataset=Dataset(),
    )

    assert payload["meta"]["class_names"] == ["tench", "goldfish", "great white shark"]
    assert payload["meta"]["class_synsets"] == ["n01440764", "n01443537", "n01484850"]
    assert payload["meta"]["num_classes"] == 3
    assert payload["class_labels"].tolist() == [0, 1, 2]
