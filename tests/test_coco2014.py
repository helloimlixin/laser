import io
import json
import pickle
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from src.data.coco2014 import COCO2014DataModule, COCO2014Dataset
from src.data.config import DataConfig


@pytest.fixture
def coco_root(tmp_path):
    (tmp_path / "annotations").mkdir()
    for split, ids in (("train", [11, 22]), ("val", [33])):
        images, annotations = [], []
        with zipfile.ZipFile(tmp_path / f"{split}2014.zip", "w") as z:
            for image_id in ids:
                name = f"COCO_{split}2014_{image_id:012d}.jpg"
                im = Image.new("RGB", (32, 24), color=(image_id, 0, 0))
                buf = io.BytesIO()
                im.save(buf, format="JPEG")
                z.writestr(f"{split}2014/{name}", buf.getvalue())
                images.append({"id": image_id, "file_name": name, "width": 32, "height": 24})
                for caption_id in range(5):
                    annotations.append({"image_id": image_id, "id": 10 * image_id + caption_id,
                                        "caption": f"image {image_id}, caption {caption_id}"})
        # Deliberately unordered input: annotation IDs define deterministic selection.
        (tmp_path / "annotations" / f"captions_{split}2014.json").write_text(
            json.dumps({"images": images[::-1], "annotations": annotations[::-1]})
        )
    return tmp_path


def test_zip_keeps_all_captions_and_split_identity(coco_root):
    first = COCO2014Dataset(coco_root)
    pairs = COCO2014Dataset(coco_root, caption_mode="all")
    val = COCO2014Dataset(coco_root, "val")
    assert len(first) == 2 and len(pairs) == 10 and len(val) == 1
    assert {pairs[i][1] for i in range(len(pairs))} == {
        f"image {image_id}, caption {c}" for image_id in [11, 22] for c in range(5)
    }
    assert first[0][1] == "image 11, caption 0"
    assert val[0][1] == "image 33, caption 0"
    assert set(first.image_paths).isdisjoint(val.image_paths)
    assert first[0][0].mode == "RGB"
    assert len(COCO2014Dataset(coco_root, caption_mode="all", max_items=3)) == 3


def test_zip_handles_survive_parent_reads_fork_and_pickle(coco_root):
    data = COCO2014Dataset(coco_root, transform=ToTensor(), caption_mode="all")
    expected_image, expected_caption = data[0]  # Opens a handle in the parent.
    restored = pickle.loads(pickle.dumps(data))
    assert torch.equal(restored[0][0], expected_image)
    assert restored[0][1] == expected_caption
    for context in ["fork", "spawn"]:
        loader = DataLoader(data, batch_size=2, num_workers=2, multiprocessing_context=context)
        captions = [text for _, texts in loader for text in texts]
        assert captions == [data[i][1] for i in range(len(data))]


def test_extracted_images_match_archive_and_missing_image_fails(coco_root):
    zipped = COCO2014Dataset(coco_root)
    expected = zipped[0][0].tobytes()
    with zipfile.ZipFile(coco_root / "train2014.zip") as z:
        z.extractall(coco_root)
    extracted = COCO2014Dataset(coco_root)
    assert extracted[0][0].tobytes() == expected
    Path(extracted.image_paths[0]).unlink()
    with pytest.raises(FileNotFoundError, match="Missing COCO image"):
        COCO2014Dataset(coco_root)


def test_missing_caption_and_missing_validation_do_not_fallback(coco_root):
    ann = coco_root / "annotations/captions_train2014.json"
    payload = json.loads(ann.read_text())
    payload["annotations"] = [a for a in payload["annotations"] if a["image_id"] != 11]
    ann.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="has no caption"):
        COCO2014Dataset(coco_root)
    (coco_root / "annotations/captions_val2014.json").unlink()
    with pytest.raises(FileNotFoundError):
        COCO2014Dataset(coco_root, "val")


def test_datamodule_returns_caption_batches_and_unique_validation_images(coco_root):
    config = DataConfig(dataset="coco2014", data_dir=str(coco_root), image_size=16,
                        mean=(0.5,) * 3, std=(0.5,) * 3, batch_size=2,
                        num_workers=0, augment=False, coco_caption_mode="all")
    dm = COCO2014DataModule(config)
    dm.setup()
    assert len(dm.train_dataset) == 10
    assert len(dm.val_dataset) == 1
    assert dm.test_dataset is dm.val_dataset
    images, captions = next(iter(dm.train_dataloader()))
    assert images.shape == (2, 3, 16, 16)
    assert images.min() >= -1 and images.max() <= 1
    assert all(isinstance(c, str) and c for c in captions)


def test_recipes_and_cache_forward_all_captions(coco_root, tmp_path, monkeypatch):
    from src.training import cli, stage2
    from src.training.paths import ROOT
    from scripts.tools.build_token_cache import _build_datamodule, _batch_texts, _attach_text_metadata
    from src.data.token_cache import TokenCacheDataModule

    first = cli.load_config(ROOT / "configs/stage1/coco2014.yaml")
    cfg = cli.load_config(ROOT / "configs/stage2/coco2014.yaml")
    assert first.data.image_size == cfg.data.image_size == 256
    assert first.data.coco_caption_mode == "first"
    assert cfg.data.coco_caption_mode == "all"
    assert cfg.ar.text_conditional and cfg.token_cache.text_tokenizer == "char"
    assert cfg.token_cache.stage1_output_root == first.output_dir
    assert cfg.train_ar.validation_split == cfg.train_ar.test_split == 0
    assert cfg.train_ar.limit_val_batches == 0
    assert cfg.train_ar.checkpoint_monitor == "train/loss"

    checkpoint = tmp_path / "fixture.ckpt"
    checkpoint.touch()
    cfg.token_cache.stage1_checkpoint = str(checkpoint)
    cfg.token_cache.output = str(tmp_path / "cache.pt")
    calls = []
    monkeypatch.setattr(stage2.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    stage2._maybe_build_token_cache(cfg)
    assert calls[0][calls[0].index("--dataset") + 1] == "coco2014"
    assert calls[0][calls[0].index("--coco_caption_mode") + 1] == "all"

    args = SimpleNamespace(dataset="coco2014", data_dir=str(coco_root), batch_size=10,
                           num_workers=0, image_size=16, mean=(0.5,) * 3, std=(0.5,) * 3,
                           seed=42, max_items=0, coco_caption_mode="all")
    dm = _build_datamodule(args)
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    captions = _batch_texts(batch, 10)
    assert len(set(captions)) == 10
    cache = {"tokens_flat": torch.ones(10, 8, dtype=torch.long), "shape": (2, 2, 2), "meta": {}}
    _attach_text_metadata(cache, captions, max_length=64)
    assert cache["text_tokens"].shape == (10, 64)
    assert cache["text_mask"].any(dim=1).all()
    path = tmp_path / "tokens.pt"
    torch.save(cache, path)
    tokens = TokenCacheDataModule(str(path), batch_size=2, validation_fraction=0, test_fraction=0)
    tokens.setup()
    assert len(tokens.train_dataset) == 10
    assert tokens.val_dataset is None
    assert tokens.test_dataset is None
