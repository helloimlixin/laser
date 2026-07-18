import io
import json
import tarfile

import torch
from PIL import Image

from scripts.tools.build_token_cache import _attach_text_metadata, _batch_texts
from src.data.cc3m import CC3MDataModule
from src.data.config import DataConfig


def _jpeg_bytes(color=(128, 64, 32)) -> bytes:
    image = Image.new("RGB", (12, 10), color=color)
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return buf.getvalue()


def _add_bytes(tf: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    tf.addfile(info, io.BytesIO(payload))


def _write_cc3m_shard(root, captions):
    shard_dir = root / "wds"
    shard_dir.mkdir(parents=True)
    shard = shard_dir / "cc3m-train-0000.tar"
    with tarfile.open(shard, "w") as tf:
        for idx, caption in enumerate(captions):
            key = f"{idx:09d}"
            _add_bytes(tf, f"{key}.jpg", _jpeg_bytes())
            _add_bytes(tf, f"{key}.txt", caption.encode("utf-8"))
            _add_bytes(tf, f"{key}.json", json.dumps({"caption": caption}).encode("utf-8"))
    return shard


def test_cc3m_datamodule_reads_webdataset_captions(tmp_path):
    _write_cc3m_shard(tmp_path / "cc3m", ["a red boat", "a blue sky"])
    cfg = DataConfig(
        dataset="cc3m",
        data_dir=str(tmp_path / "cc3m"),
        batch_size=2,
        num_workers=0,
        image_size=16,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        augment=False,
    )

    dm = CC3MDataModule(cfg)
    dm.setup("fit")
    images, captions = next(iter(dm.train_dataloader()))

    assert images.shape == (2, 3, 16, 16)
    assert list(captions) == ["a red boat", "a blue sky"]
    assert len(dm.train_dataset.image_paths) == 2


def test_token_cache_text_metadata_from_caption_batch():
    batch = (torch.zeros(2, 3, 4, 4), ["A Red Boat", "A blue sky"])
    captions = _batch_texts(batch, keep=2)
    payload = {
        "tokens_flat": torch.zeros(2, 4, dtype=torch.int32),
        "shape": (1, 2, 2),
        "meta": {},
    }

    _attach_text_metadata(payload, captions, max_length=12)

    assert payload["text_tokens"].shape == (2, 12)
    assert payload["text_mask"].dtype == torch.bool
    assert payload["meta"]["has_text_conditioning"] is True
    assert payload["meta"]["text_max_length"] == 12
    assert payload["text"][0] == "a red boat"


def test_token_cache_text_metadata_supports_rq_bpe16k():
    payload = {
        "tokens_flat": torch.zeros(1, 4, dtype=torch.int32),
        "shape": (1, 2, 2),
        "meta": {},
    }

    _attach_text_metadata(payload, ["Eiffel tower on a desert."], max_length=32, tokenizer="rq_bpe16k")

    assert payload["text_tokens"].shape == (1, 32)
    assert payload["text_tokens"].max().item() < 16384
    assert payload["meta"]["text_tokenizer"] == "rq_bpe16k"
    assert payload["meta"]["text_vocab_size"] == 16384
    assert payload["meta"]["text_max_length"] == 32
    assert payload["meta"]["text_pad_id"] == 0
