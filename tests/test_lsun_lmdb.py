import io

import lmdb
from PIL import Image
import pytest

from src.data.config import DataConfig
from src.data.image_folder import ImageFolderDataModule


@pytest.mark.parametrize("category", ["church", "church_outdoor", "bedroom", "cat"])
def test_lsun_lmdb_train_and_validation_load_as_images(tmp_path, monkeypatch, category):
    # Torchvision writes its LSUN key index to the working directory.
    monkeypatch.chdir(tmp_path)
    for split, count in [("train", 3), ("val", 2)]:
        db = tmp_path / f"{category}_{split}_lmdb"
        db.mkdir()
        with lmdb.open(str(db), map_size=1 << 20) as env:
            with env.begin(write=True) as txn:
                for index in range(count):
                    stream = io.BytesIO()
                    Image.new("RGB", (20, 20), (index, 50, 100)).save(stream, format="JPEG")
                    txn.put(str(index).encode(), stream.getvalue())
    dataset_category = "church" if category == "church_outdoor" else category
    cfg = DataConfig(dataset=f"lsun_{dataset_category}", data_dir=str(tmp_path / dataset_category),
                     image_size=16, batch_size=2, num_workers=0)
    dm = ImageFolderDataModule(cfg)
    try:
        dm.setup("fit")
        assert len(dm.train_dataset) == 3
        assert len(dm.val_dataset) == 2
        assert dm.test_dataset is dm.val_dataset
        image, _ = dm.train_dataset[0]
        assert image.shape == (3, 16, 16)
        assert next(iter(dm.val_dataloader()))[0].shape == (2, 3, 16, 16)
    finally:
        for dataset in (dm.train_dataset, dm.val_dataset):
            if dataset is not None:
                dataset.env.close()
