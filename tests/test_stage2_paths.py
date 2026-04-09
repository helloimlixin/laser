from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage2_paths import (
    default_token_cache_path,
    infer_latest_stage1_checkpoint,
    infer_latest_stage2_checkpoint,
    infer_latest_token_cache,
)


def test_infer_latest_stage1_stage2_and_token_cache(tmp_path: Path):
    output_root = tmp_path / "outputs"
    stage1_old = output_root / "checkpoints" / "run_20260325_000000" / "laser" / "last.ckpt"
    stage1_new = output_root / "checkpoints" / "run_20260326_000000" / "laser" / "last.ckpt"
    stage2_old = output_root / "ar" / "checkpoints" / "s2_20260325_000000" / "last.ckpt"
    stage2_new = output_root / "ar" / "checkpoints" / "s2_20260326_000000" / "last.ckpt"
    cache_old = output_root / "ar" / "token_cache" / "celeba__train__img128__cb256.pt"
    cache_new = output_root / "ar" / "token_cache" / "celeba__train__img128__cb512.pt"

    for path in (stage1_old, stage1_new, stage2_old, stage2_new, cache_old, cache_new):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name, encoding="utf-8")

    stage1_old.touch()
    stage1_new.touch()
    stage2_old.touch()
    stage2_new.touch()
    cache_old.touch()
    cache_new.touch()

    assert infer_latest_stage1_checkpoint(output_root=output_root, model_type="laser") == stage1_new.resolve()
    assert infer_latest_stage2_checkpoint(ar_output_dir=output_root / "ar") == stage2_new.resolve()
    assert infer_latest_token_cache(ar_output_dir=output_root / "ar", dataset="celeba", split="train") == cache_new.resolve()


def test_default_token_cache_path_uses_standardized_outputs_ar_layout(tmp_path: Path):
    path = default_token_cache_path(
        ar_output_dir=tmp_path / "outputs" / "ar",
        dataset="celeba",
        split="train",
        image_size=128,
        coeff_bins=256,
        coeff_quantization="uniform",
    )

    assert path == (tmp_path / "outputs" / "ar" / "token_cache" / "celeba__train__img128__cb256.pt").resolve()
