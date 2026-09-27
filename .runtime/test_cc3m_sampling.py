from pathlib import Path
import importlib.util
import torch
from PIL import Image

project = Path('/scratch/xl598/Projects/laser')
module_path = project / 'scripts/train_official_rqtransformer_laser_stage2.py'
spec = importlib.util.spec_from_file_location('stage2_cc3m', module_path)
stage2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage2)

conditions = stage2.encode_cc3m_prompts(stage2.RQ_VAE_PAPER_PROMPTS)
assert tuple(conditions.shape) == (8, 32)
assert int(conditions.min()) >= 0 and int(conditions.max()) < 16384

from src.text_conditioning import encode_texts
reference, _, _ = encode_texts(
    stage2.RQ_VAE_PAPER_PROMPTS,
    max_length=32,
    tokenizer='rq_bpe16k',
)
assert torch.equal(conditions, reference)

dataset = stage2.CC3MValidationDataset(
    Path('/scratch/xl598/Projects/data/cc3m'),
    transform=stage2.val_image_transform(),
)
assert len(dataset) == 13443
prompts_a = dataset.random_prompts(8, seed=12345)
prompts_b = dataset.random_prompts(8, seed=12345)
assert prompts_a == prompts_b and len(prompts_a) == 8
assert all(prompt.strip() for prompt in prompts_a)

images = torch.zeros(64, 3, 256, 256)
for index in range(64):
    images[index, index % 3] = (index + 1) / 64
target = project / '.runtime/cc3m_paper_style_grid_test.png'
stage2.save_paper_style_text_grid(images, stage2.RQ_VAE_PAPER_PROMPTS, target)
with Image.open(target) as grid:
    assert grid.size == (2560, 2048)
    pixels = grid.load()
    for row in range(8):
        for column in range(8):
            index = row * 8 + column
            expected = [0, 0, 0]
            expected[index % 3] = round(255 * (index + 1) / 64)
            for x, y in (
                (512 + column * 256, row * 256),
                (512 + column * 256 + 255, row * 256 + 255),
            ):
                actual = pixels[x, y]
                assert max(abs(actual[channel] - expected[channel]) for channel in range(3)) <= 1

import clip
clip_model, _ = clip.load(
    'ViT-B/32', device='cpu', jit=False,
    download_root='/scratch/xl598/.cache/clip',
)
assert clip_model.visual.input_resolution == 224
print('paper_conditions', tuple(conditions.shape), int(conditions.max()))
print('validation_pairs', len(dataset))
print('random_prompts', prompts_a)
print('grid_size', Image.open(target).size)
print('clip_model', 'ViT-B/32', clip_model.visual.input_resolution)
