"""Fixed CC3M text-to-image prompts and a caption-column sample grid."""
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont
import torch

PROMPTS = (
    'a cheeseburger in front of a mountain range covered with snow',
    'a cherry blossom tree on the blue ocean',
    'a river has burst its banks and has spread out onto arable farmland alongside',
    'a photograph of crowd of people under cherry blossom trees',
    'a small house in the wilderness',
    'a small house on the shore',
    'night landscape of the skyline of a city',
    'an illustration of a cathedral',
    'the fountain in the square',
)
SAMPLES_PER_PROMPT = 8


def conditioning_tokens(device):
    from src.data.cc3m_joint import text_tokenizer
    tokenizer = text_tokenizer(dropout=0.)
    text = torch.tensor([tokenizer.encode(prompt).ids for prompt in PROMPTS],
                        dtype=torch.long, device=device)
    return text.repeat_interleave(SAMPLES_PER_PROMPT, dim=0)


def save_text_image_grid(pixels, path):
    if pixels.shape != (len(PROMPTS)*SAMPLES_PER_PROMPT, 3, 256, 256):
        raise ValueError('Expected nine prompts with eight 256px samples each')
    data = pixels.detach().float().clamp(0, 1).mul(255).round().to(torch.uint8)
    data = data.cpu().permute(0, 2, 3, 1).numpy()
    margin, caption_width, caption_gap, column_gap, row_gap = 24, 480, 16, 8, 16
    image_start = margin+caption_width+caption_gap
    width = image_start+8*256+7*column_gap+margin
    height = 2*margin+len(PROMPTS)*256+(len(PROMPTS)-1)*row_gap
    grid = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(grid)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 28)
    for row, prompt in enumerate(PROMPTS):
        y = margin+row*(256+row_gap)
        lines = textwrap.wrap(prompt, width=30, break_long_words=False)
        label = '\n'.join(lines)
        box = draw.multiline_textbbox((0, 0), label, font=font, spacing=9)
        draw.multiline_text((margin, y+(256-(box[3]-box[1]))//2), label,
                            font=font, fill='black', spacing=9)
        for column in range(SAMPLES_PER_PROMPT):
            grid.paste(Image.fromarray(data[row*SAMPLES_PER_PROMPT+column]),
                       (image_start+column*(256+column_gap), y))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(path)
    return grid
