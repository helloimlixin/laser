"""Fixed ImageNet 10x8 preview using the earlier stage-2 class order and layout.

Rendering helpers are copied unchanged from src.training.rqtransformer to avoid
importing that separate training stack into the released RQ workflow.
"""
import json
from pathlib import Path
import torch
import torch.distributed as dist
from PIL import Image, ImageDraw, ImageFont

REQUESTED_CLASSES = (
    (9, "ostrich"), (22, "bald eagle"), (90, "lorikeet"),
    (200, "tibetan terrier"), (289, "snow leopard"), (849, "teapot"),
    (106, "wombat"), (277, "red fox"), (258, "samoyed"), (926, "hotpot"),
)
GRID_SEED = 20260814
SAMPLES_PER_CLASS = 8

def _preview_font(size: int):
    """Load a stable readable font while retaining a Pillow-only fallback."""
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=int(size))
        except OSError:
            continue
    return ImageFont.load_default()


def _wrap_class_name(draw, class_name: str, font, max_width: int):
    """Wrap a class name to the label column using measured pixel widths."""
    words = str(class_name).replace("_", " ").split()
    if not words:
        return ["unknown"]
    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _class_name_layout(draw, class_name: str, max_width: int, max_height: int):
    """Choose the largest font whose wrapped class name fits one image row."""
    for size in range(30, 13, -1):
        font = _preview_font(size)
        lines = _wrap_class_name(draw, class_name, font, max_width)
        text = "\n".join(lines)
        bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=4)
        if bbox[2] - bbox[0] <= max_width and bbox[3] - bbox[1] <= max_height:
            return text, font, bbox
    font = _preview_font(13)
    text = "\n".join(_wrap_class_name(draw, class_name, font, max_width))
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=3)
    return text, font, bbox


def save_class_labeled_grid(images: torch.Tensor, chosen_classes: torch.Tensor,
                            class_names, target: Path, *, samples_per_class: int = 8,
                            label_width: int = 256):
    """Save paper-style class rows with an adjacent left-hand label column.

    Image tiles are pasted edge-to-edge: there is no padding between columns,
    no padding between rows, and no margin around the image mosaic.
    """
    images = images.detach().cpu().to(torch.float32).clamp(0, 1)
    samples_per_class = int(samples_per_class)
    chosen = [int(value) for value in chosen_classes.detach().cpu().tolist()]
    expected = len(chosen) * samples_per_class
    if images.ndim != 4 or images.shape[0] != expected:
        raise ValueError(
            f"expected {expected} BCHW preview images for {len(chosen)} class rows, "
            f"got {tuple(images.shape)}"
        )
    if images.shape[1] not in (1, 3):
        raise ValueError(f"preview images need 1 or 3 channels, got {images.shape[1]}")
    tile_height, tile_width = int(images.shape[2]), int(images.shape[3])
    label_width = max(1, int(label_width))
    canvas = Image.new(
        "RGB",
        (label_width + samples_per_class * tile_width, len(chosen) * tile_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for row, class_index in enumerate(chosen):
        row_y = row * tile_height
        class_name = (
            str(class_names[class_index])
            if class_names is not None and 0 <= class_index < len(class_names)
            else f"class {class_index}"
        )
        text, font, bbox = _class_name_layout(
            draw, class_name, max_width=label_width - 24, max_height=tile_height - 24
        )
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_x = max(12, (label_width - text_width) // 2)
        text_y = row_y + max(12, (tile_height - text_height) // 2 - bbox[1])
        draw.multiline_text(
            (text_x, text_y), text, font=font, fill="black", spacing=4, align="center"
        )
        for column in range(samples_per_class):
            index = row * samples_per_class + column
            array = images[index].mul(255).round().to(torch.uint8)
            array = array.permute(1, 2, 0).contiguous().numpy()
            if array.shape[-1] == 1:
                tile = Image.fromarray(array[..., 0], mode="L").convert("RGB")
            else:
                tile = Image.fromarray(array, mode="RGB")
            canvas.paste(
                tile,
                (label_width + column * tile_width, row_y),
            )
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target)
    return target


@torch.inference_mode()
def sample_requested_grid(model, tokenizer, output, step, epoch, *, device, rank=0, world=1,
                          sampler_name='original'):
    """Generate fixed class rows while preserving the training RNG and mode."""
    from src.scaled_atom_sampling import SAMPLER_SETTINGS,sample_codes
    settings=SAMPLER_SETTINGS[sampler_name]
    labels=torch.tensor([c for c,_ in REQUESTED_CLASSES],device=device).repeat_interleave(SAMPLES_PER_CLASS)
    assert len(labels)%world==0 and 0<=rank<world
    positions=torch.arange(rank,len(labels),world,device=device)
    was_training=model.training
    model.eval()
    try:
        with torch.random.fork_rng(devices=[device.index] if device.type=='cuda' else []):
            torch.manual_seed(GRID_SEED+rank)
            image_parts,code_parts=[],[]
            for ids in positions.split(8):
                if settings['mode']=='joint':
                    codes=model.sample(torch.zeros(len(ids),8,8,4,device=device,dtype=torch.long),
                        model_aux=tokenizer,cond=labels[ids],temperature=settings['temperature'],
                        top_k=settings['top_k'],top_p=settings['top_p'],amp=True,cached=True,is_tqdm=False)
                else:
                    codes=sample_codes(model,tokenizer,labels[ids],settings)
                decoded=tokenizer.decode_code(codes).mul(.5).add(.5).clamp(0,1)
                assert torch.isfinite(decoded).all()
                assert codes.min()>=0 and codes.max()<tokenizer.quantizer.vocab_size
                image_parts.append(decoded);code_parts.append(codes)
            local_images,local_codes=torch.cat(image_parts),torch.cat(code_parts)
            if world>1:
                gathered_images=[torch.empty_like(local_images) for _ in range(world)]
                gathered_codes=[torch.empty_like(local_codes) for _ in range(world)]
                dist.all_gather(gathered_images,local_images)
                dist.all_gather(gathered_codes,local_codes)
            else:
                gathered_images,gathered_codes=[local_images],[local_codes]
            target=None
            if rank==0:
                images=torch.empty((len(labels),*local_images.shape[1:]),dtype=local_images.dtype)
                codes=torch.empty((len(labels),8,8,4),dtype=torch.long)
                for worker,(xs,cs) in enumerate(zip(gathered_images,gathered_codes)):
                    images[worker::world]=xs.cpu();codes[worker::world]=cs.cpu()
                names=[f'class {i}' for i in range(1000)]
                for class_id,name in REQUESTED_CLASSES:names[class_id]=name
                target=Path(output)/f'class-grid-step{step:07d}-10x8.png'
                save_class_labeled_grid(images,torch.tensor([c for c,_ in REQUESTED_CLASSES]),
                    names,target,samples_per_class=SAMPLES_PER_CLASS)
                torch.save(codes,target.with_suffix('.codes.pt'))
                target.with_suffix('.json').write_text(json.dumps(dict(
                    optimizer_step=step,epoch=epoch,rows=10,samples_per_class=8,
                    classes=[dict(id=c,name=n) for c,n in REQUESTED_CLASSES],labels=labels.cpu().tolist(),
                    seed=GRID_SEED,rank_seed_rule='seed + rank',world_size=world,sample_batch_size=8,
                    sampler_name=sampler_name,**settings,label_width=256,
                    layout='one labeled class per row; eight edge-to-edge images per row',
                    image=str(target),source_format='archive/scripts/sample_imagenet_requested_classes.py'),indent=2)+'\n')
    finally:
        model.train(was_training)
    return target
