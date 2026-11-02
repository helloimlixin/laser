"""Fixed ImageNet classes, with one overview and one reproducible grid per class."""
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist
from omegaconf import OmegaConf


def sampling_metadata(sampling):
    return OmegaConf.to_container(sampling, resolve=True) if OmegaConf.is_config(sampling) else dict(sampling)


def sampling_caption(sampling):
    s = sampling_metadata(sampling)
    if 'joint_temperature' in s:
        temperature = s.get('joint_temperature_by_depth', s['joint_temperature'])
        return f'Whole-pair T={temperature}; joint probability floor={s.get("joint_support_floor", 0.):g}; '
    if 'atom_temperature' not in s:
        return f'Pair-token T={s["temperature"]:g}, top-k={s["top_k"]}, top-p={s["top_p"]:g}; '
    return (f'atom T={s["atom_temperature"]:g}, top-k={s["atom_top_k"]}, top-p={s["atom_top_p"]:g}; '
            f'coefficient T={s["coefficient_temperature"]:g}, '
            f'top-k={s["coefficient_top_k"]}, top-p={s["coefficient_top_p"]:g}; ')


CLASSES = (
    ('cheeseburger', 933), ('teapot', 849), ('samoyed', 258),
    ('bald eagle', 22), ('red fox', 277), ('hotpot', 926),
    ('wombat', 106), ('timber wolf', 269),
)
MEDIA_KEYS = ('samples/fixed', *(f'samples/{name.replace(" ", "_")}' for name, _ in CLASSES))
SAMPLES_PER_CLASS = 64
OVERVIEW_PER_CLASS = 10
PREVIEW_IMAGES = len(CLASSES) * SAMPLES_PER_CLASS
OVERVIEW_IMAGES = len(CLASSES) * OVERVIEW_PER_CLASS


def preview_labels(device='cpu'):
    return torch.tensor([label for _, label in CLASSES], device=device).repeat_interleave(SAMPLES_PER_CLASS)


def save_panels(images, directory, step):
    """Keep the first ten of each class in the overview; show all 64 in its grid."""
    from PIL import Image, ImageDraw, ImageFont

    assert images.dtype == torch.uint8 and images.shape == (PREVIEW_IMAGES, 3, 256, 256)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 23)
    except OSError:
        font = ImageFont.load_default()
    overview = Image.new('RGB', (192 + OVERVIEW_PER_CLASS*256, len(CLASSES)*256), 'white')
    draw = ImageDraw.Draw(overview)
    paths = {'samples/fixed': directory/f'step{step:06d}-fixed.png'}
    for row, (name, _) in enumerate(CLASSES):
        draw.text((12, row*256 + 116), name, fill='black', font=font)
        grid = Image.new('RGB', (8*256, 8*256 + 40), 'white')
        ImageDraw.Draw(grid).text((12, 5), name, fill='black', font=font)
        for index in range(SAMPLES_PER_CLASS):
            tile = Image.fromarray(images[row*SAMPLES_PER_CLASS + index].permute(1, 2, 0).numpy())
            grid.paste(tile, ((index % 8)*256, 40 + (index // 8)*256))
            if index < OVERVIEW_PER_CLASS:
                overview.paste(tile, (192 + index*256, row*256))
        key = f'samples/{name.replace(" ", "_")}'
        paths[key] = directory/f'step{step:06d}-{name.replace(" ", "_")}.png'
        grid.save(paths[key])
    overview.save(paths['samples/fixed'])
    return paths


@torch.no_grad()
def designated_preview(profile, model, aux, device, rank, world, step, run, output):
    """Run the same ancestral sampler as FID, without advancing training RNG."""
    options = profile.recipe.preview
    indices = torch.arange(rank, PREVIEW_IMAGES, world)
    labels = preview_labels(device)[rank::world]
    was_training = model.training
    model.eval()
    started = time.monotonic()
    try:
        with torch.random.fork_rng(devices=[device.index]):
            torch.manual_seed(int(options.fixed_seed) + rank)
            torch.cuda.manual_seed(int(options.fixed_seed) + rank)
            codes = profile.sample(model, aux, len(labels), labels)
            # Transfer compact bytes, rather than four copies of float images.
            images = torch.cat([profile.decode(aux, part).float().mul(.5).add(.5)
                .clamp(0, 1).mul(255).add(.5).to(torch.uint8).cpu()
                for part in codes.split(8)])
            del codes
            gathered = [None]*world
            dist.all_gather_object(gathered, (indices, images))
            if rank == 0:
                ordered = torch.empty(PREVIEW_IMAGES, 3, 256, 256, dtype=torch.uint8)
                for positions, values in gathered:
                    ordered[positions] = values
                directory = Path(output)/('samples' if run is not None else 'smoke/samples')
                paths = save_panels(ordered, directory, step)
                metadata = dict(global_step=step, generated_images=PREVIEW_IMAGES, overview_images=OVERVIEW_IMAGES,
                    samples_per_class=SAMPLES_PER_CLASS, overview_per_class=OVERVIEW_PER_CLASS, grid_rows=8, grid_columns=8,
                    classes=[dict(name=name, imagenet_id=label) for name, label in CLASSES],
                    fixed_seed=int(options.fixed_seed), world_size=world,
                    sampler=sampling_metadata(profile.recipe.sampling),
                    filtering=sampling_caption(profile.recipe.sampling),
                    selection='first ten per class in overview; all 64 in grid; no rejection',
                    training_rng_unchanged=True, elapsed_seconds=time.monotonic()-started,
                    media={key:str(path) for key, path in paths.items()})
                (directory/f'step{step:06d}-panels.json').write_text(json.dumps(metadata, indent=2)+'\n')
                if run is not None:
                    import wandb
                    # Every media key belongs to this run and this history row.
                    values = {'samples/global_step':step}
                    for key, path in paths.items():
                        caption = (f'Step {step}; ' + sampling_caption(profile.recipe.sampling)
                            + ('10 per class, fixed seed.' if key == 'samples/fixed'
                               else f'{key.split("/")[-1].replace("_", " ")}; 64 samples, fixed seed.'))
                        values[key] = wandb.Image(str(path), caption=caption)
                    run.log(values)
    finally:
        model.train(was_training)
        torch.cuda.empty_cache()
