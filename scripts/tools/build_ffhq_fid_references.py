"""Audit FFHQ resize/split effects and build references from exact training images."""
import argparse
from datetime import timedelta
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image, __version__ as pillow_version
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
from torchvision.transforms.functional import to_tensor

from src.original_rq_training import FeatureMoments, atomic_json, file_sha256
from src.training.var_laser import get_inception_model, frechet_distance


def read_ids(path, expected_count):
    values = np.array([int(Path(line.strip()).stem) for line in Path(path).read_text().splitlines() if line.strip()], dtype=np.int64)
    if len(values) != expected_count or len(np.unique(values)) != expected_count:
        raise ValueError('Reference image IDs must be complete and unique')
    if np.any(values < 0) or np.any(values >= 70000):
        raise ValueError('FFHQ image ID out of range')
    return np.sort(values)


def real_pixels(image):
    if image.mode != 'RGB' or image.size != (256, 256):
        raise ValueError('FID reference requires 256x256 RGB images')
    # Match the released dataset normalization followed by FID unnormalization.
    return to_tensor(image).sub(.5).div(.5).mul(.5).add(.5).clamp(0, 1)


class CachedFFHQ(Dataset):
    def __init__(self, root):
        from datasets import concatenate_datasets, load_from_disk
        splits = load_from_disk(str(Path(root) / 'hf'))
        self.fingerprints = {name: data._fingerprint for name, data in splits.items()}
        if len(splits['train']) != 60000 or len(splits['validation']) != 10000:
            raise ValueError('Incomplete FFHQ cache')
        if set(splits['train']['image_id']) != set(range(60000)):
            raise ValueError('Training IDs differ from the documented official split')
        if set(splits['validation']['image_id']) != set(range(60000, 70000)):
            raise ValueError('Validation IDs differ from the documented official split')
        self.data = concatenate_datasets([splits['train'], splits['validation']])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[int(index)]
        return real_pixels(row['image']), int(row['image_id'])


class BilinearFFHQ(Dataset):
    def __init__(self, archive, metadata):
        self.archive = str(archive)
        data = json.loads(Path(metadata).read_text())
        self.digests = [data[str(i)]['image']['file_md5'] for i in range(70000)]
        with zipfile.ZipFile(self.archive) as source:
            names = {int(Path(name).stem): name for name in source.namelist() if name.lower().endswith('.png')}
        if set(names) != set(range(70000)):
            raise ValueError('Raw FFHQ archive IDs are incomplete')
        self.names = [names[i] for i in range(70000)]
        self.source = None

    def __len__(self):
        return 70000

    def __getitem__(self, index):
        if self.source is None:
            self.source = zipfile.ZipFile(self.archive)
        raw = self.source.read(self.names[index])
        if hashlib.md5(raw).hexdigest() != self.digests[index]:
            raise ValueError(f'Original FFHQ MD5 mismatch for image {index}')
        with Image.open(io.BytesIO(raw)) as image:
            if image.size != (1024, 1024):
                raise ValueError('Raw FFHQ image resolution mismatch')
            # Published RQ evaluation: PIL Resize(256, BILINEAR), CenterCrop(256).
            image = image.convert('RGB').resize((256, 256), Image.Resampling.BILINEAR)
        return real_pixels(image), index


@torch.inference_mode()
def extract(dataset, variant, model, args, rq_ids, provenance):
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    positions = list(range(rank, len(dataset), world))
    loader = DataLoader(Subset(dataset, positions), batch_size=args.batch_size,
                        num_workers=args.workers, pin_memory=True, shuffle=False)
    moments = {name: FeatureMoments(device) for name in ('laser', 'rq')}
    rq_mask = torch.zeros(70000, dtype=torch.bool, device=device)
    rq_mask[torch.from_numpy(rq_ids).to(device)] = True
    features_path = args.output / f'{variant}-features-rank{rank}.npy'
    ids_path = args.output / f'{variant}-ids-rank{rank}.npy'
    saved = np.lib.format.open_memmap(features_path, mode='w+', dtype=np.float32, shape=(len(positions), 2048))
    saved_ids = np.lib.format.open_memmap(ids_path, mode='w+', dtype=np.int64, shape=(len(positions),))
    offset, started = 0, time.monotonic()
    for batch, (pixels, ids) in enumerate(loader):
        pixels, gpu_ids = pixels.to(device), ids.to(device)
        features = model(pixels)
        for name, mask in [('laser', gpu_ids < 60000), ('rq', rq_mask[gpu_ids])]:
            if mask.any():
                moments[name].update(features[mask])
        saved[offset:offset+len(ids)] = features.cpu().numpy()
        saved_ids[offset:offset+len(ids)] = ids.numpy()
        offset += len(ids)
        if rank == 0 and batch % 20 == 0:
            status = dict(phase='reference_features', variant=variant,
                          images=min(offset * world, 70000), count=70000, time=time.time())
            atomic_json(args.output / 'status.json', status)
            print(json.dumps(status), flush=True)
    saved.flush(); saved_ids.flush()
    del saved, saved_ids
    if offset != len(positions):
        raise RuntimeError('Incomplete reference feature extraction')
    dist.barrier()
    if rank == 0:
        observed = np.concatenate([np.load(args.output / f'{variant}-ids-rank{i}.npy') for i in range(world)])
        if len(observed) != 70000 or not np.array_equal(np.sort(observed), np.arange(70000)):
            raise RuntimeError('Duplicate or missing reference images')
    for split, state in moments.items():
        count, mu, sigma = state.finish()
        if count != 60000:
            raise RuntimeError('FID reference must contain the entire 60000-image training set')
        if rank == 0:
            name = f'{split}-train-{variant}'
            stats_path = args.output / (name + '.npz')
            np.savez(stats_path, mu=mu, sigma=sigma)
            record = dict(name=name, count=count, statistics_sha256=file_sha256(stats_path),
                image_ids_sha256=file_sha256(args.output / f'{split}-train-ids.npy'),
                source_preprocessing='Actual lossless cached RGB 256x256 images; original downsampling PIL LANCZOS' if variant == 'lanczos' else
                                     'Original verified RGB 1024x1024 images; PIL BILINEAR resize to 256x256',
                uses_exact_laser_training_images=split == 'laser' and variant == 'lanczos',
                elapsed_seconds=time.monotonic()-started, **provenance)
            atomic_json(stats_path.with_suffix('.json'), record)
    dist.barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('/tmp/laser-ffhq256-data'))
    parser.add_argument('--metadata', type=Path, default=Path('/workspace/Projects/data/raw/ffhq_dataset/ffhq-dataset-v2.json'))
    parser.add_argument('--rq-ids', type=Path, default=Path('/workspace/Projects/laser/third_party/rq-vae-transformer/rqvae/img_datasets/assets/ffhqtrain.txt'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if (args.output / 'complete.json').exists():
        raise FileExistsError('Completed FID references are immutable; choose a new output directory')
    torch.set_num_threads(4)
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group('nccl', timeout=timedelta(minutes=30))
    rank = dist.get_rank()
    args.output.mkdir(parents=True, exist_ok=True)
    rq_ids = read_ids(args.rq_ids, 60000)
    cached = CachedFFHQ(args.data)
    data_manifest = json.loads((args.data / 'manifest.json').read_text())
    if rank == 0:
        for name, expected in data_manifest['files'].items():
            if file_sha256(args.data / name) != expected['sha256']:
                raise ValueError(f'Training image cache checksum mismatch: {name}')
        if file_sha256(args.metadata) != data_manifest['source_metadata_sha256']:
            raise ValueError('Original FFHQ metadata checksum mismatch')
        np.save(args.output / 'laser-train-ids.npy', np.arange(60000, dtype=np.int64))
        np.save(args.output / 'rq-train-ids.npy', rq_ids)
    dist.barrier()
    root = Path(__file__).resolve().parents[2]
    weights = Path(os.environ['TORCH_HOME']) / 'hub/checkpoints/pt_inception-2015-12-05-6726825d.pth'
    provenance = dict(protocol_version='ffhq-fid-fp32-continuous-v1',
        real_pixel_processing='uint8 RGB -> float32 [-1,1] -> clamp((x+1)/2,0,1); no random augmentation',
        generated_pixel_processing='FP32 decoder -> clamp((x+1)/2,0,1); no image-file roundtrip',
        inception_processing='bilinear 299x299, align_corners=False, antialias=False; 2*x-1; FID Inception 2048D',
        inception_weights_sha256=file_sha256(weights),
        inception_source_sha256=file_sha256(root/'third_party/rq-vae-transformer/rqvae/metrics/inception.py'),
        dataset_manifest_sha256=file_sha256(args.data/'manifest.json'),
        dataset_fingerprints=cached.fingerprints, cache_files_verified=True,
        torch_version=str(torch.__version__), torchvision_version=str(torchvision.__version__), pillow_version=pillow_version)
    model = get_inception_model().eval().requires_grad_(False).cuda()
    for variant, dataset in [('lanczos', cached), ('bilinear', BilinearFFHQ(data_manifest['source_archive'], args.metadata))]:
        extract(dataset, variant, model, args, rq_ids, provenance)
    if rank == 0:
        with np.load('/tmp/laser-rqvae-reference/ffhq_256_train.npz') as f:
            published = {k:f[k] for k in ('mu','sigma')}
        references = {}
        for split in ('laser','rq'):
            for variant in ('lanczos','bilinear'):
                name = f'{split}-train-{variant}'
                with np.load(args.output / (name+'.npz')) as f:
                    references[name] = {k:f[k] for k in ('mu','sigma')}
        distances = {}
        for name, stats in references.items():
            distances[name + '_vs_published'] = float(frechet_distance(stats['mu'],stats['sigma'],published['mu'],published['sigma']))
        for split in ('laser','rq'):
            a,b = references[f'{split}-train-lanczos'],references[f'{split}-train-bilinear']
            distances[split + '_resize_only'] = float(frechet_distance(a['mu'],a['sigma'],b['mu'],b['sigma']))
        for variant in ('lanczos','bilinear'):
            a,b = references[f'laser-train-{variant}'],references[f'rq-train-{variant}']
            distances[variant + '_split_only'] = float(frechet_distance(a['mu'],a['sigma'],b['mu'],b['sigma']))
        atomic_json(args.output / 'reference-comparison.json', dict(distances=distances,
            interpretation='Real-versus-real preprocessing/split audit; these FID distances are not additive corrections to generated FID.'))
        atomic_json(args.output / 'complete.json', dict(time=time.time(),references=list(references),images_per_variant=70000))
        print(json.dumps(distances), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
