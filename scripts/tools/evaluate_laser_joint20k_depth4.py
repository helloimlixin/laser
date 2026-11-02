#!/usr/bin/env python3
"""Prepare a four-depth joint book and compare its matched validation rFID."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DEFAULT_SOURCE = ROOT / 'outputs/imagenet-laser-sharedjoint20k-depth5-codebook-20261018/sharedjoint20k-depth5-codebook.pt'
DEFAULT_STAGE1 = ROOT / 'outputs/imagenet-rfid421-combination-soft-20260927/assets/laser-tokenizer.pt'
DEFAULT_OUT = ROOT / 'outputs/imagenet-laser-sharedjoint20k-depth4-codebook-20261022'
DEFAULT_CACHE = Path('/dev/shm/laser-joint20k-depth4-20261022')


def digest(path, algorithm='sha256'):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def write(path, value):
    payload = json.dumps(dict(value, time=time.time()), indent=2, allow_nan=False) + '\n'
    # The mounted workspace may expose rename metadata before the object is
    # readable. Evaluation writes use local RAM storage; exports write directly.
    if str(path).startswith('/workspace/'):
        path.write_text(payload)
        return
    pending = path.with_name(path.name + f'.{os.getpid()}.pending')
    pending.write_text(payload)
    pending.replace(path)


def prepare(args):
    import torch
    from src.models.laser_joint_tokenizer import JointAtomCoefficientRQ
    torch.set_num_threads(4)
    source = torch.load(args.source, map_location='cpu', weights_only=True)
    if source['atom_ids'].shape != (5, 20481):
        raise ValueError('Expected the five-depth 20,481-entry source book')
    spec = dict(source)
    for key in ('atom_ids', 'coefficients'):
        spec[key] = source[key][:4].clone()
    # Old refinement metadata described a different ancestor. Keep provenance
    # explicit rather than inheriting claims about that ancestor's prefix.
    spec.pop('first_four_depths_exactly_unchanged', None)
    spec.pop('first_four_depths_unchanged', None)
    spec.update(residual_depth=4, depth_truncation_source_sha256=digest(args.source),
                depth_truncation_source=str(args.source), source_depth=5,
                source_prefix_preserved=True, refitted_for_four_depths=False)
    q5 = JointAtomCoefficientRQ(source['dictionary'], source['atom_ids'], source['coefficients'])
    q4 = JointAtomCoefficientRQ(spec['dictionary'], spec['atom_ids'], spec['coefficients'])
    if spec.get('shared_codebook_across_depths'):
        from src.training.laser_shared_joint_vocabulary import assert_shared_codewords
        assert_shared_codewords(q5)
        assert_shared_codewords(q4)
    signals = torch.randn(64, 256, generator=torch.Generator().manual_seed(26102220481))
    codes5 = q5.quantize(signals)['codes']
    result4 = q4.quantize(signals)
    assert torch.equal(codes5[:, :4], result4['codes'])
    torch.testing.assert_close(q5.embed(codes5)[:, :4].sum(-2), result4['quantized'],
                               rtol=1.e-5, atol=2.e-6)
    target = args.out / 'joint20k-depth4-codebook.pt'
    if target.exists():
        existing = torch.load(target, map_location='cpu', weights_only=True)
        assert existing.keys() == spec.keys()
        for key, value in spec.items():
            assert torch.equal(existing[key], value) if torch.is_tensor(value) else existing[key] == value
    else:
        if str(target).startswith('/workspace/'):
            torch.save(spec, target)
        else:
            pending = target.with_suffix('.pending.pt')
            torch.save(spec, pending)
            pending.replace(target)
    report = dict(prepared=True, depth=4, vocabulary=20481, atoms=8192,
        tokens_per_image=256, source_tokens_per_image=320, token_reduction_fraction=.2,
        codebook_path=str(target), codebook_sha256=digest(target),
        source_codebook_path=str(args.source), source_codebook_sha256=digest(args.source),
        source_stage1_sha256=spec['source_stage1_checkpoint_sha256'],
        physical_vectors_shared_across_depths=bool(spec.get('shared_codebook_across_depths', False)),
        dictionary_and_coefficients_unchanged=True, greedy_prefix_verified=True,
        encoder_decoder_frozen=True, refitted_for_four_depths=False,
        reconstruction_rfid_requires_measurement=True, stage2_training_launched=False)
    write(args.out / 'prepared.json', report)
    shutil.copyfile(__file__, args.out / Path(__file__).name)
    print(json.dumps(report), flush=True)


def cache(args):
    import numpy as np
    args.cache.mkdir(parents=True, exist_ok=True)
    for source, name, expected, algorithm in [
        (args.stage1, 'stage1.pt', json.loads((args.out / 'prepared.json').read_text())['source_stage1_sha256'], 'sha256'),
        (args.validation, 'ILSVRC2012_img_val.tar', '29b22e2961454d5413ddabcf34fc5622', 'md5'),
    ]:
        target = args.cache / name
        receipt_path = target.with_suffix('.verified.json')
        if target.exists() and receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            assert target.stat().st_size == receipt['bytes'] and receipt[algorithm] == expected
            continue
        print(json.dumps(dict(phase='copying_evaluation_asset', name=name, bytes=source.stat().st_size)), flush=True)
        pending = target.with_suffix('.partial')
        source_fd = os.open(source, os.O_RDONLY)
        target_fd = os.open(pending, os.O_CREAT | os.O_TRUNC | os.O_RDWR, 0o600)
        size = source.stat().st_size
        os.ftruncate(target_fd, size)

        def copy_segment(first):
            last = min(first + 64 * 1024**2, size)
            if name == 'ILSVRC2012_img_val.tar' and args.validation_url:
                import requests
                for attempt in range(3):
                    offset = first
                    try:
                        with requests.get(args.validation_url, headers={'Range': f'bytes={first}-{last - 1}'},
                                          stream=True, timeout=(15, 60)) as response:
                            response.raise_for_status()
                            if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {first}-{last - 1}/{size}':
                                raise ValueError('Server did not honor the requested byte range')
                            for data in response.iter_content(8 * 1024**2):
                                view = memoryview(data)
                                while view:
                                    count = os.pwrite(target_fd, view, offset)
                                    if count <= 0:
                                        raise OSError('Incomplete download write')
                                    offset += count
                                    view = view[count:]
                        if offset != last:
                            raise OSError('Incomplete HTTP range')
                        return
                    except (requests.RequestException, OSError):
                        if attempt == 2:
                            raise
            while first < last:
                data = os.pread(source_fd, min(8 * 1024**2, last - first), first)
                if not data:
                    raise OSError('Truncated evaluation asset')
                view = memoryview(data)
                while view:
                    count = os.pwrite(target_fd, view, first)
                    if count <= 0:
                        raise OSError('Incomplete evaluation asset write')
                    first += count
                    view = view[count:]

        try:
            with ThreadPoolExecutor(max_workers=16) as pool:
                list(pool.map(copy_segment, range(0, size, 64 * 1024**2)))
        finally:
            os.close(source_fd)
            os.close(target_fd)
        actual = digest(pending, algorithm)
        if actual != expected:
            raise ValueError(f'{name} checksum {actual}, expected {expected}')
        pending.replace(target)
        write(receipt_path, dict(verified=True, bytes=size, **{algorithm: actual}))
    index = args.cache / 'validation-index.npz'
    if not index.exists():
        with tarfile.open(args.cache / 'ILSVRC2012_img_val.tar', 'r:') as archive:
            members = sorted(archive.getmembers(), key=lambda m: m.name)
        assert len(members) == 50000 and all(m.isfile() and m.name.endswith('.JPEG') for m in members)
        np.savez(index, offsets=np.array([m.offset_data for m in members], dtype=np.int64),
                 sizes=np.array([m.size for m in members], dtype=np.int64))
    print(json.dumps(dict(phase='evaluation_assets_ready', cache=str(args.cache))), flush=True)


class ValidationImages:
    def __init__(self, cache_path):
        import numpy as np
        from torchvision import transforms
        self.archive = str(cache_path / 'ILSVRC2012_img_val.tar')
        with np.load(cache_path / 'validation-index.npz') as index:
            self.offsets, self.sizes = index['offsets'], index['sizes']
        self.descriptor = None
        # Exact upstream ImageNet validation transforms.
        self.transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256),
            transforms.Resize((256, 256)), transforms.ToTensor(),
            transforms.Normalize([.5] * 3, [.5] * 3)])

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        from PIL import Image, ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        if self.descriptor is None:
            self.descriptor = os.open(self.archive, os.O_RDONLY)
        data = os.pread(self.descriptor, int(self.sizes[index]), int(self.offsets[index]))
        if len(data) != int(self.sizes[index]):
            raise OSError(f'Incomplete JPEG at {index}')
        with Image.open(io.BytesIO(data)) as image:
            return self.transform(image.convert('RGB'))


def evaluate(args):
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader, Subset
    from src.models.laser_joint_tokenizer import FrozenJointLaserTokenizer, JointAtomCoefficientRQ
    from src.rqvae_metrics import DistributedOriginalRQVAEMetrics, OriginalRQVAEInception
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0)))
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if world > 1:
        dist.init_process_group('nccl', device_id=device, timeout=timedelta(minutes=30))
    run = None
    if rank == 0 and args.wandb:
        import wandb
        os.environ['WANDB_API_KEY'] = args.key_file.read_text().strip()
        os.environ['WANDB_BASE_URL'] = 'https://api.wandb.ai'
        run = wandb.init(entity='helloimlixin-rutgers', project='laser',
            id=args.wandb_run_id, name='LASER shared joint20k | four vs five depths | matched 50k rFID',
            job_type='tokenizer-evaluation', dir=str(args.cache), resume='allow',
            config=json.loads((args.out / 'prepared.json').read_text()))
        write(args.out / 'wandb.json', dict(url=run.url, run_id=run.id))
    try:
        aux = FrozenJointLaserTokenizer(args.cache / 'stage1.pt', args.source).to(device)
        state_versions = [v._version for v in aux.state_dict().values()]
        spec4 = torch.load(args.out / 'joint20k-depth4-codebook.pt', map_location='cpu', weights_only=True)
        q4 = JointAtomCoefficientRQ(spec4['dictionary'], spec4['atom_ids'], spec4['coefficients']).to(device)
        network = OriginalRQVAEInception().eval().requires_grad_(False).to(device)
        four = DistributedOriginalRQVAEMetrics(device, inception=network)
        five = DistributedOriginalRQVAEMetrics(device, inception=network)
        data = ValidationImages(args.cache)
        assert len(data) == 50000
        loader = DataLoader(Subset(data, range(rank, len(data), world)),
                            batch_size=args.batch_size, num_workers=2, pin_memory=True)
        count, started = 0, time.monotonic()
        with torch.inference_mode():
            for images in loader:
                images = images.to(device)
                latent = aux.encode(images)
                codes = aux.quantizer.quantize(latent, chunk_size=1024)['codes']
                vectors = aux.quantizer.embed(codes)
                z4, z5 = vectors[..., :4, :].sum(-2), vectors.sum(-2)
                if count == 0:
                    direct = q4.quantize(latent, chunk_size=1024)
                    assert torch.equal(codes[..., :4], direct['codes'])
                    torch.testing.assert_close(z4, direct['quantized'], rtol=1.e-4, atol=2.e-5)
                four.update(images.mul(.5).add(.5).clamp(0, 1), real=True)
                for metric, z in [(four, z4), (five, z5)]:
                    metric.update(aux.decode(z).mul(.5).add(.5).clamp(0, 1), real=False)
                count += len(images)
                if rank == 0:
                    progress = dict(phase='measuring_matched_validation_rfid',
                        images_on_rank=count, images_per_rank=len(range(rank, len(data), world)),
                        estimated_total_images=min(count * world, len(data)),
                        elapsed_seconds=time.monotonic() - started)
                    write(args.out / 'progress.json', progress)
                    if count % 256 == 0:
                        print(json.dumps(progress), flush=True)
        five.real_sum.copy_(four.real_sum)
        five.real_cross.copy_(four.real_cross)
        five.real_count.copy_(four.real_count)
        results = {}
        for depth, metric in [(4, four), (5, five)]:
            fid, _, _ = metric.compute()
            assert int(metric.real_count) == int(metric.fake_count) == 50000
            results[str(depth)] = dict(measured_rfid=fid, depth=depth,
                vocabulary=20481, tokens_per_image=64 * depth)
        assert state_versions == [v._version for v in aux.state_dict().values()]
        if rank == 0:
            report = dict(completed=True, results=results, real_images=50000,
                reconstructed_images_per_candidate=50000,
                reference='Full ImageNet validation; original RQ Inception; matched images and transforms',
                change_from_five_depths=results['4']['measured_rfid'] - results['5']['measured_rfid'],
                dictionary_encoder_decoder_frozen=True,
                physical_codewords_identical_at_every_depth=bool(spec4.get('shared_codebook_across_depths', False)),
                greedy_prefix_verified_on_real_images=True, stage2_training_launched=False,
                generative_fid_not_measured=True, elapsed_seconds=time.monotonic() - started)
            write(args.out / 'full-rfid.json', report)
            print(json.dumps(report), flush=True)
            if run:
                import wandb
                run.log({'rfid/depth4': results['4']['measured_rfid'],
                         'rfid/depth5': results['5']['measured_rfid'],
                         'rfid/change_from_depth5': report['change_from_five_depths']})
                artifact = wandb.Artifact(args.wandb_run_id, type='tokenizer')
                for name in ('joint20k-depth4-codebook.pt', 'prepared.json', 'full-rfid.json', Path(__file__).name):
                    artifact.add_file(str(args.out / name))
                run.log_artifact(artifact).wait()
    finally:
        if run:
            run.finish(exit_code=0 if (args.out / 'full-rfid.json').exists() else 1)
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'cache', 'evaluate'))
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--stage1', type=Path, default=DEFAULT_STAGE1)
    parser.add_argument('--validation', type=Path, default=Path('/workspace/Projects/data/imagenet/archives/ILSVRC2012_img_val.tar'))
    parser.add_argument('--validation-url', help='Download and verify the official archive instead of copying the saved archive')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--cache', type=Path, default=DEFAULT_CACHE)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb-run-id', default=DEFAULT_OUT.name)
    parser.add_argument('--key-file', type=Path, default=Path('/root/.config/laser/imagenet-stage2-wandb.key'))
    args = parser.parse_args()
    os.environ.setdefault('TORCH_HOME', str(args.cache / 'torch-cache'))
    args.out.mkdir(parents=True, exist_ok=True)
    {'prepare': prepare, 'cache': cache, 'evaluate': evaluate}[args.phase](args)


if __name__ == '__main__':
    main()
