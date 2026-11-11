"""Evaluate all fitted vocabulary sizes on identical official validation images."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import yaml

ROOT = Path('/workspace/Projects/laser')
UPSTREAM = Path('/tmp/laser-rq24576-native-8h100-20261108/upstream')
sys.path.insert(0, str(ROOT))
os.environ['RQ_UPSTREAM_ROOT'] = str(UPSTREAM)
os.environ['TORCH_HOME'] = '/tmp/laser-rq24576-native-8h100-20261108/torch-cache'
from src import rq_upstream, rqvae_metrics as metrics
from src.models.laser_tokenizer_adapter import FrozenLaserTokenizer
from src.training.rq_imagenet_runtime import TarImageNet
from rqvae.img_datasets.transforms import create_transforms
from rqvae.models.rqvae.quantizations import RQBottleneck
from rqvae.utils.config import load_config


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def module_digest(module):
    value = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        value.update(name.encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(value, indent=2) + '\n')
    pending.replace(path)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world', type=int, default=8)
    parser.add_argument('--limit', type=int, default=50000)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--tag', default='validation50k')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    base, work, output = [Path(config[key]) for key in ['source_run_directory', 'work_directory', 'output_directory']]
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    torch.cuda.set_per_process_memory_fraction(config['evaluation']['memory_gib_per_gpu'] * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('highest')
    started = time.monotonic()
    checkpoint = base / 'inputs/stage1-model.pt'
    assert digest(checkpoint) == config['stage1_checkpoint_sha256']
    aux = FrozenLaserTokenizer(checkpoint, num_atoms=16384, coeff_vocab_size=2,
                               coeff_max=1., coeff_scales=[1.] * 4, sparsity_level=4).to(device)
    quantizers, books, book_hashes = {}, {}, {}
    for vocabulary in config['vocabularies']:
        record = json.loads((output / 'books' / f'T{vocabulary}.json').read_text())
        path = work / 'books' / f'T{vocabulary}.pt'
        assert digest(path) == record['sha256']
        vectors = torch.load(path, map_location='cpu', weights_only=True)
        assert vectors.shape == (4, vocabulary, 256) and torch.isfinite(vectors).all()
        assert torch.count_nonzero(vectors[:, 0]) == 0
        q = RQBottleneck(latent_shape=[8, 8, 256], code_shape=[8, 8, 4], n_embed=vocabulary,
                         decay=.99, shared_codebook=False, restart_unused_codes=False)
        for depth, table in enumerate(q.codebooks):
            table.weight[:-1].copy_(vectors[depth])
            table.weight[-1].zero_()
            table.embed_ema.copy_(vectors[depth])
            table.cluster_size_ema.fill_(1.)
        q.requires_grad_(False).eval().to(device)
        quantizers[vocabulary] = q
        books[vocabulary] = vectors
        book_hashes[str(vocabulary)] = record['sha256']
    assert all(not parameter.requires_grad for parameter in aux.parameters())
    before_cnn = module_digest(aux)
    before_books = {str(size): module_digest(q) for size, q in quantizers.items()}
    source_manifest = json.loads((output / 'cache-verification.json').read_text())
    assert before_cnn == source_manifest['source_cnn_digest']
    metrics.INCEPTION_PATH = UPSTREAM / 'rqvae/metrics/inception.py'
    network = metrics.OriginalRQVAEInception().to(device).eval().requires_grad_(False)
    settings = load_config(base / 'inputs/stage2-training-config.yaml')
    transforms = create_transforms(settings.dataset, split='val')
    data = TarImageNet(base / 'data', 'val', transforms, seed=config['seed'])
    assert len(data) == 50000
    assert np.array_equal(np.bincount(data.labels), np.full(1000, 50))
    indices = list(range(args.rank, args.limit, args.world))
    loader = DataLoader(Subset(data, indices), batch_size=args.batch_size or config['evaluation']['batch_size'],
                        num_workers=2, pin_memory=True, persistent_workers=True,
                        multiprocessing_context='spawn')
    directory = work / args.tag
    directory.mkdir(parents=True, exist_ok=True)
    names = ['real'] + [f'T{size}' for size in config['vocabularies']]
    features = {name: np.lib.format.open_memmap(directory / f'{name}-rank{args.rank}.npy', mode='w+', dtype=np.float32,
                                                shape=(len(indices), 2048)) for name in names}
    errors = torch.zeros(len(quantizers), dtype=torch.float64, device=device)
    seen = 0
    for images, _ in loader:
        images = images.to(device, non_blocking=True).float()
        latent = aux.quant_conv(aux.encoder(images)).permute(0, 2, 3, 1).contiguous()
        real, _ = network(images.mul(.5).add(.5).clamp(0, 1))
        features['real'][seen:seen + len(images)] = real.cpu().numpy()
        for index, (vocabulary, q) in enumerate(quantizers.items()):
            _, codes = q.quantize(latent)
            quantized = q.embed_code(codes)
            assert codes.shape[1:] == (8, 8, 4) and codes.min() >= 0 and codes.max() < vocabulary
            if seen == 0:
                lookup = books[vocabulary].to(device)
                expected = torch.stack([lookup[d][codes[..., d]] for d in range(4)], -2).sum(-2)
                torch.testing.assert_close(quantized, expected, rtol=0, atol=0)
                del lookup, expected
            errors[index] += (latent - quantized).square().sum().double()
            reconstructed = aux.decoder(aux.post_quant_conv(quantized.permute(0, 3, 1, 2).contiguous()))
            fake = reconstructed.float().mul(.5).add(.5).clamp(0, 1)
            value, _ = network(fake)
            features[f'T{vocabulary}'][seen:seen + len(images)] = value.cpu().numpy()
        seen += len(images)
        if seen % 128 == 0 or seen == len(indices):
            row = dict(rank=args.rank, images=seen, target=len(indices), seconds=time.monotonic() - started,
                       tag=args.tag, peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3)
            write(output / args.tag / f'progress-rank{args.rank}.json', row)
            print(json.dumps(row), flush=True)
    assert seen == len(indices)
    for value in features.values():
        value.flush()
    assert before_cnn == module_digest(aux)
    assert before_books == {str(size): module_digest(q) for size, q in quantizers.items()}
    write(output / args.tag / f'result-rank{args.rank}.json',
          dict(rank=args.rank, images=seen, image_indices=indices, vocabularies=config['vocabularies'],
               feature_directory=str(directory), codebook_hashes=book_hashes,
               source_cnn_digest=before_cnn, frozen_encoder_decoder_verified=True,
               frozen_vector_tables_verified=True, integer_lookup_parity_verified=True,
               latent_squared_error_by_vocabulary=errors.cpu().tolist(),
               preprocessing=str(transforms), seconds=time.monotonic() - started,
               peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3))
    print(json.dumps(dict(completed=True, rank=args.rank, images=seen)), flush=True)


if __name__ == '__main__':
    main()
