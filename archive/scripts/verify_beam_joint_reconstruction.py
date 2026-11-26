#!/usr/bin/env python3
"""Reconstruct real images with the exact joint16k tokenizer used by the FID14.1634 run."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def digest_file(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value, indent=2) + '\n'); temporary.replace(path)


def module_digest(module):
    result = hashlib.sha256()
    for name, value in module.state_dict().items():
        result.update(name.encode()); result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def preview(path, examples):
    """A labeled scientific comparison; all columns use identical image identities."""
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    names = ['Original image', 'Original OMP', 'Joint greedy', 'Joint beam16', 'Native soft T=0.5']
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    canvas = Image.new('RGB', (5*260, 48+len(examples)*284), 'white')
    draw = ImageDraw.Draw(canvas)
    for column, name in enumerate(names): draw.text((column*260+5, 12), name, fill='black', font=font)
    for row, item in enumerate(examples):
        draw.text((5, 48+row*284), f"{item['class_name']}; {item['image_identity']}", fill='black', font=font)
        for column, pixels in enumerate(item['images']):
            image = Image.fromarray((pixels.permute(1,2,0).clamp(0,1).numpy()*255).round().astype(np.uint8))
            canvas.paste(image, (column*260, 48+row*284+25))
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--indices', type=Path)
    parser.add_argument('--split', choices=['train', 'val'], default='train')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=50000)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--wandb', action='store_true')
    args = parser.parse_args()
    base, output = args.base.resolve(), args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    os.environ['RQ_UPSTREAM_ROOT'] = str(base / 'upstream')
    os.environ['TORCH_HOME'] = str(base / 'torch-cache')
    sys.path[:0] = [str(base/'source'), str(base/'upstream'), str(base)]
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader, Subset
    from omegaconf import OmegaConf
    from src import rqvae_metrics as metrics
    from src.models.rfid_preserved_joint_tokenizer import RFIDPreservedJointTokenizer
    from src.training.laser_hard_ar import sparse_encoding
    from src.training.rq_imagenet_runtime import TarImageNet
    from rqvae.img_datasets.transforms import create_transforms
    from rqvae.utils.config import load_config
    rank, world, local = [int(os.environ.get(k,'0' if k != 'WORLD_SIZE' else '1'))
                          for k in ['RANK','WORLD_SIZE','LOCAL_RANK']]
    device = torch.device('cuda', local); torch.cuda.set_device(device); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('highest')
    if world > 1: dist.init_process_group('nccl', device_id=device)
    torch.manual_seed(2611071416+rank); torch.cuda.manual_seed(2611071416+rank)
    recipe = OmegaConf.load(base/'inputs/recipe.yaml')
    expected = {'stage1-model.pt': recipe.tokenizer.sha256,
        'joint-codebook.pt': recipe.tokenizer.codebook_sha256,
        'reconstruction-codebook.pt': recipe.tokenizer.reconstruction_codebook_sha256}
    hashes = {name:digest_file(base/'inputs'/name) for name in expected}
    if hashes != expected: raise ValueError('Recovered tokenizer hash differs from the FID14.1634 recipe')
    config = load_config(base/'inputs/stage2-training-config.yaml')
    dataset = TarImageNet(args.data,args.split,create_transforms(config.dataset,split='val'))
    if args.indices is None:
        if args.split != 'val': parser.error('--indices is required for training-subset evaluation')
        indices = np.arange(len(dataset),dtype=np.int64)[:args.limit]
    else:
        indices = np.load(args.indices)[:args.limit]
    if len(indices) != args.limit or len(np.unique(indices)) != len(indices):
        raise ValueError('Reconstruction requires the requested number of distinct real image identities')
    if indices.min() < 0 or indices.max() >= len(dataset):
        raise ValueError('Image identity is outside the requested dataset split')
    if len(indices) == 50000 and not np.array_equal(np.bincount(dataset.labels[indices],minlength=1000),np.full(1000,50)):
        raise ValueError('The 50k reconstruction sample must contain fifty images from every class')
    data_receipt = None
    if args.split == 'val':
        data_receipt = json.loads((args.data/'verified.json').read_text())
        if (len(dataset) != 50000 or not data_receipt.get('official_md5_verified')
                or data_receipt.get('md5') != '29b22e2961454d5413ddabcf34fc5622'
                or data_receipt.get('canonical_image_id_count') != 50000
                or data_receipt.get('index_sha256') != digest_file(args.data/'val-index.npz')):
            raise ValueError('Validation requires the verified original ImageNet archive and canonical index')
        if len(indices) == 50000 and not np.array_equal(np.sort(indices),np.arange(50000)):
            raise ValueError('Full validation must cover every canonical image identity exactly once')
    complete_validation = args.split == 'val' and len(indices) == 50000
    split_description = 'ImageNet 2012 validation' if args.split == 'val' else 'class-balanced ImageNet training subset'
    if rank == 0: np.save(output/'indices.npy',indices)
    if world > 1: dist.barrier()
    indices_hash = digest_file(output/'indices.npy')
    run = None
    if rank == 0 and args.wandb:
        os.environ.update(WANDB_API_KEY=Path('/root/.config/laser/wandb.key').read_text().strip(),
            WANDB_BASE_URL='https://api.wandb.ai',WANDB_DISABLE_GIT='true',WANDB_CONSOLE='off')
        import wandb
        run = wandb.init(entity='helloimlixin-rutgers',project='laser',
            id=output.name+'-beam-reconstruction-20261107',
            name=f'Beam16 joint16k | {split_description} | recovered FID14.1634 tokenizer',
            dir=str(output),config=dict(source_run=str(recipe.run_id),hashes=hashes,images=len(indices),
                split=split_description,validation_rfid_remeasured=complete_validation,ar_training_updates=0),
            settings=wandb.Settings(console='off',disable_git=True,x_disable_stats=True))
        write_json(output/'wandb.json',dict(url=run.url,id=run.id))
    aux = RFIDPreservedJointTokenizer(base/'inputs/stage1-model.pt',base/'inputs/joint-codebook.pt',
        base/'inputs/reconstruction-codebook.pt').to(device).eval().requires_grad_(False)
    before = module_digest(aux)
    loader = DataLoader(Subset(dataset,indices[rank::world].tolist()),batch_size=args.batch_size,
        num_workers=4,pin_memory=True,prefetch_factor=2)
    metrics.INCEPTION_PATH = base/'upstream/rqvae/metrics/inception.py'
    inception = metrics.OriginalRQVAEInception().to(device).eval().requires_grad_(False)
    variants = ['original_omp','joint_greedy','joint_beam16','native_soft05']
    accum = {name:metrics.DistributedOriginalRQVAEMetrics(device,inception=inception) for name in variants}
    sums = torch.zeros(len(variants),4,dtype=torch.float64,device=device)
    native_greedy_differences = torch.zeros((),dtype=torch.long,device=device)
    beam_margin = torch.tensor(float('inf'),dtype=torch.float64,device=device)
    source_dictionary = aux.backbone.dictionary.float()
    gram = (source_dictionary.T @ source_dictionary).contiguous()
    classes = [(933,'cheeseburger'),(849,'teapot'),(258,'samoyed'),(22,'bald eagle'),
               (277,'red fox'),(926,'hotpot'),(106,'wombat'),(269,'timber wolf')]
    examples = []; selected = []
    if rank == 0:
        for label, name in classes:
            found = indices[dataset.labels[indices] == label]
            if len(found): selected.append((int(found[0]),name))
    seen = 0; started = time.monotonic(); parity = None

    @torch.inference_mode()
    def reconstruct(images, audit=False):
        latent = aux.encode(images.float())
        greedy = aux.physical_book._quantize_greedy(latent,chunk_size=256)
        beam = aux.quantize_reconstruction(latent)
        native_latent, native_codes = aux.quantizer.quantize(latent)
        native_greedy_differences.add_((native_codes != greedy['codes']).sum())
        probabilities, soft_codes = aux.quantizer.get_soft_codes(latent,temp=.5,stochastic=True)
        if audit:
            torch.testing.assert_close(probabilities.sum(-1),torch.ones_like(probabilities[...,0]),rtol=2.e-5,atol=2.e-5)
        del probabilities
        atoms, coeff = sparse_encoding(latent,source_dictionary,4,gram=gram)
        original = (source_dictionary.T[atoms]*coeff[...,None]).sum(-2)
        greedy_error = (latent-greedy['quantized']).square().sum(-1)
        beam_error = (latent-beam['quantized']).square().sum(-1)
        margin = greedy_error-beam_error
        beam_margin.copy_(torch.minimum(beam_margin,margin.min().double()))
        if not torch.all(beam_error <= greedy_error + 1.e-5):
            raise ValueError('Beam16 reconstruction lost the greedy anchor guarantee')
        native_beam = aux.quantizer.embed_code(beam['codes'])
        torch.testing.assert_close(native_beam,beam['quantized'],rtol=1.e-6,atol=1.e-6)
        decoded = [aux.decode(original),aux.decode_code(greedy['codes']),
                   aux.decode_code(beam['codes']),aux.decode_code(soft_codes)]
        latents = [original,aux.quantizer.embed_code(greedy['codes']),native_beam,aux.quantizer.embed_code(soft_codes)]
        if any(not value.isfinite().all() for value in decoded): raise ValueError('Nonfinite reconstruction')
        if audit:
            for codes in [greedy['codes'],beam['codes'],soft_codes]:
                if codes.dtype != torch.long or codes.shape[1:] != (8,8,4) or codes.min()<0 or codes.max()>=16384:
                    raise ValueError('Invalid joint-ID encoding')
        return [value.float().mul(.5).add(.5).clamp(0,1) for value in decoded], latents, latent

    with torch.inference_mode():
        if selected:
            for index, name in selected:
                pixels, _ = dataset[index]; images = pixels[None].to(device)
                decoded,_,_ = reconstruct(images,audit=True)
                identity = f'ILSVRC2012_val_{index+1:08d}.JPEG' if args.split == 'val' else f'train image {index}'
                examples.append(dict(index=index,class_name=name,image_identity=identity,
                    images=[images[0].mul(.5).add(.5).cpu(),*[d[0].cpu() for d in decoded]]))
            preview(output/'reconstruction-comparison.png',examples)
            if run:
                run.log({'comparison/reconstructions':wandb.Image(str(output/'reconstruction-comparison.png'),
                    caption='Columns: real image, original OMP, joint greedy, joint beam16, native stochastic teacher T0.5')})
            write_json(output/'preview-identities.json',dict(columns=['original','original_omp','joint_greedy','joint_beam16','native_soft05'],
                split=split_description,
                images=[dict(index=e['index'],class_name=e['class_name'],image_identity=e['image_identity']) for e in examples],
                selection='first sampled image of each predeclared preview class; identical pixels in every column'))
        native_greedy_differences.zero_();beam_margin.fill_(float('inf'))
        for images,_ in loader:
            images = images.to(device,non_blocking=True)
            decoded,latents,latent = reconstruct(images,audit=seen==0)
            if seen == 0:
                parity = dict(passed=True,hashes_match=True,same_physical_teacher_and_reconstruction_tables=True,
                    fixed_integer_products=True,per_image_coefficient_refitting=False,
                    native_and_physical_beam_decoding_match=True,valid_joint_ids=True,
                    beam_width=16,ar_training_updates=0,encoder_decoder_frozen=True)
                write_json(output/f'parity-rank{rank}.json',parity)
            real = images.mul(.5).add(.5).clamp(0,1)
            features,_ = inception(real); features = features.double()
            total, cross = features.sum(0), features.T @ features
            for column,name in enumerate(variants):
                metric = accum[name]; fake = decoded[column]
                metric.real_sum.add_(total);metric.real_cross.add_(cross);metric.real_count.add_(len(images))
                metric.update(fake,real=False)
                pixel_mse = (fake-real).square().flatten(1).mean(-1)
                sums[column,0] += pixel_mse.double().sum()
                sums[column,1] += (-10*pixel_mse.clamp_min(1.e-20).log10()).double().sum()
                sums[column,2] += (latents[column]-latent).square().flatten(1).mean(-1).double().sum()
                sums[column,3] += len(images)
            seen += len(images)
            if rank == 0 and (seen == len(images) or seen % 200 == 0):
                row = dict(phase='reconstructing',images_on_rank=seen,target_on_rank=len(indices[rank::world]),
                    seconds=time.monotonic()-started,time=time.time())
                write_json(output/'progress.json',row);print(json.dumps(row),flush=True)
                if run:run.log({'reconstruction/images_on_rank':seen,
                    'reconstruction/progress_fraction':seen/len(indices[rank::world]),
                    'reconstruction/seconds':row['seconds']})
        if world > 1:
            dist.all_reduce(sums);dist.all_reduce(native_greedy_differences)
            dist.all_reduce(beam_margin,op=dist.ReduceOp.MIN)
        results = {}
        for column,name in enumerate(variants):
            # Small pilots verify the codec; a 2048-dimensional FID from 64 images is not useful.
            fid = accum[name].compute()[0] if len(indices) > 2048 else None
            count = float(sums[column,3])
            results[name] = dict(rfid=fid,images=int(count),pixel_mse=float(sums[column,0]/count),
                mean_psnr_db=float(sums[column,1]/count),latent_mse=float(sums[column,2]/count))
        if module_digest(aux) != before: raise ValueError('Verification changed the frozen tokenizer')
        if rank == 0:
            report = dict(passed=True,source_run=str(recipe.run_id),source_best_generation_fid=14.1633995743926,
                split=split_description,images=len(indices),results=results,
                validation_rfid_remeasured=complete_validation,previous_reported_validation_beam_rfid=4.5122860099749005,
                limitation=None if complete_validation else 'This is a subset evaluation, not full ImageNet validation.',
                data_verification=data_receipt,preprocessing=str(dataset.transform),
                hashes=hashes,indices_sha256=indices_hash,frozen_tokenizer_state_sha256=before,
                frozen_tokenizer_unchanged=True,encoding_and_decoding_parity=parity,
                minimum_latent_squared_error_improvement_over_greedy=float(beam_margin),
                physical_vs_native_greedy_token_differences=int(native_greedy_differences),
                teacher_temperature=.5,teacher_seed_base=2611071416,source_encoder_precision='fp32',
                metric='original RQ-VAE TensorFlow-FID-compatible Inception, paired real/reconstruction statistics',
                ar_training_updates=0,seconds=time.monotonic()-started)
            write_json(output/'results.json',report);write_json(output/'progress.json',dict(phase='completed',images=len(indices),time=time.time()))
            print(json.dumps(report),flush=True)
            if run:
                values={f'reconstruction/{name}/{key}':value for name,row in results.items() for key,value in row.items()}
                run.log(values);run.summary.update(dict(ar_training_updates=0,verification_passed=True,
                    validation_rfid_remeasured=complete_validation,split=split_description,images=len(indices)))
                run.save(str(output/'results.json'),base_path=str(output),policy='now');run.finish()
    if world > 1: dist.destroy_process_group()


if __name__ == '__main__': main()
