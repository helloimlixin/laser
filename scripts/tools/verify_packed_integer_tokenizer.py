#!/usr/bin/env python3
"""Compare packed IDs to the current compound512 codec on real images."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    import torch
    from omegaconf import OmegaConf
    from src import rq_upstream
    from rqvae.img_datasets.transforms import create_transforms
    from src.models.laser_tokenizer_adapter import FrozenLaserTokenizer
    from src.training.laser_packed_integer_recipe import PackedIntegerLaserRecipe
    from src.training.laser_compound_replay_recipe import CompoundReplayRecipe
    from src.training.rq_imagenet_runtime import TarImageNet

    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('/tmp/laser-compound512-origlr-imagenet-20261016-r3'))
    parser.add_argument('--recipe',type=Path,default=ROOT/'configs/stage2/imagenet-laser-packed-pair512-proj128-scratch2-20261017.yaml')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(4);torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(3*2**30/torch.cuda.get_device_properties(0).total_memory,0)
    with tempfile.TemporaryDirectory(prefix='laser-pairid-tokenizer-') as folder:
        base=Path(folder);(base/'inputs').mkdir()
        (base/'inputs/recipe.yaml').symlink_to(args.recipe.resolve())
        (base/'inputs/stage1-model.pt').symlink_to(args.source/'inputs/stage1-model.pt')
        profile=PackedIntegerLaserRecipe(base)
        aux,_=profile.tokenizer(torch.device('cuda',0))
        before=profile.digest(aux)
        old=CompoundReplayRecipe(args.source)
        config=OmegaConf.load(args.source/'inputs/stage2-training-config.yaml')
        dataset=TarImageNet(args.source/'data','val',create_transforms(config.dataset,split='val'))
        indices=[0,933,849,258]
        maximum_reconstruction_error=0.
        for index in indices:
            image=dataset[index][0][None].cuda()
            torch.manual_seed(261017514+index)
            targets,codes=profile.targets(aux,image)
            aux._compound_gram=aux._packed_gram
            torch.manual_seed(261017514+index)
            previous,scalar=old.targets(aux,image)
            expected=scalar[...,0::2]*512+scalar[...,1::2]-16384
            assert torch.equal(codes,expected)
            assert torch.equal(targets['atoms'],previous['atoms'])
            torch.testing.assert_close(targets['coefficient_probabilities'],
                previous['coefficient_probabilities'],rtol=0,atol=0)
            with torch.no_grad():
                actual_image=aux.decode_code(codes)
                expected_image=FrozenLaserTokenizer.decode_tokens(aux,scalar)
            maximum_reconstruction_error=max(maximum_reconstruction_error,
                float((actual_image-expected_image).abs().max()))
            torch.testing.assert_close(actual_image,expected_image,rtol=0,atol=0)
        assert profile.digest(aux)==before
        with (args.source/'inputs/stage1-model.pt').open('rb') as stream:
            source_sha=hashlib.file_digest(stream,'sha256').hexdigest()
        assert source_sha==profile.recipe.tokenizer.sha256
        result=dict(passed=True,real_validation_images=len(indices),image_indices=indices,
            exact_packed_code_identity=True,exact_soft_target_identity=True,
            maximum_decoded_image_difference=maximum_reconstruction_error,
            frozen_stage1_unchanged=True,stage1_checkpoint_sha256=source_sha,
            all_16384_atoms_preserved=True,coefficient_bins=512,
            coefficient_coordinates='original source physical coefficients',
            new_trajectory_targets=False,new_rfid_measurement=False,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            training_updates_added=0)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
