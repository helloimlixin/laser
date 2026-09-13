import numpy as np
import pytest
import torch

from src.audio_scaled_quantizer import (pack_integer_frames, unpack_integer_frames, rate_spec,
    fit_residual_centers, nearest_centers, sparse_to_joint)
from src.scaled_atom_rq import ScaledAtomRQ


@pytest.mark.parametrize('levels,rate,refinement', [(8,5.1,64),(16,5.4,16),(32,5.7,4)])
def test_real_payload_rate_without_per_frame_padding(levels, rate, refinement):
    spec=rate_spec(levels); assert spec['nominal_kbps']==rate
    full=rate_spec(levels,refine=True)
    assert full['bits_per_frame']==40 and full['refinement_entries']==refinement
    for setup in (spec,full):
        sizes=setup['vocab_sizes']
        codes=np.random.default_rng(4).integers(0,sizes,size=(151,len(sizes)),dtype=np.int64)
        codes[0]=np.array(sizes)-1;codes[1]=0
        payload=pack_integer_frames(codes,sizes)
        assert len(payload)==(151*setup['bits_per_frame']+7)//8
        np.testing.assert_array_equal(unpack_integer_frames(payload,sizes,151),codes)


def test_invalid_codes_truncation_and_nonzero_padding_are_rejected():
    sizes=[65537]*2
    with pytest.raises(ValueError): pack_integer_frames(np.array([[65537,0]]),sizes)
    blob=pack_integer_frames(np.array([[1,2]]),sizes)
    with pytest.raises(ValueError): unpack_integer_frames(blob[:-1],sizes,1)
    with pytest.raises(ValueError): unpack_integer_frames(blob[:-1]+bytes([blob[-1]|1]),sizes,1)
    with pytest.raises(ValueError): unpack_integer_frames(bytes([255])*5,sizes,1)


def test_scalar_omp_quantization_preserves_atoms_and_canonicalizes_zero():
    levels=torch.tensor([-2.,-1.,1.,2.]); dictionary=torch.eye(2)
    q=ScaledAtomRQ(dictionary,levels,depth=2)
    atoms=torch.tensor([[1,0]]); values=torch.tensor([[1.8,.1]])
    codes=sparse_to_joint(atoms,values,levels)
    assert codes.tolist()==[[8,0]]
    torch.testing.assert_close(q.embed(codes).sum(-2),torch.tensor([[0.,2.]]))


def test_refinement_leaves_zero_available_and_cannot_increase_latent_error():
    torch.manual_seed(41)
    train=torch.randn(300,5); heldout=torch.randn(50,5)
    centers=fit_residual_centers(train,16,iterations=5)
    assert centers[0].eq(0).all()
    torch.testing.assert_close(centers,fit_residual_centers(train,16,iterations=5))
    ids=nearest_centers(heldout,centers)
    assert ((heldout-centers[ids]).square().sum(1)<=heldout.square().sum(1)+1e-5).all()
