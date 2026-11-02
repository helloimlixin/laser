import torch
from torch.nn import functional as F

from src.training.laser_latent_decoder_calibration import (
    latent_regression_statistics,select_center_statistics,solve_latent_correction,correct_latent,
)


def test_decoder_correction_recovers_spatial_signal_on_unseen_latents():
    torch.manual_seed(26101749)
    z=torch.randn(96,2,4,4)
    weight=torch.randn(2,2,3,3)*.08;bias=torch.tensor([.1,-.2])
    target=z+F.conv2d(z,weight,bias,padding=1)
    stats=latent_regression_statistics(z,target,images_per_chunk=7)
    adapter=solve_latent_correction(stats,ridge_fraction=1.e-5)
    held=torch.randn(12,2,4,4);expected=held+F.conv2d(held,weight,bias,padding=1)
    torch.testing.assert_close(correct_latent(held,adapter),expected,rtol=1.e-4,atol=1.e-4)


def test_center_statistics_match_direct_one_by_one_regression():
    torch.manual_seed(26101750)
    z=torch.randn(21,3,4,4);target=z+torch.randn_like(z)*.1
    direct=latent_regression_statistics(z,target,kernel=1,images_per_chunk=5)
    center=select_center_statistics(latent_regression_statistics(z,target,images_per_chunk=5))
    for key in ['gram','rhs','feature_sum','residual_sum']:
        torch.testing.assert_close(center[key],direct[key],rtol=1.e-12,atol=1.e-12)
    a,b=solve_latent_correction(center),solve_latent_correction(direct)
    torch.testing.assert_close(a['weight'],b['weight']);torch.testing.assert_close(a['bias'],b['bias'])
