"""Adapt the recovered FFHQ compound model to the frozen Church K=4 cache.

Set LASER_RECOVERED_RUNTIME to a copy of recovered/ffhqcmp0804205803.
The original trainer, backbone, coefficient head and objective remain intact.
"""
import importlib.util
import os
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

RUNTIME = Path(os.environ['LASER_RECOVERED_RUNTIME']).resolve()
sys.path.insert(0, str(RUNTIME))
spec = importlib.util.spec_from_file_location(
    'recovered_ffhq', RUNTIME/'scripts/train_official_rqtransformer_laser_stage2.py')
ffhq = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ffhq
spec.loader.exec_module(ffhq)


class ChurchAux(ffhq.LaserAux):
    def __init__(self, checkpoint, coeff_scales, *, num_atoms=16384):
        nn.Module.__init__(self)
        self.sparsity_level = len(coeff_scales)
        stage1 = ffhq.RQVAE(
            embed_dim=256, n_embed=num_atoms, decay=.99, loss_type='mse',
            latent_loss_weight=.25, bottleneck_type='rq',
            ddconfig=dict(double_z=False,z_channels=256,resolution=256,in_channels=3,
                out_ch=3,ch=128,ch_mult=[1,1,2,2,4,4],num_res_blocks=2,
                attn_resolutions=[8],dropout=0.),
            latent_shape=[8,8,256],code_shape=[8,8,self.sparsity_level],
            shared_codebook=True,restart_unused_codes=True)
        payload = ffhq.load_stage1_checkpoint(Path(checkpoint))
        state = payload['state_dict']
        filtered = {k:v for k,v in state.items() if not k.startswith('quantizer.')}
        missing, unexpected = stage1.load_state_dict(filtered,strict=False)
        assert not unexpected and all(k.startswith('quantizer.') for k in missing)
        self.encoder,self.quant_conv = stage1.encoder,stage1.quant_conv
        self.post_quant_conv,self.decoder = stage1.post_quant_conv,stage1.decoder
        self.register_buffer('dictionary',F.normalize(state['quantizer.dictionary'].float(),dim=0))
        assert self.dictionary.shape == (256,num_atoms)
        self.register_buffer('coeff_bins',torch.linspace(-3.,3.,2048))
        self.register_buffer('coeff_scales',torch.tensor(coeff_scales,dtype=torch.float32))
        assert torch.isfinite(self.coeff_scales).all() and (self.coeff_scales>0).all()
        self.num_atoms,self.coeff_vocab_size,self.vocab_size = num_atoms,2048,num_atoms+2048
        self.coeff_max,self.coeff_scale = 3.,1.
        self.soft_target_physical,self.clamp_coeffs = False,True
        self.eval().requires_grad_(False)

    def encode_sparse_components(self, images):
        raise RuntimeError('This adapter consumes the verified K=4 Church token cache')

    @torch.no_grad()
    def compound_embeddings(self, atoms, coeff_ids):
        vectors = self.dictionary.t()[atoms.long()]
        coefficients = self.coeff_bins[coeff_ids.long().clamp(0,2047)]
        scales = self.coeff_scales.view(*([1]*(coefficients.ndim-1)),self.sparsity_level)
        return vectors*(coefficients*scales)[...,None]

    @torch.no_grad()
    def physical_contributions(self, atoms, coeffs):
        vectors = self.dictionary.t()[atoms.long()]
        scales = self.coeff_scales.view(*([1]*(coeffs.ndim-1)),self.sparsity_level)
        return vectors*(coeffs.float()*scales)[...,None]


def build_model(*, num_atoms=16384, sparsity_level=4):
    # Exact FFHQ-350m configuration, with Church's atom vocabulary/depth.
    cfg = ffhq.OmegaConf.create({
        'type':'rq-transformer','block_size':[8,8,sparsity_level],
        'embed_dim':1024,'input_embed_dim':256,'shared_tok_emb':True,
        'shared_cls_emb':True,'input_emb_vqvae':True,'head_emb_vqvae':True,
        'cumsum_depth_ctx':True,'vocab_size':num_atoms,'vocab_size_cond':1,
        'block_size_cond':1,'body':{'n_layer':24,'block':{'n_head':16}},
        'head':{'n_layer':4,'block':{'n_head':16}}})
    return ffhq.CompoundLaserRQTransformer(
        ffhq.RQTransformerConfig.create(cfg),num_atoms=num_atoms,coeff_vocab_size=2048,
        refiner_layers=0,geometry_head=False,micro_transformer_layers=2,
        depth_specific_coeff_heads=True)
