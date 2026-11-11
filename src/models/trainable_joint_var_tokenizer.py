"""Train finite LASER joint tokens at the same 680-token budget as VAR.

The four-scale recipe has 340 spatial sites and TWO independent joint IDs per
site. Dictionary directions, scalar levels, CNNs and residual convolutions are
optimized during stage 1; no post-training scalar fitting is required.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .joint_multiscale_var import JointMultiScaleQuantizer
from .multiscale_laser_var import VQVAE

PATCH_NUMS_680 = (2, 4, 8, 16)


class MultiScaleGeometry(nn.Module):
    def __init__(self, channels=32, sparsity=2, patch_nums=PATCH_NUMS_680, quant_resi=None):
        super().__init__()
        if len(patch_nums)<2 or any(a>=b for a,b in zip(patch_nums,patch_nums[1:])):
            raise ValueError('Use strictly increasing scale sizes')
        self.Cvae=int(channels);self.sparsity=int(sparsity);self.v_patch_nums=tuple(patch_nums)
        self.quant_resi=quant_resi

    def contribution(self, vectors, scale):
        pn,final=self.v_patch_nums[scale],self.v_patch_nums[-1]
        x=vectors.transpose(1,2).reshape(len(vectors),self.Cvae,pn,pn)
        if pn!=final:x=F.interpolate(x,(final,final),mode='bicubic',align_corners=False)
        return self.quant_resi[scale/(len(self.v_patch_nums)-1)](x) if self.quant_resi is not None else x

    def next_input(self, accumulated, scale):
        return F.interpolate(accumulated,(self.v_patch_nums[scale],)*2,mode='area').flatten(2).transpose(1,2)


class TrainableJointMultiScale(JointMultiScaleQuantizer):
    def __init__(self, geometry, dictionary, coefficients, beta=.25):
        super().__init__(geometry,dictionary,coefficients)
        # The inherited frozen tables are assignment snapshots. Embeddings below
        # use trainable parameters directly, so dictionary/scalar gradients flow.
        self.atom_directions=nn.Parameter(dictionary.detach().float().clone())
        magnitudes=coefficients.detach().float().abs().clamp_min(1.e-4)
        self.raw_magnitudes=nn.Parameter(torch.log(torch.expm1(magnitudes)))
        self.register_buffer('level_signs',torch.sign(coefficients.detach().float()))
        self.source.requires_grad_(True)
        self.beta=beta

    def physical_tables(self):
        return F.normalize(self.atom_directions.float(),dim=0), self.level_signs*(F.softplus(self.raw_magnitudes.float())+1.e-6)

    @torch.no_grad()
    def sync_assignment_tables(self):
        dictionary,coefficients=self.physical_tables()
        self.dictionary.copy_(dictionary);self.coefficients.copy_(coefficients)

    def embed_joint(self,codes,scale):
        dictionary,coefficients=self.physical_tables()
        atoms=codes.div(self.levels,rounding_mode='floor');bins=codes.remainder(self.levels)
        return F.embedding(atoms,dictionary.T)*coefficients[scale,atoms,bins,None]

    def decompose(self,latent):
        with torch.autocast(device_type=latent.device.type,enabled=False):
            z=latent.float();self.sync_assignment_tables()
            if z.shape[1:]!=(self.Cvae,self.v_patch_nums[-1],self.v_patch_nums[-1]):raise ValueError('Invalid latent shape')
            if not torch.isfinite(z).all():raise FloatingPointError('Nonfinite latent')
            accumulated=torch.zeros_like(z);codes=[];inputs=[];losses=[]
            for scale,pn in enumerate(self.v_patch_nums):
                if scale:inputs.append(self.next_input(accumulated.detach(),scale))
                residual=F.interpolate(z.detach()-accumulated.detach(),(pn,pn),mode='area').flatten(2).transpose(1,2)
                with torch.no_grad():ids,_=self.quantize_scale(residual,scale,chunk_size=2048)
                vectors=self.embed_joint(ids,scale).sum(-2)
                accumulated=accumulated+self.contribution(vectors,scale)
                losses.append(F.mse_loss(accumulated,z.detach())+self.beta*F.mse_loss(accumulated.detach(),z))
                codes.append(ids)
            return dict(codes=torch.cat(codes,1),inputs=torch.cat(inputs,1),latent=accumulated,loss=torch.stack(losses).mean())

    def forward(self,latent,ret_usages=False):
        result=self.decompose(latent);self.last_atom_ids=result['codes'].detach();self.last_clip_fraction=latent.new_zeros(())
        straight=latent.float()+(result['latent']-latent.float()).detach()
        return straight,None,result['loss']

    @torch.no_grad()
    def export_codebook(self):
        dictionary,coefficients=self.physical_tables()
        return dict(kind='multiscale_joint_atom_coefficient',format_version=1,
                    dictionary=dictionary.cpu(),coefficients=coefficients.cpu(),
                    patch_nums=list(self.v_patch_nums),sparsity=self.sparsity,
                    vocabulary=self.vocab_size,tokens_per_image=sum(p*p for p in self.v_patch_nums)*self.sparsity)


def fresh_joint_quantizer(channels=32, directions=1024, levels=4, quant_resi=None, seed=17):
    if levels!=4:raise ValueError('The matched four-scale draft retains four coefficient levels')
    generator=torch.Generator().manual_seed(seed)
    dictionary=F.normalize(torch.randn(channels,directions,generator=generator),dim=0)
    coefficients=torch.tensor([-1.,-1/3,1/3,1.]).expand(4,directions,4).clone()
    geometry=MultiScaleGeometry(channels=channels,quant_resi=quant_resi)
    return TrainableJointMultiScale(geometry,dictionary,coefficients)


class FourScaleJointVAE(VQVAE):
    def __init__(self, channels=32, directions=1024, ch=160, seed=17):
        if directions!=1024:raise ValueError('The 4096-symbol draft requires 1024 directions and four levels')
        super().__init__(vocab_size=4096,z_channels=channels,ch=ch,
                         v_patch_nums=PATCH_NUMS_680,test_mode=False)
        residual_convs=self.quantize.quant_resi
        self.quantize=fresh_joint_quantizer(channels,directions,4,residual_convs,seed)

    @torch.no_grad()
    def tokenize(self,images):
        return self.quantize.decompose(self.quant_conv(self.encoder(images)))

    def load_state_dict(self,state_dict,strict=True,assign=False):
        return nn.Module.load_state_dict(self,state_dict,strict=strict,assign=assign)
