"""Original LASER CNN with frozen, native RQ vector lookup tables."""
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from src import rq_upstream
from src.models.laser_tokenizer_adapter import FrozenLaserTokenizer
from rqvae.models.rqvae.quantizations import RQBottleneck


def file_digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def module_digest(module):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def read_vectors(book):
    book = Path(book)
    spec = torch.load(book, map_location='cpu', weights_only=True)
    assert spec['kind'] == 'native_rq24576_vectors' and spec['vocabulary'] == 24576
    parts = []
    for record in spec['vector_shards']:
        path = book.parent / record['name']
        assert file_digest(path) == record['sha256']
        parts.append(torch.load(path, map_location='cpu', weights_only=True))
    vectors = torch.stack(parts)
    assert vectors.shape == (4, 24576, 256) and torch.isfinite(vectors).all()
    return vectors, spec


class NativeVectorTokenizer(nn.Module):
    def __init__(self, checkpoint, book):
        super().__init__()
        vectors, self.spec = read_vectors(book)
        assert file_digest(checkpoint) == self.spec['stage1_checkpoint_sha256']
        self.backbone = FrozenLaserTokenizer(checkpoint, num_atoms=16384, coeff_vocab_size=2,
                                            coeff_max=1., coeff_scales=[1.] * 4, sparsity_level=4)
        self.quantizer = RQBottleneck(latent_shape=[8, 8, 256], code_shape=[8, 8, 4],
                                     n_embed=24576, decay=.99, shared_codebook=False,
                                     restart_unused_codes=False)
        with torch.no_grad():
            for depth, table in enumerate(self.quantizer.codebooks):
                table.weight[:-1].copy_(vectors[depth])
                table.weight[-1].zero_()
                table.embed_ema.copy_(vectors[depth])
                table.cluster_size_ema.fill_(1.)
        self.quantizer.vocab_size = 24576
        self.quantizer.depth = 4
        self.code_shape = torch.Size([8, 8, 4])
        self.requires_grad_(False).eval()

    def encode(self, images):
        return self.backbone.quant_conv(self.backbone.encoder(images)).permute(0, 2, 3, 1).contiguous()

    def decode(self, latent):
        x = self.backbone.post_quant_conv(latent.permute(0, 3, 1, 2).contiguous())
        return self.backbone.decoder(x)

    def get_code_emb_with_depth(self, codes):
        return self.quantizer.embed_code_with_depth(codes)

    def decode_code(self, codes):
        return self.decode(self.quantizer.embed_code(codes))

    @torch.no_grad()
    def quantize_reconstruction(self, latent):
        _, codes = self.quantizer.quantize(latent)
        return dict(codes=codes, quantized=self.quantizer.embed_code(codes))

    @torch.no_grad()
    def forward(self, images):
        codes = self.quantize_reconstruction(self.encode(images))['codes']
        return self.decode_code(codes), images.new_zeros(()), codes
