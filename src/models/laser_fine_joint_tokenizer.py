"""Frozen LASER encoding with packed atom / orthogonal coefficient IDs."""
import torch

from src.models.laser_tokenizer_adapter import FrozenLaserTokenizer
from src.orthogonal_sparse_codec import dictionary_to_orthogonal_coefficients, ordered_support_basis
from src.training.laser_hard_ar import sparse_encoding, nearest_coefficient_ids


class FrozenFineJointTokenizer(FrozenLaserTokenizer):
    def __init__(self, checkpoint, **options):
        super().__init__(checkpoint, **options)
        self.vocab_size = self.num_atoms * self.coeff_vocab_size

    @torch.no_grad()
    def encode(self, images):
        return self.quant_conv(self.encoder(images.float())).permute(0, 2, 3, 1).contiguous().float()

    @torch.no_grad()
    def latent_codes(self, latent):
        if not hasattr(self, '_encoding_gram'):
            self._encoding_gram = (self.dictionary.T @ self.dictionary).contiguous()
        atoms, coefficients = sparse_encoding(latent, self.dictionary, self.sparsity_level,
                                              gram=self._encoding_gram)
        gamma, _ = dictionary_to_orthogonal_coefficients(self.dictionary.T[atoms], coefficients)
        bins = nearest_coefficient_ids(gamma / self.coeff_scales, self.coeff_bins)
        return atoms * self.coeff_vocab_size + bins, gamma

    @torch.no_grad()
    def get_code_emb_with_depth(self, codes):
        atoms, bins = codes // self.coeff_vocab_size, codes % self.coeff_vocab_size
        basis, _ = ordered_support_basis(self.dictionary.T[atoms])
        return basis * (self.coeff_bins[bins] * self.coeff_scales)[..., None], None

    @torch.no_grad()
    def decode_code(self, codes):
        latent = self.get_code_emb_with_depth(codes)[0].sum(-2)
        return self.decoder(self.post_quant_conv(latent.permute(0, 3, 1, 2).contiguous()))

    @torch.no_grad()
    def forward(self, images):
        codes, _ = self.latent_codes(self.encode(images))
        return self.decode_code(codes), None, codes
