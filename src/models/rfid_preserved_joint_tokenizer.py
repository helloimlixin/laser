"""Beam reconstruction and native RQ training over the same frozen products.

The stochastic teacher is inherited unchanged. Reconstruction explicitly uses
the saved beam encoder; decoding never depends on how the integers were chosen.
"""
import torch

from src.models.native_joint_tokenizer import NativeJointTokenizer


class RFIDPreservedJointTokenizer(NativeJointTokenizer):
    def __init__(self, checkpoint, teacher_codebook, reconstruction_codebook):
        super().__init__(checkpoint, teacher_codebook)
        spec = torch.load(reconstruction_codebook, map_location='cpu', weights_only=True)
        # Encoding policy may differ; the physical integer vocabulary may not.
        for name in ('dictionary', 'atom_ids', 'coefficients'):
            if not torch.equal(getattr(self.physical_book, name).cpu(), spec[name]):
                raise ValueError(f'Reconstruction and teacher differ in {name}')
        if spec.get('zero_code_id', 0) != self.physical_book.zero_code_id:
            raise ValueError('Reconstruction and teacher zero-code policies differ')
        width = spec.get('encoding_beam_width')
        if width != 16:
            raise ValueError('The verified reconstruction encoder requires beam width 16')
        self.physical_book.encoding_beam_width = width
        self.reconstruction_beam_width = width
        self.requires_grad_(False).eval()

    @torch.no_grad()
    def quantize_reconstruction(self, latent):
        return self.physical_book.quantize(latent)

    @torch.no_grad()
    def forward(self, images):
        latent = self.encode(images)
        codes = self.quantize_reconstruction(latent)['codes']
        return self.decode_code(codes), latent.new_zeros(()), codes
