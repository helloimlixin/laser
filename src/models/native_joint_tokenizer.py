"""A frozen dictionary-product vocabulary exposed as the native RQ bottleneck."""
import torch
from src import rq_upstream
from src.models.laser_joint_tokenizer import FrozenJointLaserTokenizer
from rqvae.models.rqvae.quantizations import RQBottleneck


class NativeJointTokenizer(FrozenJointLaserTokenizer):
    def __init__(self,checkpoint,codebook):
        super().__init__(checkpoint,codebook)
        self.physical_book=self.quantizer
        book=self.physical_book
        native=RQBottleneck(latent_shape=[8,8,256],code_shape=[8,8,book.depth],
            n_embed=book.vocab_size,decay=.99,shared_codebook=False,restart_unused_codes=False)
        with torch.no_grad():
            for depth,table in enumerate(native.codebooks):
                values=book.expanded_codebook(depth)
                table.weight[:-1].copy_(values);table.weight[-1].zero_()
                table.embed_ema.copy_(values);table.cluster_size_ema.fill_(1.)
        native.vocab_size=book.vocab_size;native.depth=book.depth
        native.encoding_beam_width=1;native.training_target_mode='native_stochastic_soft'
        self.quantizer=native
        self.requires_grad_(False).eval()

    def get_code_emb_with_depth(self,codes):
        return self.quantizer.embed_code_with_depth(codes)

    def decode_code(self,codes):
        return self.decode(self.quantizer.embed_code(codes))

    def forward(self,images):
        latent=self.encode(images)
        _,codes=self.quantizer.quantize(latent)
        return self.decode_code(codes),latent.new_zeros(()),codes
