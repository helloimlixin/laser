"""Keep the native stochastic AR teacher; use beam search only for reconstruction."""
from pathlib import Path
import importlib.util
import torch
from src.models.laser_joint_tokenizer import FrozenJointLaserTokenizer


class PairedReconstructionTokenizer(FrozenJointLaserTokenizer):
    def __init__(self,checkpoint,codebook,*,reconstruction_beam_width=4):
        super().__init__(checkpoint,codebook)
        if reconstruction_beam_width not in (1,2,4,8,16):
            raise ValueError('Unsupported reconstruction beam width')
        if getattr(self.quantizer,'encoding_beam_width',1)!=1:
            raise ValueError('The AR teacher must remain the native stochastic quantizer')
        self.reconstruction_beam_width=int(reconstruction_beam_width)
        path=Path(__file__).with_name('beam_search.py')
        if not path.exists():path=Path('/workspace/Projects/laser/src/models/laser_joint_search.py')
        spec=importlib.util.spec_from_file_location('paired_fixed_codeword_beam',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self._reconstruction_search=module.beam_quantize

    @torch.no_grad()
    def quantize_reconstruction(self,latent):
        greedy=self.quantizer.quantize(latent,chunk_size=1024)
        return self._reconstruction_search(self.quantizer,latent,
            beam_width=self.reconstruction_beam_width,chunk_size=256,greedy=greedy)

    @torch.no_grad()
    def forward(self,images):
        latent=self.encode(images)
        result=self.quantize_reconstruction(latent)
        return self.decode(result['quantized']),latent.new_zeros(()),result['codes']
