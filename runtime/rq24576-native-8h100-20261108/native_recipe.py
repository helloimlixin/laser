"""Unmodified RQTransformer and upstream stochastic soft-target quantization."""
import torch
from src.training.laser_joint_recipe import JointLaserRecipe
from runtime.tokenizer import NativeVectorTokenizer, read_vectors, module_digest


class NativeRecipe(JointLaserRecipe):
    def tokenizer(self, device):
        return NativeVectorTokenizer(self.base / 'inputs/stage1-model.pt',
                                     self.base / 'inputs/joint-codebook.pt').to(device), None

    @torch.no_grad()
    def verify_codec(self, aux):
        assert not aux.training and all(not p.requires_grad for p in aux.parameters())
        assert module_digest(aux.backbone) == aux.spec['source_cnn_digest']
        vectors, _ = read_vectors(self.base / 'inputs/joint-codebook.pt')
        for depth, table in enumerate(aux.quantizer.codebooks):
            torch.testing.assert_close(table.weight[:-1], vectors[depth].to(table.weight.device), rtol=0, atol=0)
        return dict(frozen_codec_verified=True, source_CNN_unchanged=True,
                    codeword_tables_verified=True)

    def target_temperature(self):
        return float(self.recipe.targets.temperature)

    @torch.no_grad()
    def targets(self, aux, images):
        with torch.autocast(device_type=images.device.type, enabled=False):
            latent = torch.cat([aux.encode(chunk.float()) for chunk in images.split(8)])
            return aux.quantizer.get_soft_codes(latent, temp=self.target_temperature(), stochastic=True)

    def verify_targets(self, aux, targets, codes):
        result = super().verify_targets(aux, targets, codes)
        assert result['vocabulary'] == 24576 and result['entropy'] > 0
        result.update(native_RQBottleneck=True, beam_search=False, mixture_teacher=False,
                      no_separate_coefficient_tokens=True)
        return result

    def metadata(self):
        result = super().metadata()
        result.update(stage1='Original frozen LASER CNN, fitted24576-vector native RQ codebooks',
                      final_codewords=24576, no_separate_coefficient_tokens=True,
                      beam_search=False, mixture_teacher=False, native_RQBottleneck=True,
                      vector_codebook_fitting_images=50000, heldout_training_images=10000,
                      measured_tokenizer_rfid=float(self.recipe.tokenizer.measured_rfid),
                      teacher_sampled_reconstruction_rfid=float(self.recipe.tokenizer.teacher_sampled_rfid),
                      measured_teacher_entropy=float(self.recipe.targets.measured_entropy))
        return result
