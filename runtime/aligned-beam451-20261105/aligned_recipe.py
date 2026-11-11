"""Native RQ cross entropy with a beam-aligned conditional soft teacher."""
import torch
from src.models.rfid_preserved_joint_tokenizer import RFIDPreservedJointTokenizer
from src.training.laser_joint_recipe import JointLaserRecipe
from runtime.beam_teacher import aligned_targets


class AlignedBeamRecipe(JointLaserRecipe):
    def tokenizer(self, device):
        return RFIDPreservedJointTokenizer(self.base/'inputs/stage1-model.pt',
            self.base/'inputs/joint-codebook.pt', self.base/'inputs/reconstruction-codebook.pt').to(device), None

    def target_temperature(self):
        return float(self.recipe.targets.sequence_temperature)

    @torch.no_grad()
    def targets(self, aux, images):
        with torch.autocast(device_type=images.device.type, enabled=False):
            latent = torch.cat([aux.encode(chunk.float()) for chunk in images.split(8)])
            return aligned_targets(aux.physical_book, latent, self.target_temperature())

    def verify_targets(self, aux, targets, codes):
        result = super().verify_targets(aux, targets, codes)
        result.update(teacher='conditional posterior over beam16 complete sequences',
            beam_width=16, prefixes_from_same_teacher_distribution=True,
            canonical_encoding='exact saved rFID4.512 beam16 with greedy anchor',
            native_greedy_soft_teacher_used=False, one_hot_beam_targets=False,
            max_candidate_relative_latent_error_above_canonical=.01)
        return result

    def metadata(self):
        result = super().metadata()
        result.update(teacher='beam16 sequence posterior with exact prefix conditionals',
            teacher_beam_width=16, reconstruction_encoding_beam_width=16,
            teacher_matches_reconstruction_search=True,
            teacher_sampled_reconstruction_rfid=float(self.recipe.tokenizer.teacher_sampled_rfid),
            source_canonical_reconstruction_rfid=4.5122860099749005,
            teacher_sequence_temperature=self.target_temperature(),
            teacher_calibration='training-only entropy and latent-error budget',
            native_greedy_soft_teacher_used=False, hard_beam_targets=False)
        return result
