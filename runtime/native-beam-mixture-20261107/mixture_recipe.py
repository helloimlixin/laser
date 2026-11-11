"""Native RQ training with the coherent full-support native/beam mixture."""
import torch
from src.training.laser_joint_recipe import JointLaserRecipe
from src.models.rfid_preserved_joint_tokenizer import RFIDPreservedJointTokenizer
from runtime.mixture_teacher import get_targets


class MixtureRecipe(JointLaserRecipe):
    def tokenizer(self,device):
        return RFIDPreservedJointTokenizer(self.base/'inputs/stage1-model.pt',
            self.base/'inputs/joint-codebook.pt',self.base/'inputs/reconstruction-codebook.pt').to(device),None

    def target_temperature(self):return .5

    @torch.no_grad()
    def targets(self,aux,images):
        with torch.autocast(device_type=images.device.type,enabled=False):
            latent=torch.cat([aux.encode(chunk.float()) for chunk in images.split(8)])
            return get_targets(aux,latent,float(self.recipe.targets.native_weight),temperature=.5)

    def verify_targets(self,aux,targets,codes):
        result=super().verify_targets(aux,targets,codes)
        result.update(teacher='native-soft/beam-best sequence mixture; exact prefix conditionals',
            native_weight=float(self.recipe.targets.native_weight),native_temperature=.5,
            prefixes_from_same_teacher_distribution=True,full_vocabulary_native_component=True,
            finite_beam_support_limit=False,near_best_error_mask=False)
        if float(self.recipe.targets.native_weight)>0:assert result['entropy']>.10
        return result

    def metadata(self):
        result=super().metadata()
        result.update(teacher='native-soft/beam-best sequence mixture',native_temperature=.5,
            native_weight=float(self.recipe.targets.native_weight),
            full_vocabulary_native_component=True,teacher_and_prefix_distribution_aligned=True,
            teacher_sampled_reconstruction_rfid=float(self.recipe.tokenizer.teacher_sampled_rfid),
            measured_teacher_entropy=float(self.recipe.targets.measured_entropy),
            teacher_calibration='training screen; predeclared mixture weights selected by full validation reconstruction and entropy',
            validation_used_for_gradient_training=False)
        return result
