"""Restore beam weights and optimizer, then change only the declared recipe."""
import json
import torch
from omegaconf import OmegaConf
from src.training.laser_joint_recipe import JointLaserRecipe
from src.models.rfid_preserved_joint_tokenizer import RFIDPreservedJointTokenizer
from runtime.mixture_teacher import get_targets


class MixtureRecipe(JointLaserRecipe):
    def tokenizer(self, device):
        return RFIDPreservedJointTokenizer(
            self.base / 'inputs/stage1-model.pt', self.base / 'inputs/joint-codebook.pt',
            self.base / 'inputs/reconstruction-codebook.pt').to(device), None

    def allowed_source_runs(self):
        return (str(self.recipe.lineage.source_run),)

    def target_temperature(self):
        return float(self.recipe.targets.native_temperature)

    @torch.no_grad()
    def targets(self, aux, images):
        with torch.autocast(device_type=images.device.type, enabled=False):
            latent = torch.cat([aux.encode(chunk.float()) for chunk in images.split(8)])
            return get_targets(aux, latent, float(self.recipe.targets.native_weight),
                               temperature=self.target_temperature())

    def verify_targets(self, aux, targets, codes):
        result = super().verify_targets(aux, targets, codes)
        assert result['entropy'] > .2
        result.update(teacher='75% beam-best + 25% native-soft sequence mixture',
                      native_weight=float(self.recipe.targets.native_weight),
                      native_temperature=self.target_temperature(),
                      exact_prefix_conditionals=True, full_vocabulary_native_component=True)
        return result

    def metadata(self):
        result = super().metadata()
        result.update(
            teacher='beam-best/native-soft complete-sequence mixture',
            native_weight=float(self.recipe.targets.native_weight),
            native_temperature=self.target_temperature(),
            teacher_sampled_reconstruction_rfid=float(self.recipe.tokenizer.teacher_sampled_rfid),
            teacher_rfid_independent_seed=float(self.recipe.tokenizer.teacher_rfid_independent_seed),
            measured_teacher_entropy=float(self.recipe.targets.measured_entropy),
            calibration_images=50000, calibration_split='train',
            full_vocabulary_native_component=True, exact_prefix_conditionals=True,
            coefficient_refitting=False, reconstruction_beam_width=16,
            continuation_source_run=str(self.recipe.lineage.source_run),
            continuation_source_step=7512)
        return result

    def verify_resume(self, payload):
        assert payload['joint_codebook_sha256'] == self.codebook_sha256
        if payload['run_id'] == self.run_id:
            assert payload['recipe'] == OmegaConf.to_container(self.recipe, resolve=True)
            return
        assert payload['run_id'] in self.allowed_source_runs()
        assert payload['global_step'] == 7512 and payload['next_epoch'] == 12
        assert payload['next_microbatch'] == 0
        source = payload['recipe']
        for key in ['backbone', 'decoder', 'target_mode', 'objective',
                    'auxiliary_loss_terms', 'global_batch', 'lr_schedule_epochs', 'initialization']:
            assert source[key] == OmegaConf.to_container(self.recipe, resolve=True)[key], key
        for key in ['sha256', 'codebook_sha256', 'reconstruction_codebook_sha256',
                    'atoms', 'vocab_size', 'depth', 'coefficient_levels_per_atom']:
            assert source['tokenizer'][key] == self.recipe.tokenizer[key], key
        assert source['targets']['encoding'] == 'beam16 complete-sequence posterior and exact prefix conditionals'
        assert float(source['targets']['max_relative_candidate_error']) == .01
        assert payload['runtime']['parameter_count'] == 1387529728
        self.source_transition = dict(
            source_run=payload['run_id'], source_step=payload['global_step'],
            source_evaluation=payload.get('evaluation'),
            model_and_optimizer_restored=True, cosine_schedule_not_restarted=True,
            changed_teacher=True, changed_precision='float16 -> bfloat16',
            changed_gradient_clip='1.0 -> 0.1', source_evaluation_sampler_differs=True)
