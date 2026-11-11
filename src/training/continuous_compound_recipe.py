"""Original deterministic LASER OMP targets with an unquantized compound prior."""
import torch

from src.continuous_sparse_codec import continuous_image_codes, continuous_latents
from src.models.continuous_compound_rq import build_continuous_compound_model, continuous_pair_loss
from src.training.rq_imagenet_recipes import LaserRecipe


class ContinuousCompoundRecipe(LaserRecipe):
    kind = 'laser_continuous_compound'

    def __init__(self, base):
        super().__init__(base)
        if self.recipe.target_mode != 'original_omp_continuous':
            raise ValueError('Continuous compound requires original deterministic OMP targets')
        if self.recipe.auxiliary_loss_terms != 0:
            raise ValueError('This comparison has no auxiliary losses')
        self.shape = (8, 8, int(self.recipe.tokenizer.depth))

    def tokenizer(self, device):
        from src.models.laser_tokenizer_adapter import FrozenLaserTokenizer
        options = self.recipe.tokenizer
        aux = FrozenLaserTokenizer(self.base / 'inputs/stage1-model.pt',
            num_atoms=int(options.num_atoms), coeff_vocab_size=None, coeff_max=None,
            coeff_scales=list(options.coefficient_scales), sparsity_level=int(options.depth)).to(device)
        return aux, None

    def model(self, arch):
        t, density = self.recipe.tokenizer, self.recipe.density
        return build_continuous_compound_model(arch, num_atoms=int(t.num_atoms),
            coefficient_scales=list(t.coefficient_scales), components=int(density.components),
            min_std=float(density.min_std))

    def adapt(self, model, aux):
        return None

    @torch.no_grad()
    def targets(self, aux, images):
        codes = continuous_image_codes(aux, images,
            encoder_batch_size=int(self.recipe.targets.encoder_batch_size))
        return codes, codes

    def verify_targets(self, aux, targets, codes):
        codes.validate(aux.num_atoms, aux.sparsity_level)
        assert targets is codes and not codes.coefficients.requires_grad
        return dict(shape=list(codes.shape), original_dictionary=True, deterministic_source_support=True,
            final_least_squares_coefficients=True, coefficient_representation='physical_fp32',
            coefficient_quantization=False, coefficient_clipping=False, coefficient_noise=False,
            stochastic_prefixes=False, auxiliary_loss_terms=0)

    def loss(self, model, output, targets):
        loss, self.last_loss_metrics = continuous_pair_loss(output, targets, model.coefficient_scales,
                                                           atom_weight=float(self.recipe.atom_ce_weight))
        return loss

    def sampling_settings(self):
        s = self.recipe.sampling
        return dict(atom_temperature=float(s.atom_temperature), atom_top_k=int(s.atom_top_k),
                    atom_top_p=float(s.atom_top_p))

    def sample(self, model, aux, count, labels):
        return model.sample_sparse(count, aux, labels, amp=True, **self.sampling_settings())

    def decode(self, aux, codes):
        latent = continuous_latents(codes, aux.dictionary).permute(0, 3, 1, 2).contiguous()
        return aux.decoder(aux.post_quant_conv(latent)).clamp(-1, 1)

    def metadata(self):
        return dict(architecture='native compound RQ with selected-atom continuous density heads',
            original_dictionary=True, coefficient_representation='physical_fp32',
            coefficient_quantization=False, coefficient_clipping=False, coefficient_noise=False,
            stochastic_codes=False, soft_labels=False, objective=str(self.recipe.objective),
            atom_ce_weight=float(self.recipe.atom_ce_weight), coefficient_nll_weight=1.,
            continuous_density='asinh Gaussian mixture with physical Jacobian',
            loss_component_diagnostics=True, auxiliary_loss_terms=0,
            native_sampling=self.sampling_settings(),
            density_nll_comparable_to_bin_cross_entropy=False,
            source_stage1_rfid=float(self.recipe.tokenizer.source_rfid))
