"""Representation adapters for the shared upstream ImageNet training driver."""
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src import rq_upstream
from rqvae.models import create_model
from rqvae.utils.config import load_config


class RQRecipe:
    kind = 'rq'
    run_id = 'imagenet-original-rq1400m-rfid473-scratch20-4h100-20261014'
    name = 'Original RQ-Transformer 1.4B | rFID4.73 tokenizer | fresh20'
    shape = (8, 8, 4)
    checkpoint_key = 'original_rqvae_stage1_sha256'
    microbatch = 64
    epochs = 20
    schedule_epochs = 100
    generation_batch = 256

    def __init__(self, base):
        self.base = Path(base)
        self._native_sampling_override = None

    def tokenizer(self, device):
        from rqvae.utils.config import augment_arch_defaults
        config = load_config(self.base/'inputs/stage1-config.yaml')
        model, _ = create_model(augment_arch_defaults(config.arch), ema=False)
        state = torch.load(self.base/'inputs/stage1-model.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(state['state_dict'], strict=True)
        model.requires_grad_(False).eval().to(device)
        assert list(model.code_shape) == [8, 8, 4] and model.quantizer.shared_codebook
        assert model.quantizer.codebooks[0].weight.shape == (16385, 256)
        return model, config

    def model(self, arch):
        model, _ = create_model(arch, ema=False)
        model.apply(model._init_weights)
        return model

    @torch.no_grad()
    def targets(self, aux, images):
        latents = torch.cat([aux.encode(chunk.float()) for chunk in images.split(16)])
        return aux.quantizer.get_soft_codes(latents, temp=.5, stochastic=True)

    def verify_targets(self, aux, targets, codes):
        assert targets.shape == (*codes.shape, 16384) and codes.shape[1:] == self.shape
        torch.testing.assert_close(targets.sum(-1), torch.ones_like(codes, dtype=torch.float32),
                                   rtol=2e-5, atol=2e-5)
        assert torch.isfinite(targets).all() and targets.dtype == torch.float32
        entropy = -(targets * targets.clamp_min(1e-30).log()).sum(-1).mean()
        return dict(soft_targets=True, stochastic_prefixes=True, shape=list(targets.shape),
                    dtype=str(targets.dtype), temperature=.5, entropy=float(entropy),
                    original_native_quantizer=True)

    def loss(self, model, logits, targets):
        return model.compute_loss(logits.float(), targets, use_soft_target=True)

    def sampling_settings(self):
        """Use the experiment config, or an existing checkpoint's sampler."""
        if self._native_sampling_override is not None:
            return dict(self._native_sampling_override)
        # The released checkpoint's top-level sampling section differs from
        # the repository training YAML's experiment.sample defaults.
        archive = self.base/'inputs/stage2-archive-config.yaml'
        archived = load_config(archive) if archive.exists() else None
        if archived is not None and archived.get('sampling') is not None:
            options = archived.sampling
        else:
            config = load_config(self.base/'inputs/stage2-training-config.yaml')
            options = config.experiment.sample
        def values(value, cast):
            return [cast(v) for v in value] if OmegaConf.is_list(value) else [cast(value)]
        return dict(temperature=float(options.get('temperature', options.get('temp', 1.))),
                    top_k=values(options.top_k, int), top_p=values(options.top_p, float))

    def sample(self, model, aux, count, labels):
        return model.sample(torch.zeros(count, *self.shape, dtype=torch.long, device=labels.device),
            model_aux=aux, cond=labels, amp=True, **self.sampling_settings())

    def decode(self, aux, codes):
        return aux.decode_code(codes)

    def digest(self, aux):
        import hashlib
        digest = hashlib.sha256()
        for name, value in aux.quantizer.state_dict().items():
            digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def metadata(self):
        return dict(architecture='unmodified upstream RQTransformer',stage1='original RQ-VAE rFID4.73',
            stochastic_codes=True,soft_labels=True,soft_label_temperature=.5,
            objective='native_soft_target_cross_entropy',auxiliary_loss_terms=0,
            native_sampling=self.sampling_settings(),
            shared_upstream=rq_upstream.provenance())

    def checkpoint_extra(self):
        return dict(native_sampling=self.sampling_settings())

    def logging_sampling_settings(self):
        options = self.sampling_settings()
        return dict(sampler_temperature=options['temperature'], sampler_top_k=options['top_k'][0],
                    sampler_top_p=options['top_p'][0])

    def verify_resume(self, payload):
        # Older shared-driver checkpoints used a hard-coded sampler even when
        # their YAML specified another one. Preserve those FID curves on resume.
        self._native_sampling_override = payload.get('native_sampling',
            dict(temperature=1., top_k=[256], top_p=[.95]))

    def adapt(self, model, aux):
        """Optional representation adapter; native RQ uses the upstream model."""
        return None


class LaserRecipe(RQRecipe):
    kind = 'laser'
    name = 'LASER | shared upstream RQ | one CE | fresh20'
    shape = (8, 8, 8)
    checkpoint_key = 'stage1_checkpoint_sha256'

    def __init__(self, base):
        super().__init__(base)
        self.recipe = OmegaConf.load(self.base/'inputs/recipe.yaml')
        assert self.recipe.auxiliary_loss_terms == 0
        self.run_id = str(self.recipe.run_id)
        self.name = str(self.recipe.get('name', self.name))
        self.microbatch = int(self.recipe.microbatch)
        self.epochs = int(self.recipe.epochs)
        self.schedule_epochs = int(self.recipe.lr_schedule_epochs)
        self.generation_batch = int(self.recipe.evaluation.batch_size)

    def tokenizer(self, device):
        from src.models.laser_tokenizer_adapter import FrozenLaserTokenizer
        options = self.recipe.tokenizer
        model = FrozenLaserTokenizer(self.base/'inputs/stage1-model.pt',
            num_atoms=int(options.num_atoms), coeff_vocab_size=int(options.coefficient_bins),
            coeff_max=float(options.coefficient_max), coeff_scales=list(options.coefficient_scales),
            sparsity_level=int(options.depth)).to(device)
        return model, None

    def model(self, arch):
        from src.training.laser_rq_minimal import build_model
        options = self.recipe.tokenizer
        if self.recipe.get('decoder') == 'compound_pair':
            from src.models.laser_compound_rq_adapter import build_compound_model
            build_model = build_compound_model
        elif self.recipe.get('decoder') == 'scalar_pair_context':
            from src.models.laser_pair_context_adapter import build_pair_context_model
            build_model = build_pair_context_model
        return build_model(arch, num_atoms=int(options.num_atoms),
            coefficient_bins=int(options.coefficient_bins),depth=int(options.depth))

    def adapt(self, model, aux):
        factors = self.recipe.get('ar_factors')
        result = None
        if factors is not None:
            from src.training.laser_ar_factors import configure
            result = configure(self, model, aux, coordinates=str(factors.coordinates),
                               bins=int(factors.coefficient_bins))
        if self.recipe.get('coefficient_policy') == 'soft':
            from src.training.laser_pair_ablation import configure_coefficient_policy
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            self.coefficient_generator = configure_coefficient_policy(
                self, model, aux, policy='soft', seed=261015 + rank)
        return result

    def targets(self, aux, images):
        if self.recipe.get('target_mode') == 'hard_sparse_tokens':
            from src.training.laser_hard_ar import image_tokens
            tokens = image_tokens(aux, images, self.recipe.targets)
            return tokens, tokens
        from src.training.laser_rq_minimal import image_targets
        targets = image_targets(aux, images, self.recipe.teacher, backend=str(self.recipe.teacher.backend))
        return targets, targets['tokens']

    def verify_targets(self, aux, targets, codes):
        if self.recipe.get('target_mode') == 'hard_sparse_tokens':
            from src.training.laser_hard_ar import verify_tokens
            verify_tokens(aux, codes)
            assert codes.shape[1:] == self.shape
            soft = isinstance(targets, dict)
            if soft:
                probabilities = targets['coefficient_probabilities']
                assert torch.isfinite(probabilities).all() and (probabilities >= 0).all()
                torch.testing.assert_close(probabilities.sum(-1), torch.ones_like(probabilities[..., 0]),
                                           rtol=2.e-5, atol=2.e-5)
                torch.testing.assert_close(targets['tokens'][..., 0::2], codes[..., 0::2], rtol=0, atol=0)
            return dict(soft_targets=soft, stochastic_codes=False, shape=list(codes.shape),
                dtype=str(codes.dtype), encoding='deterministic_stage1_sparse_support_and_coefficients',
                coefficient_quantization='nearest_bin', sampled_trajectories=0, auxiliary_loss_terms=0)
        from src.training.laser_rq_minimal import verify_targets
        verify_targets(aux, targets)
        assert codes.shape[1:] == self.shape
        return dict(soft_targets=True,stochastic_prefixes=True,shape=list(codes.shape),
            teacher=OmegaConf.to_container(self.recipe.teacher),prefix_conditional_labels=True,
            auxiliary_loss_terms=0)

    def loss(self, model, logits, targets):
        if self.recipe.get('target_mode') == 'hard_sparse_tokens':
            from src.training.laser_hard_ar import token_cross_entropy
            return token_cross_entropy(logits, targets)
        from src.training.laser_rq_minimal import token_cross_entropy
        return token_cross_entropy(logits, targets)

    def sample(self, model, aux, count, labels):
        settings = self.recipe.sampling
        return model.sample_sparse(count,aux,cond=labels,amp=True,
            atom_temperature=float(settings.atom_temperature),atom_top_k=int(settings.atom_top_k),
            atom_top_p=float(settings.atom_top_p),coeff_temperature=float(settings.coefficient_temperature),
            coeff_top_k=int(settings.coefficient_top_k),coeff_top_p=float(settings.coefficient_top_p))

    def decode(self, aux, codes):
        return aux.decode_tokens(codes)

    def digest(self, aux):
        from src.training.laser_rq_minimal import frozen_digest
        return frozen_digest(aux)

    def metadata(self):
        if self.recipe.get('target_mode') == 'hard_sparse_tokens':
            soft = self.recipe.get('coefficient_policy') == 'soft'
            return dict(architecture=('upstream RQTransformer with LASER four-pair coefficient decoder'
                    if self.recipe.get('decoder') == 'compound_pair' else
                    'upstream RQTransformer with zero-initialized LASER pair context'
                    if self.recipe.get('decoder') == 'scalar_pair_context' else
                    'upstream RQTransformer with LASER physical-pair embedding adapter'),
                stage1='LASER source checkpoint rFID4.21091413', stochastic_codes=False, soft_labels=soft,
                objective=str(self.recipe.objective), auxiliary_loss_terms=0,
                target_mode='hard_sparse_tokens', sampled_trajectories=0, teacher_bank_size=0,
                coefficient_quantization='nearest_bin', coefficient_jitter=soft,
                ar_factors=OmegaConf.to_container(self.recipe.get('ar_factors',
                    OmegaConf.create(dict(coordinates='physical',coefficient_bins=2048)))),
                targets=OmegaConf.to_container(self.recipe.targets), shared_upstream=rq_upstream.provenance())
        return dict(architecture='upstream RQTransformer with LASER physical-pair embedding adapter',
            stage1='LASER rFID4.21091413',stochastic_codes=True,soft_labels=True,
            objective=str(self.recipe.objective),auxiliary_loss_terms=0,
            teacher_compute_backend='fused_fp32_full_bank_cuda_graph_v1',
            teacher=OmegaConf.to_container(self.recipe.teacher),shared_upstream=rq_upstream.provenance())

    def checkpoint_extra(self):
        return dict(recipe=OmegaConf.to_container(self.recipe,resolve=True))

    def logging_sampling_settings(self):
        options = self.recipe.sampling
        return dict(sampler_temperature=float(options.atom_temperature),
                    sampler_top_k=int(options.atom_top_k), sampler_top_p=float(options.atom_top_p))

    def verify_resume(self, payload):
        assert payload['recipe'] == OmegaConf.to_container(self.recipe,resolve=True)
