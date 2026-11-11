"""Original LASER compound representation with exact, deterministic 7-bit coefficients."""
import hashlib
import json

import torch
from torch.nn import functional as F
from omegaconf import OmegaConf

from src.training.rq_imagenet_recipes import LaserRecipe
from src.training.laser_hard_ar import image_tokens, sparse_encoding, verify_tokens


class Compound128Recipe(LaserRecipe):
    kind = 'laser_compound128'

    def __init__(self, base):
        super().__init__(base)
        assert self.recipe.tokenizer.num_atoms == 16384
        assert self.recipe.tokenizer.coefficient_bins == 128
        assert self.recipe.target_mode == 'hard_sparse_tokens'
        self.codebook_sha256 = hashlib.sha256(json.dumps(
            OmegaConf.to_container(self.recipe.tokenizer, resolve=True), sort_keys=True).encode()).hexdigest()

    def model(self, arch):
        from src.models.laser_native_compound_adapter import build_native_compound_model
        model = build_native_compound_model(arch, num_atoms=16384, coefficient_bins=128, depth=4)
        model.apply(model._init_weights)
        backend = self.recipe.get('execution', {}).get('attention_backend', 'native')
        if backend == 'sdpa_spatial':
            from src.training.rq_sdpa_backend import sdpa_attention
            self._attention_context = sdpa_attention(model.body_transformer)
            self._attention_context.__enter__()
        elif backend != 'native':
            raise ValueError(f'Unknown attention backend: {backend}')
        return model

    def adapt(self, model, aux):
        return None

    @torch.no_grad()
    def clean_components(self, aux, images):
        parts = []
        with torch.autocast(device_type=images.device.type, enabled=False):
            if not hasattr(aux, '_hard_ar_gram'):
                aux._hard_ar_gram = (aux.dictionary.float().T @ aux.dictionary.float()).contiguous()
            for chunk in images.split(int(self.recipe.targets.encoder_batch_size)):
                z = aux.quant_conv(aux.encoder(chunk.float())).permute(0, 2, 3, 1).contiguous()
                atoms, coefficients = sparse_encoding(z, aux.dictionary, 4, gram=aux._hard_ar_gram)
                parts.append((atoms, coefficients / aux.coeff_scales))
        return tuple(torch.cat([p[i] for p in parts]) for i in (0, 1))

    @torch.no_grad()
    def hard_codes(self, aux, images):
        return image_tokens(aux, images, self.recipe.targets)

    @torch.no_grad()
    def targets(self, aux, images):
        codes = self.hard_codes(aux, images)
        return codes, codes

    def verify_targets(self, aux, targets, codes):
        verify_tokens(aux, codes)
        assert codes.shape[1:] == (8, 8, 8) and torch.equal(targets, codes)
        return dict(soft_targets=False, stochastic_prefixes=False, coefficient_noise=False,
            coefficient_quantization='nearest of 128 uniform normalized bins',
            original_dictionary=True, atoms=16384, coefficient_bins=128, coefficient_bits=7,
            shape=list(codes.shape), training_and_evaluation_codec_identical=True,
            deterministic_source_support=True, final_least_squares_coefficients=True,
            auxiliary_loss_terms=0)

    def loss(self, model, logits, targets):
        atoms, coeff = logits['atom_logits'].float(), logits['coeff_logits'].float()
        atom_ce = F.cross_entropy(atoms.reshape(-1, atoms.shape[-1]), targets[..., 0::2].reshape(-1))
        coeff_ce = F.cross_entropy(coeff.reshape(-1, 128), (targets[..., 1::2] - 16384).reshape(-1))
        weight = float(self.recipe.atom_ce_weight)
        self.last_loss_metrics = dict(atom_ce=atom_ce.detach(), coefficient_ce=coeff_ce.detach())
        return (weight * atom_ce + coeff_ce) / (weight + 1)

    def sampling_settings(self):
        s = self.recipe.sampling
        return dict(temperature=float(s.atom_temperature), top_k=[int(s.atom_top_k)],
            top_p=[float(s.atom_top_p)], coefficient_temperature=float(s.coefficient_temperature),
            coefficient_top_k=int(s.coefficient_top_k), coefficient_top_p=float(s.coefficient_top_p))

    def target_temperature(self):
        return 0.

    def verify_resume(self, payload):
        if payload['recipe'] != OmegaConf.to_container(self.recipe, resolve=True):
            raise ValueError('Resume recipe differs from the saved compound128 experiment')

    def metadata(self):
        p = self.base / 'tokenizer-verification.json'
        measured = json.loads(p.read_text()) if p.exists() else {}
        return dict(architecture='native upstream RQ backbone with archived atom-conditioned compound coefficient head',
            stage1='original frozen LASER encoder, dictionary and decoder', original_dictionary=True,
            num_atoms=16384, coefficient_bins=128, coefficient_bits=7, scalar_events_per_site=8,
            compound_events_per_site=4, vocabulary=16512, codebook_sha256=self.codebook_sha256,
            objective=str(self.recipe.objective), atom_ce_weight=float(self.recipe.atom_ce_weight),
            coefficient_ce_weight=1., soft_labels=False, stochastic_codes=False,
            coefficient_noise=False, training_and_evaluation_codec_identical=True,
            auxiliary_loss_terms=0, source_stage1_rfid=float(self.recipe.tokenizer.source_rfid),
            measured_tokenizer_rfid=measured.get('measured_reconstruction_rfid',measured.get('measured_validation_rfid')),
            native_sampling=self.sampling_settings(), loss_component_diagnostics=True,
            attention_backend=self.recipe.get('execution', {}).get('attention_backend', 'native'),
            fused_adamw=bool(self.recipe.get('execution', {}).get('fused_adamw', False)))
