"""Hard support-plus-pattern targets from the independently verified 1K codec."""
import hashlib
import json
import math

import torch
import torch.distributed as dist
from torch.nn import functional as F
from omegaconf import OmegaConf

from src.training.rq_imagenet_recipes import LaserRecipe
from runtime.compound_recipe import Compound128Recipe
from runtime.pattern_model import build_model, pack, unpack, verify_codes
from runtime.pattern_codec import assign_coefficient_patterns, reconstruct_pattern_latents, selected_support_grams


class PatternRecipe(LaserRecipe):
    shape = (8, 8, 4)
    kind = 'laser_pattern1024'
    clean_components = Compound128Recipe.clean_components

    def __init__(self, base):
        super().__init__(base)
        assert self.recipe.tokenizer.num_atoms == 16384
        assert self.recipe.tokenizer.coefficient_patterns == 1024
        self.codebook_sha256 = str(self.recipe.tokenizer.pattern_codebook_sha256)

    def tokenizer(self, device):
        aux, _ = LaserRecipe.tokenizer(self, device)
        path = self.base/'inputs/coefficient-patterns.pt'
        with path.open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == self.codebook_sha256
        payload = torch.load(path, map_location='cpu', weights_only=False)
        patterns = payload['patterns']
        assert patterns.shape == (1024, 4) and torch.isfinite(patterns).all()
        aux.register_buffer('coefficient_patterns', patterns.to(device))
        aux.pattern_count = 1024
        return aux, None

    def model(self, arch):
        return build_model(arch)

    def adapt(self, model, aux):
        return None

    @torch.no_grad()
    def targets(self, aux, images):
        atoms, normalized = self.clean_components(aux, images)
        physical = normalized*aux.coeff_scales
        grams = selected_support_grams(atoms.reshape(-1, 4), aux.dictionary.T)
        ids, _ = assign_coefficient_patterns(physical.reshape(-1, 4), aux.coefficient_patterns,
            grams=grams, chunk_size=1024)
        ids = ids.reshape(atoms.shape[:-1])
        return dict(atoms=atoms, patterns=ids), pack(atoms, ids, 1024)

    def verify_codes(self, aux, codes):
        verify_codes(codes)

    def verify_targets(self, aux, targets, codes):
        self.verify_codes(aux, codes)
        atoms, patterns = unpack(codes, 1024)
        assert codes.shape[1:] == self.shape
        assert torch.equal(atoms, targets['atoms']) and torch.equal(patterns, targets['patterns'])
        return dict(passed=True, atoms=16384, coefficient_patterns=1024, shape=list(codes.shape),
            categorical_events_per_site=5, native_depth_steps=4, deterministic_source_support=True,
            coefficient_pattern_assignment='nearest in full selected-support physical latent geometry',
            hard_labels=True, stochastic_prefixes=False, teacher_entropy_nats=0.,
            prefixes_from_same_codec_as_targets=True, coefficient_noise=False)

    def loss(self, model, logits, targets):
        atom = logits['atom_logits'].float()
        pattern = logits['pattern_logits'].float()
        a = F.cross_entropy(atom.reshape(-1, 16384), targets['atoms'].reshape(-1))
        p = F.cross_entropy(pattern.reshape(-1, 1024), targets['patterns'].reshape(-1))
        with torch.no_grad():
            logp = pattern.log_softmax(-1)
            entropy = -(logp.exp()*logp).sum(-1).mean()/math.log(2.)
        self.last_loss_metrics = dict(atom_ce=a.detach(), pattern_ce=p.detach(),
            joint_nll_bits_per_image=64*(4*a.detach()+p.detach())/math.log(2.),
            pattern_predicted_entropy_bits=entropy, teacher_entropy_nats=entropy.new_zeros(()))
        return (4*a+p)/5

    @torch.no_grad()
    def decode(self, aux, codes):
        atoms, patterns = unpack(codes, 1024)
        latent = reconstruct_pattern_latents(atoms, patterns, aux.coefficient_patterns, aux.dictionary.T)
        return aux.decoder(aux.post_quant_conv(latent.permute(0, 3, 1, 2).contiguous())).clamp(-1, 1)

    def sampling_settings(self):
        s = self.recipe.sampling
        return dict(temperature=float(s.atom_temperature), top_k=[int(s.atom_top_k)],
            top_p=[float(s.atom_top_p)], pattern_temperature=float(s.pattern_temperature),
            pattern_top_k=int(s.pattern_top_k), pattern_top_p=float(s.pattern_top_p))

    def sample(self, model, aux, count, labels):
        s = self.recipe.sampling
        return model.sample_patterns(count, aux, labels, atom_temperature=float(s.atom_temperature),
            atom_top_k=int(s.atom_top_k), atom_top_p=float(s.atom_top_p),
            pattern_temperature=float(s.pattern_temperature), pattern_top_k=int(s.pattern_top_k),
            pattern_top_p=float(s.pattern_top_p), amp=True)

    def target_temperature(self):
        return 0.

    def verify_resume(self, payload):
        assert payload['recipe']['tokenizer'] == OmegaConf.to_container(self.recipe.tokenizer, resolve=True)
        assert payload['recipe']['target_mode'] == 'hard_support_and_pattern'

    def metadata(self):
        return dict(architecture='native RQ spatial and four-step depth stacks plus ordered-support pattern MLP',
            stage1='original frozen LASER encoder, atom dictionary and image decoder',
            num_atoms=16384, coefficient_patterns=1024, vocabulary=16384, pattern_vocabulary=1024,
            categorical_events_per_site=5, native_depth_steps=4,
            codebook_sha256=self.codebook_sha256, original_dictionary=True,
            objective='joint support-plus-pattern categorical NLL, averaged over five decisions',
            atom_loss_weight=4., pattern_loss_weight=1., auxiliary_loss_terms=0,
            soft_labels=False, stochastic_codes=False, teacher_entropy_nats=0.,
            measured_tokenizer_rfid=4.297528190759465, native_sampling=self.sampling_settings(),
            separate_spatial_and_depth_embeddings=True, current_pattern_excluded_from_atom_predictions=True,
            native_patch='separate depth-embedding hook and full-forward classification hook; defaults preserve native RQ')
