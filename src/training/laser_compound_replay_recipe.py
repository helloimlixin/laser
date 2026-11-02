"""Full-dictionary compound continuation; no geometry or trajectory losses."""
import json
import math
import torch
from torch.nn import functional as F
from omegaconf import OmegaConf

from src import rq_upstream
from src.training.rq_imagenet_recipes import LaserRecipe, RQRecipe
from src.training.laser_hard_ar import sparse_encoding, nearest_coefficient_ids, verify_tokens


class CompoundReplayRecipe(LaserRecipe):
    kind = 'laser_compound_replay'

    def __init__(self, base):
        super().__init__(base)
        assert self.recipe.decoder in ('archived_compound_pair', 'native_compound_pair')
        assert self.recipe.tokenizer.num_atoms == 16384 and self.recipe.tokenizer.coefficient_bins in (512, 2048)
        assert self.recipe.auxiliary_loss_terms == 0
        assert float(self.recipe.targets.temperature) > 0
        assert float(self.recipe.atom_ce_weight) > 0

    def model(self, arch):
        from src.models.laser_native_compound_adapter import build_native_compound_model
        initialization = self.recipe.get('initialization', {})
        reference = None
        if initialization.get('paired_fresh_backbone'):
            reference_arch = OmegaConf.create(OmegaConf.to_container(arch, resolve=True))
            reference_arch.vocab_size = int(initialization.reference_vocab_size)
            reference = RQRecipe.model(self, reference_arch)
        model = build_native_compound_model(arch, num_atoms=int(self.recipe.tokenizer.num_atoms),
            coefficient_bins=int(self.recipe.tokenizer.coefficient_bins), depth=4)
        if reference is not None:
            model.apply(model._init_weights)
            state = model.state_dict()
            with torch.no_grad():
                for name, value in reference.state_dict().items():
                    if name.startswith('classifier.linear.'):
                        value = value[:int(self.recipe.tokenizer.num_atoms)]
                    state[name].copy_(value)
        return model

    def adapt(self, model, aux):
        return None

    @torch.no_grad()
    def clean_components(self, aux, images):
        components = []
        with torch.autocast(device_type=images.device.type, enabled=False):
            if not hasattr(aux, '_compound_gram'):
                aux._compound_gram = aux.dictionary.float().T @ aux.dictionary.float()
            for chunk in images.split(int(self.recipe.targets.encoder_batch_size)):
                z = aux.quant_conv(aux.encoder(chunk.float())).permute(0, 2, 3, 1).contiguous()
                atoms, values = sparse_encoding(z, aux.dictionary, 4, gram=aux._compound_gram)
                normalized = values/aux.coeff_scales
                if self.recipe.targets.get('clip_centers', True):
                    normalized = normalized.clamp(-3., 3.)
                components.append((atoms, normalized))
        return tuple(torch.cat([v[i] for v in components]) for i in (0, 1))

    @torch.no_grad()
    def hard_codes(self, aux, images):
        atoms, values = self.clean_components(aux, images)
        codes = torch.empty(*atoms.shape[:-1], 8, device=images.device, dtype=torch.long)
        codes[..., 0::2] = atoms
        codes[..., 1::2] = nearest_coefficient_ids(values, aux.coeff_bins) + aux.num_atoms
        return codes

    @torch.no_grad()
    def targets(self, aux, images):
        atoms, values = self.clean_components(aux, images)
        probabilities = (-(values[..., None]-aux.coeff_bins).square()/float(self.recipe.targets.temperature)).softmax(-1)
        ids = torch.multinomial(probabilities.reshape(-1, aux.coeff_vocab_size), 1).reshape_as(atoms)
        contexts = torch.empty(*atoms.shape[:-1], 8, device=atoms.device, dtype=torch.long)
        contexts[..., 0::2], contexts[..., 1::2] = atoms, ids + aux.num_atoms
        return dict(atoms=atoms, coefficients=values, coefficient_probabilities=probabilities), contexts

    def verify_targets(self, aux, targets, codes):
        verify_tokens(aux, codes)
        p = targets['coefficient_probabilities']
        assert torch.isfinite(p).all() and p.dtype == torch.float32 and not p.requires_grad
        torch.testing.assert_close(p.sum(-1), torch.ones_like(p[..., 0]), rtol=2.e-5, atol=2.e-5)
        assert torch.equal(codes[..., 0::2], targets['atoms'])
        return dict(soft_targets=True, stochastic_coefficient_prefixes=True,
            deterministic_source_support=True, original_dictionary=True,
            coefficient_temperature=float(self.recipe.targets.temperature),
            normalized_sigma=math.sqrt(float(self.recipe.targets.temperature)/2), coefficient_bins=aux.coeff_vocab_size,
            shape=list(codes.shape), sampled_trajectories=0, auxiliary_loss_terms=0,
            target_centers=('continuous clipped source coefficients before bin rounding'
                if self.recipe.targets.get('clip_centers', True) else
                'continuous unclipped source coefficients before bin rounding'))

    def loss(self, model, logits, targets):
        atoms, coefficients = logits['atom_logits'].float(), logits['coeff_logits'].float()
        a = F.cross_entropy(atoms.reshape(-1, atoms.shape[-1]), targets['atoms'].reshape(-1))
        logp = coefficients.log_softmax(-1)
        c = -(targets['coefficient_probabilities']*logp).sum(-1).mean()
        weight = float(self.recipe.atom_ce_weight)
        self.last_loss_metrics = {'atom_ce':a.detach(), 'coefficient_ce':c.detach()}
        return (weight*a+c)/(weight+1.)

    def metadata(self):
        receipt = self.base/'tokenizer-verification.json'
        measured = json.loads(receipt.read_text()) if receipt.exists() else {}
        return dict(architecture='upstream RQ forward/cache plus checkpoint-compatible LASER compound head',
            stage1='original frozen LASER encoder, 16384-atom dictionary and decoder',
            source_stage1_rfid=float(self.recipe.tokenizer.source_rfid),
            coefficient_bins=int(self.recipe.tokenizer.coefficient_bins), scalar_events_per_site=8, compound_events_per_site=4,
            objective=str(self.recipe.get('objective', 'historical_weighted_atom_and_soft_coefficient_cross_entropy')),
            atom_ce_weight=float(self.recipe.atom_ce_weight), coefficient_ce_weight=1., soft_labels=True,
            coefficient_temperature=float(self.recipe.targets.temperature),
            coefficient_noise_sigma_normalized=math.sqrt(float(self.recipe.targets.temperature)/2),
            original_dictionary=True, auxiliary_loss_terms=0, geometry_weight=0.,
            sampled_trajectories=0, native_sampling=OmegaConf.to_container(self.recipe.sampling, resolve=True),
            precision='native FP16 AMP', encoder_precision='FP32',
            measured_tokenizer_rfid=measured.get('measured_validation_rfid'),
            initialization=OmegaConf.to_container(self.recipe.initialization, resolve=True)
                if self.recipe.get('initialization') is not None else {},
            shared_upstream=rq_upstream.provenance())
