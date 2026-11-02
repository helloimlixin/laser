"""Four native RQ depth steps with a compact joint categorical classifier."""
import hashlib

import torch
from omegaconf import OmegaConf

from src import rq_upstream
from src.training.rq_imagenet_recipes import RQRecipe


class FineJointLaserRecipe(RQRecipe):
    kind = 'laser_fine_joint'
    decoder = 'conditional_joint_integer'
    shape = (8, 8, 4)
    checkpoint_key = 'stage1_checkpoint_sha256'

    def __init__(self, base):
        super().__init__(base)
        self.recipe = OmegaConf.load(self.base/'inputs/recipe.yaml')
        assert self.recipe.decoder == self.decoder
        assert self.recipe.auxiliary_loss_terms == 0
        self.run_id, self.name = str(self.recipe.run_id), str(self.recipe.name)
        self.microbatch, self.epochs = int(self.recipe.microbatch), int(self.recipe.epochs)
        self.schedule_epochs = int(self.recipe.lr_schedule_epochs)
        self.generation_batch = int(self.recipe.evaluation.batch_size)

    def tokenizer(self, device):
        from src.models.laser_fine_joint_tokenizer import FrozenFineJointTokenizer
        t = self.recipe.tokenizer
        return FrozenFineJointTokenizer(self.base/'inputs/stage1-model.pt',
            num_atoms=int(t.num_atoms), coeff_vocab_size=int(t.coefficient_bins),
            coeff_max=float(t.coefficient_max), coeff_scales=list(t.coefficient_scales),
            sparsity_level=int(t.depth)).to(device), None

    def model(self, arch):
        from src.models.laser_fine_joint_adapter import install_conditional_joint_head
        model = super().model(arch)
        assert type(model) is rq_upstream.RQTransformer
        assert model.config.input_emb_vqvae and model.config.head_emb_vqvae
        return install_conditional_joint_head(model, int(self.recipe.tokenizer.coefficient_bins),
                                              int(self.recipe.head.context_width))

    def adapt(self, model, aux):
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self.coefficient_generator = torch.Generator(device=aux.dictionary.device).manual_seed(261015+rank)

    @torch.no_grad()
    def targets(self, aux, images):
        all_codes, all_gamma = [], []
        with torch.autocast(device_type=images.device.type, enabled=False):
            for chunk in images.split(int(self.recipe.targets.encoder_batch_size)):
                codes, gamma = aux.latent_codes(aux.encode(chunk))
                all_codes.append(codes); all_gamma.append(gamma)
            hard, gamma = torch.cat(all_codes), torch.cat(all_gamma)
            atoms = hard // aux.coeff_vocab_size
            values = aux.coeff_scales[:, None] * aux.coeff_bins[None, :]
            p = (-(gamma[..., None]-values).square() /
                 (2*float(self.recipe.targets.physical_sigma)**2)).softmax(-1)
            bins = torch.multinomial(p.reshape(-1, aux.coeff_vocab_size), 1,
                                    generator=self.coefficient_generator).reshape_as(hard)
            sampled = atoms * aux.coeff_vocab_size + bins
        return dict(atoms=atoms, coefficient_probabilities=p, hard_codes=hard), sampled

    def verify_targets(self, aux, targets, codes):
        assert codes.shape[1:] == self.shape and codes.dtype == torch.long
        assert ((codes >= 0) & (codes < aux.vocab_size)).all()
        atoms, p = targets['atoms'], targets['coefficient_probabilities']
        assert torch.equal(atoms, codes // aux.coeff_vocab_size)
        assert (atoms.sort(-1).values.diff(dim=-1) > 0).all()
        assert p.dtype == torch.float32 and not p.requires_grad and torch.isfinite(p).all()
        torch.testing.assert_close(p.sum(-1), torch.ones_like(p[..., 0]), rtol=2e-5, atol=2e-5)
        return dict(integer_joint_ids=True, shape=list(codes.shape),
            semantic_vocabulary=aux.vocab_size, coefficient_bins=aux.coeff_vocab_size,
            soft_targets=True, stochastic_prefixes=True, sampled_trajectories=0,
            physical_sigma=float(self.recipe.targets.physical_sigma), auxiliary_loss_terms=0)

    def loss(self, model, features, targets):
        from src.models.laser_fine_joint_adapter import joint_cross_entropy
        return joint_cross_entropy(model.classifier.linear, features, targets)

    def sampling_settings(self):
        s = self.recipe.sampling
        return dict(temperature=float(s.atom_temperature), top_k=[int(s.atom_top_k)],
                    top_p=[float(s.atom_top_p)])

    @torch.no_grad()
    def sample(self, model, aux, count, labels):
        from rqvae.utils.utils import sample_from_logits
        s, head = self.recipe.sampling, model.classifier.linear
        codes = torch.zeros(count, *self.shape, dtype=torch.long, device=labels.device)
        model.init_cache()
        for h in range(8):
            for w in range(8):
                for d in range(4):
                    features = model.cached_forward(codes[:, :h+1], aux, labels, amp=True,
                                                    sample_loc=(h, w, d))
                    logits = features[:, :aux.num_atoms].float().clone()
                    if d:
                        logits.scatter_(-1, codes[:, h, w, :d] // aux.coeff_vocab_size, -torch.inf)
                    atoms = sample_from_logits(logits, temperature=float(s.atom_temperature),
                        top_k=int(s.atom_top_k) or None, top_p=float(s.atom_top_p)).reshape(count)
                    coefficient_logits = head.coefficient_logits(features, atoms)
                    bins = sample_from_logits(coefficient_logits.float(),
                        temperature=float(s.coefficient_temperature),
                        top_k=int(s.coefficient_top_k) or None,
                        top_p=float(s.coefficient_top_p)).reshape(count)
                    codes[:, h, w, d] = atoms * aux.coeff_vocab_size + bins
        model.init_cache()
        return codes

    def digest(self, aux):
        digest = hashlib.sha256()
        for name, value in aux.state_dict().items():
            digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def metadata(self):
        t = self.recipe.tokenizer
        return dict(architecture='upstream RQTransformer with compact conditional joint classifier',
            integer_joint_ids=True, semantic_vocabulary=int(t.num_atoms)*int(t.coefficient_bins),
            atom_classifier_width=int(t.num_atoms), coefficient_bins=int(t.coefficient_bins),
            scalar_events_per_site=4, objective='joint_categorical_cross_entropy',
            joint_probability='p(atom|prefix)*p(coefficient|atom,prefix)',
            auxiliary_loss_terms=0, sampled_trajectories=0, stochastic_codes=True, soft_labels=True,
            coefficient_coordinates='prefix_stable_orthogonal',
            coefficient_sigma=float(self.recipe.targets.physical_sigma),
            coefficient_posterior_center='continuous_source_coefficient',
            native_sampling=OmegaConf.to_container(self.recipe.sampling),
            shared_upstream=rq_upstream.provenance())

    def checkpoint_extra(self):
        return dict(recipe=OmegaConf.to_container(self.recipe, resolve=True))

    def verify_resume(self, payload):
        assert payload['recipe'] == OmegaConf.to_container(self.recipe, resolve=True)
