"""Native RQTransformer with one categorical event per calibrated LASER pair."""
import hashlib

import torch
from omegaconf import OmegaConf

from src import rq_upstream
from src.training.rq_imagenet_recipes import RQRecipe
from src.training.rq_imagenet_runtime import sha256


class CatalogIntegerLaserRecipe(RQRecipe):
    kind = 'laser_catalog_integer'
    shape = (8, 8, 4)
    checkpoint_key = 'stage1_checkpoint_sha256'

    def __init__(self, base):
        super().__init__(base)
        self.recipe = OmegaConf.load(self.base/'inputs/recipe.yaml')
        assert self.recipe.decoder == 'native_catalog_pair_integer' and self.recipe.auxiliary_loss_terms == 0
        self.run_id, self.name = str(self.recipe.run_id), str(self.recipe.name)
        self.microbatch, self.epochs = int(self.recipe.microbatch), int(self.recipe.epochs)
        self.schedule_epochs = int(self.recipe.lr_schedule_epochs)
        self.generation_batch = int(self.recipe.evaluation.batch_size)
        self.vocabulary = int(self.recipe.tokenizer.vocab_size)
        assert self.vocabulary == int(self.recipe.tokenizer.num_atoms)*int(self.recipe.tokenizer.levels_per_atom)
        self.catalog_sha256 = sha256(self.base/'inputs/pair-catalog.pt')
        assert self.catalog_sha256 == self.recipe.tokenizer.catalog_sha256

    def tokenizer(self, device):
        from src.models.laser_catalog_integer_tokenizer import FrozenCatalogIntegerLaserTokenizer
        return FrozenCatalogIntegerLaserTokenizer(self.base/'inputs/stage1-model.pt',
            self.base/'inputs/pair-catalog.pt').to(device), None

    def model(self, arch):
        arch = OmegaConf.create(OmegaConf.to_container(arch, resolve=True))
        assert arch.input_emb_vqvae and arch.head_emb_vqvae and arch.shared_cls_emb
        arch.block_size = [*arch.block_size[:2], 4]
        arch.vocab_size = int(self.recipe.initialization.reference_vocab_size)
        model = RQRecipe.model(self, arch)
        old = model.classifier.linear
        model.classifier.linear = torch.nn.Linear(old.in_features, self.vocabulary,
            bias=old.bias is not None, device=old.weight.device, dtype=old.weight.dtype)
        model.classifier.linear.apply(model._init_weights)
        model.config.vocab_size = [self.vocabulary]*4
        model.vocab_size = model.config.vocab_size
        mask = model.classifier.logit_mask
        model.classifier.logit_mask = type(mask)(model.vocab_size, value=mask.value)
        assert type(model) is rq_upstream.RQTransformer
        return model

    @torch.no_grad()
    def targets(self, aux, images):
        atoms, values = aux.encode_components(images, int(self.recipe.targets.encoder_batch_size))
        levels = aux.coefficient_values(atoms)
        probabilities = (-(values[..., None]-levels).square()/float(self.recipe.targets.temperature)).softmax(-1)
        slots = torch.multinomial(probabilities.reshape(-1, aux.levels_per_atom), 1).reshape_as(atoms)
        return dict(atoms=atoms, coefficient_probabilities=probabilities), atoms*aux.levels_per_atom+slots

    def verify_targets(self, aux, targets, codes):
        atoms, _ = aux.unpack_catalog(codes)
        p = targets['coefficient_probabilities']
        assert codes.shape[1:] == self.shape and torch.equal(atoms, targets['atoms'])
        assert not p.requires_grad and torch.isfinite(p).all() and (p >= 0).all()
        torch.testing.assert_close(p.sum(-1), torch.ones_like(p[..., 0]), rtol=2.e-5, atol=2.e-5)
        return dict(integer_joint_ids=True, vocabulary=self.vocabulary,
            coefficient_grid_bins=512, catalog_levels_per_atom=aux.levels_per_atom,
            source_omp_support_unchanged=True, source_final_ls_coefficients=True,
            soft_targets=True, auxiliary_loss_terms=0, sampled_trajectories=0)

    def loss(self, model, logits, targets):
        from src.training.packed_pair_loss import sparse_pair_cross_entropy
        return sparse_pair_cross_entropy(logits, targets['atoms'], targets['coefficient_probabilities'],
            row_chunk_size=int(self.recipe.targets.loss_row_chunk_size))

    def sampling_settings(self):
        s = self.recipe.sampling
        return dict(temperature=float(s.temperature), top_k=[int(s.top_k) or self.vocabulary],
                    top_p=[float(s.top_p)])

    @torch.no_grad()
    def sample(self, model, aux, count, labels):
        from src.models.laser_pair_id_head import skip_redundant_native_topk
        with skip_redundant_native_topk(model):
            return torch.cat([RQRecipe.sample(self, model, aux, len(part), part)
                              for part in labels.split(self.generation_batch)])

    def metadata(self):
        return dict(architecture='unmodified upstream RQTransformer',
            integer_joint_ids=True, vocabulary=self.vocabulary, compound_events_per_site=4,
            objective='single_global_pair_cross_entropy', auxiliary_loss_terms=0, sampled_trajectories=0,
            source_omp_support_unchanged=True, source_final_ls_coefficients=True,
            coefficient_grid_bins=512, catalog_levels_per_atom=int(self.recipe.tokenizer.levels_per_atom),
            full_cartesian_vocabulary=False, catalog_sha256=self.catalog_sha256,
            measured_tokenizer_rfid=float(self.recipe.tokenizer.measured_rfid),
            native_sampling=self.sampling_settings(), shared_upstream=rq_upstream.provenance())

    def digest(self, aux):
        digest = hashlib.sha256()
        for name, value in aux.state_dict().items():
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def checkpoint_extra(self):
        return dict(recipe=OmegaConf.to_container(self.recipe, resolve=True),
                    pair_catalog_sha256=self.catalog_sha256)

    def verify_resume(self, payload):
        assert payload['recipe'] == OmegaConf.to_container(self.recipe, resolve=True)
        assert payload['pair_catalog_sha256'] == self.catalog_sha256
