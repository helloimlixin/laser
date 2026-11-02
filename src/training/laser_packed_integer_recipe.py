"""Literal full-pair integer vocabulary on the unmodified upstream prior."""
import hashlib

import torch
from omegaconf import OmegaConf

from src import rq_upstream
from src.models.laser_packed_integer_tokenizer import pack_pair_ids, unpack_pair_ids
from src.training.rq_imagenet_recipes import RQRecipe
from src.training.packed_pair_loss import sparse_pair_cross_entropy


class PackedIntegerLaserRecipe(RQRecipe):
    kind = 'laser_packed_integer'
    shape = (8, 8, 4)
    checkpoint_key = 'stage1_checkpoint_sha256'

    def __init__(self, base):
        super().__init__(base)
        self.recipe = OmegaConf.load(self.base / 'inputs/recipe.yaml')
        if self.recipe.decoder != 'native_packed_pair_integer' or self.recipe.auxiliary_loss_terms != 0:
            raise ValueError('Packed integer recipe requires the native classifier and no auxiliary losses')
        t = self.recipe.tokenizer
        self.vocabulary = int(t.num_atoms) * int(t.coefficient_bins)
        if self.recipe.get('execution',{}).get('loss_backend','torch_chunked') != 'torch_chunked':
            raise ValueError('Only the verified Torch global-softmax backend is enabled')
        if int(t.depth) != 4 or int(t.vocab_size) != self.vocabulary:
            raise ValueError('Packed integer recipe must retain every atom/coefficient pair')
        self.run_id, self.name = str(self.recipe.run_id), str(self.recipe.name)
        self.microbatch, self.epochs = int(self.recipe.microbatch), int(self.recipe.epochs)
        self.schedule_epochs = int(self.recipe.lr_schedule_epochs)
        self.generation_batch = int(self.recipe.evaluation.batch_size)

    def tokenizer(self, device):
        from src.models.laser_packed_integer_tokenizer import FrozenPackedIntegerLaserTokenizer
        t = self.recipe.tokenizer
        return FrozenPackedIntegerLaserTokenizer(self.base / 'inputs/stage1-model.pt',
            num_atoms=int(t.num_atoms), coeff_vocab_size=int(t.coefficient_bins),
            coeff_max=float(t.coefficient_max), coeff_scales=list(t.coefficient_scales),
            sparsity_level=int(t.depth)).to(device), None

    def model(self, arch):
        arch = OmegaConf.create(OmegaConf.to_container(arch, resolve=True))
        if not (arch.input_emb_vqvae and arch.head_emb_vqvae and arch.shared_cls_emb):
            raise ValueError('Use the original physical code embeddings and shared dense classifier')
        arch.block_size = [*arch.block_size[:2], 4]
        arch.vocab_size = self.vocabulary
        execution = self.recipe.get('execution', {})
        classifier = execution.get('classifier', 'native_dense')
        if classifier == 'projected_global_pair_ids':
            from src.models.laser_pair_id_head import ProjectedPairIDLinear
            # Build only a small temporary classifier; never instantiate the
            # 12.9B dense matrix as an intermediate allocation.
            initialization = self.recipe.get('initialization', {})
            arch.vocab_size = int(initialization.get('reference_vocab_size',self.recipe.tokenizer.num_atoms))
            model = RQRecipe.model(self, arch)
            model.classifier.linear = ProjectedPairIDLinear(int(arch.embed_dim), self.vocabulary,
                int(execution.projection_dim),
                chunked_training_loss=bool(execution.get('chunked_training_loss', False)))
            model.classifier.linear.apply(model._init_weights)
            model.config.vocab_size = [self.vocabulary] * 4
            model.vocab_size = model.config.vocab_size
            mask = model.classifier.logit_mask
            model.classifier.logit_mask = type(mask)(model.vocab_size, value=mask.value)
            assert type(model) is rq_upstream.RQTransformer
            return model
        if classifier != 'native_dense':
            raise ValueError('Unknown pair-ID classifier')
        # Refuse a huge real allocation in the existing replicated driver.
        # Meta construction is safe and is also used by the resource audit.
        if torch.empty(0).device.type != 'meta' and self.vocabulary > 131072:
            raise RuntimeError('Full-pair dense vocabulary requires a sharded execution backend; '
                'run audit_packed_integer_rq.py for exact parameter/logit costs. '
                'The existing replicated driver is intentionally blocked before allocation.')
        model = RQRecipe.model(self, arch)
        assert type(model) is rq_upstream.RQTransformer
        return model

    @torch.no_grad()
    def targets(self, aux, images):
        from src.training.laser_hard_ar import nearest_coefficient_ids
        atoms, values = aux.encode_components(images, int(self.recipe.targets.encoder_batch_size))
        if self.recipe.targets.mode == 'hard':
            bins = nearest_coefficient_ids(values, aux.coeff_bins)
            codes = pack_pair_ids(atoms, bins, aux.num_atoms, aux.coeff_vocab_size)
            return codes, codes
        if self.recipe.targets.mode != 'soft_coefficient':
            raise ValueError('Unknown packed integer target mode')
        probabilities = (-(values[..., None] - aux.coeff_bins).square()
                         / float(self.recipe.targets.temperature)).softmax(-1)
        bins = torch.multinomial(probabilities.reshape(-1, aux.coeff_vocab_size), 1).reshape_as(atoms)
        codes = pack_pair_ids(atoms, bins, aux.num_atoms, aux.coeff_vocab_size)
        return dict(atoms=atoms, coefficient_probabilities=probabilities), codes

    def verify_targets(self, aux, targets, codes):
        assert codes.shape[1:] == self.shape
        atoms, _ = unpack_pair_ids(codes, aux.num_atoms, aux.coeff_vocab_size)
        if isinstance(targets, dict):
            assert torch.equal(atoms, targets['atoms'])
            p = targets['coefficient_probabilities']
            assert not p.requires_grad and torch.isfinite(p).all() and (p >= 0).all()
            torch.testing.assert_close(p.sum(-1), torch.ones_like(p[..., 0]), rtol=2.e-5, atol=2.e-5)
        else:
            assert torch.equal(targets, codes)
        return dict(integer_joint_ids=True, vocabulary=self.vocabulary,
            full_cartesian_vocabulary=True, compound_events_per_site=4,
            soft_targets=isinstance(targets, dict), auxiliary_loss_terms=0, sampled_trajectories=0)

    def loss(self, model, logits, targets):
        execution = self.recipe.get('execution', {})
        if execution.get('chunked_training_loss', False) and model.training:
            from src.training.pair_id_linear_loss import chunked_pair_linear_cross_entropy
            head = model.classifier.linear.classes
            if isinstance(targets, dict):
                p = targets['coefficient_probabilities']
                ids = targets['atoms'][...,None] * p.shape[-1] + torch.arange(p.shape[-1],device=p.device)
            else:
                ids, p = targets[...,None], torch.ones_like(targets[...,None],dtype=torch.float32)
            return chunked_pair_linear_cross_entropy(logits,head.weight,head.bias,ids,p,
                row_chunk=int(execution.loss_row_chunk_size),
                class_chunk=int(execution.loss_class_chunk_size))
        if not isinstance(targets, dict):
            return model.compute_loss(logits.float(), targets, use_soft_target=False)
        return sparse_pair_cross_entropy(logits, targets['atoms'], targets['coefficient_probabilities'],
            row_chunk_size=int(self.recipe.targets.loss_row_chunk_size))

    def sampling_settings(self):
        s = self.recipe.sampling
        return dict(temperature=float(s.temperature),
            top_k=[int(s.top_k) or self.vocabulary], top_p=[float(s.top_p)])

    @torch.no_grad()
    def sample(self, model, aux, count, labels):
        # The global categorical has 8M classes. Bound logits and native top-p
        # sorting memory while retaining exactly the original sampler.
        from src.models.laser_pair_id_head import skip_redundant_native_topk
        with skip_redundant_native_topk(model):
            return torch.cat([RQRecipe.sample(self, model, aux, len(part), part)
                for part in labels.split(self.generation_batch)])

    def digest(self, aux):
        digest = hashlib.sha256()
        for name, value in aux.state_dict().items():
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def metadata(self):
        execution = self.recipe.get('execution', {})
        projected = execution.get('classifier', 'native_dense') == 'projected_global_pair_ids'
        return dict(architecture=('upstream RQTransformer with projected global pair-ID classifier'
                if projected else 'unmodified upstream RQTransformer'), integer_joint_ids=True,
            pair_id='atom_id * coefficient_bins + coefficient_bin',
            vocabulary=self.vocabulary, full_cartesian_vocabulary=True,
            objective='single_global_pair_cross_entropy', compound_events_per_site=4,
            separate_atom_or_coefficient_heads=False, auxiliary_loss_terms=0, sampled_trajectories=0,
            classifier=execution.get('classifier', 'native_dense'),
            output_projection_dim=int(execution.get('projection_dim', 1536)),
            exact_chunked_global_softmax=bool(execution.get('chunked_training_loss', False)),
            loss_backend=execution.get('loss_backend','torch_chunked'),
            initialization=OmegaConf.to_container(self.recipe.initialization,resolve=True)
                if self.recipe.get('initialization') is not None else {},
            native_sampling=self.sampling_settings(), shared_upstream=rq_upstream.provenance())

    def checkpoint_extra(self):
        return dict(recipe=OmegaConf.to_container(self.recipe, resolve=True))

    def verify_resume(self, payload):
        assert payload['recipe'] == OmegaConf.to_container(self.recipe, resolve=True)
