"""Finite LASER joint IDs with the unchanged upstream RQTransformer."""
import hashlib
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src import rq_upstream
from src.training.rq_imagenet_recipes import RQRecipe
from src.training.rq_imagenet_runtime import sha256


class JointLaserRecipe(RQRecipe):
    kind = 'laser_joint'
    shape = (8, 8, 4)
    encoder_batch_size = 8
    checkpoint_key = 'stage1_checkpoint_sha256'

    def __init__(self, base):
        super().__init__(base)
        self.recipe = OmegaConf.load(self.base/'inputs/recipe.yaml')
        assert self.recipe.decoder == 'native_joint_integer' and self.recipe.auxiliary_loss_terms == 0
        self.run_id, self.name = str(self.recipe.run_id), str(self.recipe.name)
        self.shape = (8, 8, int(self.recipe.tokenizer.get('depth', 4)))
        self.microbatch, self.epochs = int(self.recipe.microbatch), int(self.recipe.epochs)
        self.schedule_epochs = int(self.recipe.lr_schedule_epochs)
        self.generation_batch = int(self.recipe.evaluation.batch_size)
        self.codebook_sha256 = sha256(self.base/'inputs/joint-codebook.pt')
        assert self.codebook_sha256 == self.recipe.tokenizer.codebook_sha256
        execution_path = self.base/'inputs/execution.json'
        execution = json.loads(execution_path.read_text()) if execution_path.exists() else {}
        self.execution_options = execution
        self.encoder_batch_size = int(execution.get('encoder_batch_size', 8))
        if self.encoder_batch_size < 1:
            raise ValueError('Encoder batch size must be positive')
        self.generation_batch = int(execution.get('generation_batch_size', self.generation_batch))
        if self.generation_batch < 1:
            raise ValueError('Generation batch size must be positive')
        policy = self.target_policy()
        if policy not in ('soft_stochastic', 'hard_deterministic'):
            raise ValueError(f'Unknown joint target policy: {policy}')
        if bool(self.recipe.targets.get('stochastic_prefixes', True)) != (policy == 'soft_stochastic'):
            raise ValueError('Target policy and stochastic_prefixes must agree')
        self.depth_loss_weights()  # Validate opt-in experimental weights before allocating a model.

    def tokenizer(self, device):
        from src.models.laser_joint_tokenizer import FrozenJointLaserTokenizer
        return FrozenJointLaserTokenizer(self.base/'inputs/stage1-model.pt',
                                         self.base/'inputs/joint-codebook.pt').to(device), None

    def model(self, arch):
        arch = OmegaConf.create(OmegaConf.to_container(arch, resolve=True))
        vocabulary = int(self.recipe.tokenizer.vocab_size)
        arch.block_size[-1] = int(self.recipe.tokenizer.get('depth', arch.block_size[-1]))
        initialization = self.recipe.get('initialization', {})
        reference = int(initialization.get('reference_vocab_size', vocabulary))
        paired = initialization.get('paired_fresh_backbone', False)
        if reference != vocabulary and not paired:
            raise ValueError('Reference vocabulary initialization requires paired_fresh_backbone')
        if paired and not (arch.input_emb_vqvae and arch.head_emb_vqvae and arch.shared_cls_emb):
            raise ValueError('Paired initialization requires native physical input/head embeddings and shared classifier')
        arch.vocab_size = reference
        model = super().model(arch)
        if reference != vocabulary:
            previous = model.classifier.linear
            classifier = torch.nn.Linear(previous.in_features, vocabulary,
                bias=previous.bias is not None, device=previous.weight.device, dtype=previous.weight.dtype)
            classifier.apply(model._init_weights)
            model.classifier.linear = classifier
            model.config.vocab_size = [vocabulary] * model.block_size[-1]
            model.vocab_size = model.config.vocab_size
            mask = model.classifier.logit_mask
            model.classifier.logit_mask = type(mask)(model.vocab_size, value=mask.value)
        assert type(model) is rq_upstream.RQTransformer
        return model

    def adapt(self, model, aux):
        path = self.base/'inputs/execution.json'
        if path.exists():
            options = json.loads(path.read_text())
            if options.get('align_classifier'):
                from src.models.laser_joint_execution import align_joint_classifier
                align_joint_classifier(model, multiple=int(options.get('alignment_multiple', 64)))
            if options.get('geometry_shared_head', False):
                from src.models.geometric_joint_adapter import add_geometric_joint_classifier
                add_geometric_joint_classifier(model, aux.quantizer)
        return None

    @torch.no_grad()
    def targets(self, aux, images):
        with torch.autocast(device_type=images.device.type, enabled=False):
            latents = torch.cat([aux.encode(chunk.float()) for chunk in images.split(self.encoder_batch_size)])
            if self.target_policy() == 'hard_deterministic':
                codes = aux.quantizer.quantize(latents,
                    chunk_size=int(self.recipe.targets.chunk_size))['codes']
                # Keep the existing exact, chunked CE implementation. Allocate
                # FP32 directly rather than constructing a large int64 one-hot.
                targets = latents.new_zeros(*codes.shape, aux.quantizer.vocab_size)
                targets.scatter_(-1, codes.unsqueeze(-1), 1.)
                return targets, codes
            return aux.quantizer.get_soft_codes(latents, temp=self.target_temperature(),
                stochastic=True, chunk_size=int(self.recipe.targets.chunk_size))

    def target_policy(self):
        return str(self.recipe.targets.get('policy', 'soft_stochastic'))

    def target_temperature(self):
        value = self.recipe.targets.temperature
        return [float(t) for t in value] if OmegaConf.is_list(value) else float(value)

    def verify_targets(self, aux, targets, codes):
        assert codes.shape[1:] == self.shape and codes.dtype == torch.long
        assert targets.shape == (*codes.shape, aux.quantizer.vocab_size)
        assert targets.dtype == torch.float32 and not targets.requires_grad
        assert ((codes >= 0) & (codes < aux.quantizer.vocab_size)).all()
        # Bound the normalization audit's intermediate allocation as well.
        flat = targets.reshape(-1, targets.shape[-1])
        entropy = 0.
        for chunk in flat.split(128):
            assert torch.isfinite(chunk).all() and (chunk >= 0).all()
            torch.testing.assert_close(chunk.sum(-1), torch.ones_like(chunk[:, 0]), rtol=2.e-5, atol=2.e-5)
            entropy += float(-(chunk * chunk.clamp_min(1.e-30).log()).sum())
        soft = self.target_policy() == 'soft_stochastic'
        if not soft:
            assert entropy == 0.
            assert torch.equal(targets.argmax(-1), codes)
        return dict(soft_targets=soft, stochastic_prefixes=soft, integer_joint_ids=True,
            shape=list(codes.shape), vocabulary=aux.quantizer.vocab_size,
            target_policy=self.target_policy(),
            temperature=self.target_temperature() if soft else None, entropy=entropy / len(flat),
            sampled_trajectories=0, auxiliary_loss_terms=0, residual_codebook_encoding=True)

    def loss(self, model, logits, targets):
        weights = self.depth_loss_weights()
        if weights is not None:
            from src.training.depth_weighted_ce import depth_weighted_soft_cross_entropy
            loss, self.last_loss_metrics = depth_weighted_soft_cross_entropy(
                logits, targets, weights, chunk_size=int(self.recipe.targets.chunk_size))
            return loss
        # Exact native soft CE and gradient; row chunks bound FP32 intermediates.
        from src.scaled_atom_training import soft_cross_entropy
        return soft_cross_entropy(logits, targets, chunk_size=int(self.recipe.targets.chunk_size))

    def depth_loss_weights(self):
        values = self.recipe.get('loss', {}).get('depth_weights')
        if values is None:
            return None
        from src.training.depth_weighted_ce import normalize_depth_weights
        return normalize_depth_weights(values, self.shape[-1])

    def sampling_settings(self):
        from src.training.rq_joint_sampling import joint_top_k
        return dict(temperature=float(self.recipe.sampling.temperature),
                    top_k=joint_top_k(self.recipe.sampling.top_k, self.shape[-1],
                        int(self.recipe.tokenizer.vocab_size)),
                    top_p=[float(self.recipe.sampling.top_p)])

    def logging_sampling_settings(self):
        values=super().logging_sampling_settings()
        limits=self.sampling_settings()['top_k']
        values['sampler_top_k_by_depth']=limits*self.shape[-1] if len(limits)==1 else limits
        return values

    def sample(self, model, aux, count, labels):
        if getattr(self, 'execution_options', {}).get('sampling_cast_cache', False):
            # Keep the native FP16 forward scopes nested until sampling ends.
            # The outer disabled scope leaves sampling probabilities in their
            # original precision while retaining cached weight casts between
            # tokens. Exiting it clears the cache before training can update
            # the parameters again.
            with torch.autocast(device_type=labels.device.type, enabled=False):
                return super().sample(model, aux, count, labels)
        return super().sample(model, aux, count, labels)

    def digest(self, aux):
        digest = hashlib.sha256()
        for name, value in aux.state_dict().items():
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def metadata(self):
        execution_path = self.base/'inputs/execution.json'
        execution = json.loads(execution_path.read_text()) if execution_path.exists() else {}
        learned = bool(self.recipe.tokenizer.get('learned_compact_dictionary', False))
        soft = self.target_policy() == 'soft_stochastic'
        weights = self.depth_loss_weights()
        objective = 'native_soft_target_cross_entropy' if soft else 'hard_target_cross_entropy'
        if weights is not None:
            objective = 'depth_weighted_soft_target_cross_entropy' if soft else 'depth_weighted_hard_target_cross_entropy'
        return dict(architecture=('native RQTransformer with signed-atom geometric joint scoring'
                    if execution.get('geometry_shared_head', False) else 'unmodified upstream RQTransformer'),
            geometry_shared_head=bool(execution.get('geometry_shared_head', False)),
            stage1=('frozen LASER encoder/decoder with learned compact pair dictionary' if learned else
                    'frozen LASER encoder/dictionary/decoder with finite joint residual codebooks'),
            codebook_shared_across_depths=bool(self.recipe.tokenizer.get('shared_codebook_across_depths', False)),
            integer_joint_ids=True, scalar_events_per_site=int(self.recipe.tokenizer.get('depth', 4)),
            vocabulary=int(self.recipe.tokenizer.vocab_size),
            soft_labels=soft, stochastic_codes=soft, target_policy=self.target_policy(),
            soft_label_temperature=self.target_temperature() if soft else None,
            objective=objective, depth_loss_weights=weights,
            auxiliary_loss_terms=0,
            sampled_trajectories=0, coefficient_refitting=False,
            codebook_sha256=self.codebook_sha256, native_sampling=self.sampling_settings(),
            measured_tokenizer_rfid=float(self.recipe.tokenizer.get('measured_rfid', float('nan'))),
            learned_compact_dictionary=learned,
            initialization=OmegaConf.to_container(self.recipe.initialization, resolve=True)
                if self.recipe.get('initialization') is not None else {},
            execution=execution,
            shared_upstream=rq_upstream.provenance())

    def checkpoint_extra(self):
        return dict(recipe=OmegaConf.to_container(self.recipe, resolve=True),
                    joint_codebook_sha256=self.codebook_sha256)

    def verify_resume(self, payload):
        assert payload['recipe'] == OmegaConf.to_container(self.recipe, resolve=True)
        assert payload['joint_codebook_sha256'] == self.codebook_sha256
