"""Reproducible online ImageNet compound training with zero-based offset IDs."""
from pathlib import Path
import hashlib
import json
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

from src.training import rqtransformer as training
from src.training.rq_imagenet_runtime import TarImageNet, write_json

STAGE1_SHA = 'dd28db9306d526bdc9fbf8016403e106c4f317fd8f5c04792f0953e53310fdab'
SCALES = [4.587941646575928, 2.4749884605407715, 1.6106663942337036, 1.0176019668579102]


class ImageNetCompoundAux(training.LaserAux):
    """Freeze the teacher in FP32 and reuse its derived dictionary Gram matrix."""
    @torch.no_grad()
    def encode_sparse_components(self, images, **kwargs):
        matmul = torch.backends.cuda.matmul.allow_tf32
        convolution = torch.backends.cudnn.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            with torch.autocast(device_type=images.device.type, enabled=False):
                if not hasattr(self, '_cached_gram'):
                    self._cached_gram = self.dictionary.float().T @ self.dictionary.float()
                kwargs.setdefault('dictionary_gram', self._cached_gram)
                parts = [super(ImageNetCompoundAux, self).encode_sparse_components(x.float(), **kwargs)
                         for x in images.split(int(os.environ.get('LASER_ENCODER_BATCH', '16')))]
                return tuple(torch.cat([p[i] for p in parts]) for i in range(len(parts[0])))
        finally:
            torch.backends.cuda.matmul.allow_tf32 = matmul
            torch.backends.cudnn.allow_tf32 = convolution

    @torch.no_grad()
    def decode_tokens(self, tokens):
        convolution = torch.backends.cudnn.allow_tf32
        try:
            torch.backends.cudnn.allow_tf32 = False
            with torch.autocast(device_type=tokens.device.type, enabled=False):
                return super().decode_tokens(tokens)
        finally:
            torch.backends.cudnn.allow_tf32 = convolution

    @torch.no_grad()
    def decode_compound(self, atoms, coefficient_ids):
        tokens = torch.stack((atoms, coefficient_ids + self.num_atoms), dim=-1).flatten(-2)
        return self.decode_tokens(tokens)


def make_aux(checkpoint, device, *, coeff_vocab_size=512):
    return ImageNetCompoundAux(Path(checkpoint), 16384, coeff_vocab_size, 3., coeff_scales=SCALES,
                              soft_target_physical=False, clamp_coeffs=True,
                              sparsity_level=4).to(device).eval()


def install(cfg):
    """Select local JPEG access, bounded teacher precision, and recovery policy."""
    import wandb
    from src.training.recoverable_checkpoints import LocalFirstCheckpoints
    from src.models.offset_pair_prior import OffsetPairRQTransformer

    options = cfg.options
    num_atoms = int(options.num_atoms)
    coefficient_bins = int(options.coeff_vocab_size)
    sparsity = int(options.sparsity_level)
    if num_atoms != 16384 or sparsity != 4 or coefficient_bins < 2:
        raise ValueError('Audited ImageNet teacher requires 16384 atoms, depth 4, and at least 2 coefficient bins')
    with Path(options.checkpoint).open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != STAGE1_SHA:
            raise ValueError('The frozen stage-1 checkpoint hash does not match the audited tokenizer')
    if options.get('init_stage2_checkpoint'):
        raise ValueError('This runtime requires full-state recovery rather than weights-only initialization')
    resume_step = 0
    if options.resume:
        path = Path(options.get('resume_checkpoint') or Path(options.checkpoint_dir)/'last.pt')
        snapshot = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        validate_resume_recipe(snapshot['config'], options)
        resume_step = int(snapshot['global_step'])
        del snapshot
    elif options.get('resume_checkpoint'):
        raise ValueError('A recovery checkpoint requires resume=true')
    training.LaserAux = ImageNetCompoundAux
    original_source = training.source_image_dataset
    def source(dataset, root, transform, *, split='train'):
        if dataset == 'imagenet':
            return TarImageNet(root, split, transform, seed=int(options.seed))
        return original_source(dataset, root, transform, split=split)
    training.source_image_dataset = source
    original_loader = training.DataLoader
    def loader(*args, **kwargs):
        kwargs['num_workers'] = int(os.environ.get('LASER_DATA_WORKERS', '4'))
        kwargs['persistent_workers'] = kwargs['num_workers'] > 0
        if kwargs['num_workers']:
            kwargs.update(prefetch_factor=2, multiprocessing_context='spawn')
        return original_loader(*args, **kwargs)
    training.DataLoader = loader
    builder = training.build_model
    def build(*args, **kwargs):
        model = builder(*args, **kwargs)
        if type(model) is not OffsetPairRQTransformer:
            raise TypeError('Expected the audited zero-based compound pair model')
        if (model.token_vocabulary_size != num_atoms + coefficient_bins
                or model.coeff_vocab_size != coefficient_bins
                or tuple(model.token_block_size) != (8, 8, 2*sparsity)):
            raise ValueError('Unexpected compound vocabulary or image shape')
        return model
    training.build_model = build
    def wrap(model, backend, device, world_size):
        joint_sampling = int(options.get('joint_pair_top_k', 0)) > 0
        micro_blocks = {id(block) for block in model.coeff_micro_transformer.modules()}
        if os.environ.get('LASER_COMPILE_BLOCKS', '0') == '1':
            from src.models.rqtransformer.attentions import AttentionBlock
            from src.training.compound_balance import configure_retained_gradient_probes
            configure_retained_gradient_probes()
            for block in model.modules():
                if isinstance(block, AttentionBlock):
                    block.attn.short_attention_backend = 'compiled'
                    eager = block.forward
                    compiled = torch.compile(eager, fullgraph=True, dynamic=True)
                    joint_micro = joint_sampling and id(block) in micro_blocks
                    def forward(x, eager=eager, compiled=compiled, module=block, joint_micro=joint_micro):
                        training_graph = module.training and torch.is_grad_enabled()
                        joint_inference = joint_micro and not module.training and not torch.is_grad_enabled()
                        return compiled(x) if training_graph or joint_inference else eager(x)
                    block.forward = forward
        if joint_sampling:
            for block in model.coefficient_history_decoder.blocks:
                block.forward_candidates = torch.compile(block.forward_candidates, fullgraph=True, dynamic=True)
        if world_size == 1:
            return model
        if backend != 'ddp':
            raise ValueError('This ImageNet recipe uses DDP')
        return training.DDP(model, device_ids=[device.index], broadcast_buffers=False,
                            gradient_as_bucket_view=True, bucket_cap_mb=100)
    training.wrap_distributed_model = wrap
    root = Path(options.checkpoint_dir)
    root.mkdir(parents=True, exist_ok=True)
    training.persistent_checkpoint_dir = lambda output, explicit: root
    import math
    checkpoints = LocalFirstCheckpoints(root,
        final_step=math.ceil(1281167/int(options.total_batch_size))*int(options.epochs))
    training.atomic_torch_save = checkpoints.save
    training.snapshot_checkpoint = checkpoints.snapshot
    def upload(wb, **kwargs):
        return checkpoints.upload(wb, **kwargs)
    upload.close = checkpoints.close
    training.upload_selected_checkpoint_files = upload
    original_init = wandb.init
    def init(*args, **kwargs):
        if resume_step:
            kwargs['allow_val_change'] = True
        run = original_init(*args, **kwargs)
        run.config.update(dict(token_convention='zero_based_half_open',
            atom_token_range=[0, num_atoms], coefficient_token_range=[num_atoms, num_atoms+coefficient_bins],
            compound_factorization='p(atom | completed pairs) p(coeff | completed pairs, current atom)',
            teacher_precision='FP32 encoder/OMP, TF32 disabled, cached Gram',
            augmentation='upstream Resize256 RandomCrop256 RandomHorizontalFlip(0.5)',
            initial_stage2_step=0, initial_optimizer_age=0, execution_resume_step=resume_step,
            batch_size=int(options.batch_size),
            accumulation_steps=int(os.environ.get('LASER_ACCUMULATION', '1')),
            attention_blocks_compiled=os.environ.get('LASER_COMPILE_BLOCKS', '0') == '1',
            checkpoint_shared_filesystem_access=False), allow_val_change=True)
        if int(options.get('joint_pair_top_k', 0)) > 0:
            run.config.update(dict(generation_policy='certified_global_joint_atom_coefficient_topk',
                generation_product_vocabulary=num_atoms*coefficient_bins,
                joint_pair_top_k=int(options.joint_pair_top_k),
                joint_pair_temperature=float(options.get('joint_pair_temperature', 1.)),
                joint_pair_top_p=float(options.get('joint_pair_top_p', 1.)),
                independent_field_truncation=False), allow_val_change=True)
        records = Path(cfg.audit_directory)
        artifacts = (('production.yaml', 'selection.json', 'benchmarks.json',
            'compilation-parity.json', 'speedup-source.tar.gz', 'speedup-source-manifest.json',
            'speedup-tests.xml') if resume_step else
            ('recipe.yaml', 'audit.json', 'tests.xml', 'source.tar.gz',
             'source-manifest.json', 'quantization.json', 'canary.json'))
        for name in artifacts:
            path = records/name
            if path.is_file():
                run.save(str(path), base_path=str(records), policy='now')
        write_json(records/'wandb.json', dict(url=run.url, id=run.id))
        return run
    wandb.init = init
    return checkpoints


def validate_resume_recipe(saved, options):
    """Execution may change; the model, teacher, objective and LR policy may not."""
    fields = ('dataset', 'model_preset', 'backbone', 'sparsity_level', 'num_atoms',
        'coeff_vocab_size', 'coeff_max', 'coeff_scales', 'coeff_target_space',
        'coeff_target_mode', 'stochastic_coefficient_inputs', 'compound_tokens',
        'physical_pair_context', 'efficient_pair_backbone', 'compound_micro_transformer_layers',
        'compound_depth_specific_coeff_heads', 'coefficient_history_layers',
        'coefficient_history_width', 'coefficient_history_heads', 'coefficient_history_dropout',
        'coefficient_history_normalize_fields', 'mask_seen_atoms_training',
        'atom_loss_weight', 'geometry_loss_weight', 'lr', 'epochs', 'lr_schedule_epochs',
        'lr_schedule', 'min_lr', 'warmup_epochs', 'seed', 'total_batch_size')
    changed = {key: (saved.get(key), options.get(key)) for key in fields
               if saved.get(key) != options.get(key)}
    # Older categorical checkpoints predate these options. An architecture
    # change needs an explicitly migrated experiment checkpoint, never an
    # accidental recovery into different coefficient classifiers.
    defaults = dict(coefficient_head='categorical', coefficient_mixture_components=8,
                    coefficient_mixture_width=256, coefficient_mixture_min_scale=1.e-3)
    head = options.get('coefficient_head', 'categorical')
    for key, default in defaults.items():
        if key != 'coefficient_head' and head == 'categorical':
            continue
        before, after = saved.get(key, default), options.get(key, default)
        if before != after:
            changed[key] = (before, after)
    if changed:
        raise ValueError(f'Recovery changes the training recipe: {changed}')


class HeldoutValidation:
    def __init__(self, data):
        self.data, self.cache = Path(data), None

    @torch.no_grad()
    def __call__(self, model, aux, device, epoch, global_step, wb, output):
        raw = training.unwrap(model)
        was_training = raw.training
        try:
            with torch.random.fork_rng(devices=[device.index]):
                raw.eval()
                if self.cache is None:
                    dataset = TarImageNet(self.data, 'val', training.val_image_transform())
                    world = dist.get_world_size() if dist.is_initialized() else 1
                    indices = range(training.rank(), 4096, world)
                    self.cache = []
                    for images, labels in DataLoader(Subset(dataset, indices), batch_size=8):
                        atoms, values = aux.encode_sparse_components(images.to(device))
                        tokens, (_, probs) = aux.sparse_targets(atoms, values, hard=True,
                                                               stochastic=False, compact=True)
                        self.cache.append((tokens.cpu(), atoms.cpu(), probs.argmax(-1).cpu(), labels))
                sums = torch.zeros(4, 4, dtype=torch.float64, device=device)
                for batch in self.cache:
                    tokens, atoms, coeffs, labels = [x.to(device) for x in batch]
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        result = raw(tokens, aux, labels)
                    a, c = result['atom_logits'].float(), result['coeff_logits'].float()
                    ac = torch.nn.functional.cross_entropy(a.flatten(0, -2), atoms.flatten(), reduction='none').reshape_as(atoms)
                    cc = torch.nn.functional.cross_entropy(c.flatten(0, -2), coeffs.flatten(), reduction='none').reshape_as(coeffs)
                    for d in range(4):
                        sums[d] += torch.stack((ac[..., d].sum(), cc[..., d].sum(),
                            (a[..., d, :].argmax(-1) == atoms[..., d]).sum(),
                            torch.tensor(atoms[..., d].numel(), device=device)))
                if dist.is_initialized():
                    dist.all_reduce(sums)
                if training.rank() == 0:
                    result = {'train/global_step': global_step, 'train/epoch': epoch + 1}
                    for d in range(4):
                        for i, name in enumerate(('atom_ce', 'coeff_ce', 'atom_top1')):
                            result[f'heldout/depth{d+1}/{name}'] = float(sums[d, i]/sums[d, 3])
                    with (Path(output)/'heldout.jsonl').open('a') as handle:
                        handle.write(json.dumps(result)+'\n')
                    if wb is not None:
                        wb.log(result)
                    print(json.dumps(dict(phase='heldout_validation', **result)), flush=True)
        finally:
            raw.train(was_training)
