"""Frozen entrypoint for a fresh ImageNet RQ-recipe LASER compound prior."""
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

BASE = Path(os.environ['LASER_RUN_BASE'])
ROOT = BASE/'source/runtime'
EVIDENCE = Path(os.environ['LASER_PERSISTENT_BASE'])
VERIFY = EVIDENCE/'verification'/os.environ['LASER_PHASE']
VERIFY.mkdir(parents=True, exist_ok=True)
sys.path[:0] = [str(ROOT), str(BASE/'support')]

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
import yaml
from src.training import rqtransformer as training
from src.models.physical_compound_prior import PhysicalCompoundRQTransformer
from src.models.rqtransformer.attentions import AttentionBlock
from src.training.fresh_images import EpochImageFolder
from src.training.background_checkpoint import BackgroundCheckpointWriter
from src.training.checkpoint_upload import CheckpointUploader
from src.training.training_loss_tracker import TrainingLossTracker
from src.training.physical_pair_crps import physical_pair_objective_components
from src.training import stochastic_image_pairs as pair_teacher
from src.training.full_resume_upload import recovery_metadata
from src.training import k4_checkpoint_io as checkpoint_io
from src.soft_omp_bank import sparse_atom_entropy
from cpu_checkpoint_snapshot import CPUCheckpointSnapshotter
from verified_wandb_checkpoint_upload import VerifiedCloudUpload
from official_metrics import install as install_official_metrics

PLAN = json.loads((BASE/'plan.json').read_text())
POLICY = PLAN['target_policy']
ARGS = MODEL = WB = UPLOADER = None
STOPPING = False
CURSOR = 0
LAUNCH_START_STEP = 0
TRACKER = TrainingLossTracker(0)
LOSS_SUM = None
LATEST = {}
BEST_FID, BEST_IS = [], []
CHECKPOINT_EPOCH = 0
OFFICIAL = None
WRITER = BackgroundCheckpointWriter()
SNAPSHOTTER = CPUCheckpointSnapshotter()
COMPILE = os.environ.get('LASER_COMPILE_BLOCKS', '1') == '1'


def record(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str)+'\n')
    temporary.replace(path)


record(VERIFY/('process-rank'+os.environ['RANK']+'.json'), dict(pid=os.getpid(),time=time.time()))
STACK = (VERIFY/('stacks-rank'+os.environ['RANK']+'.log')).open('a')
faulthandler.register(signal.SIGUSR1, file=STACK, all_threads=True)
def stop(*_):
    global STOPPING
    STOPPING = True
signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)

native_parser = training.build_parser
def parser():
    p = native_parser()
    parse = p.parse_args
    def parse_args(*args, **kwargs):
        global ARGS
        ARGS = parse(*args, **kwargs)
        assert ARGS.lr == .0005 and ARGS.total_batch_size == 2048
        assert ARGS.epochs == ARGS.lr_schedule_epochs == 100 and ARGS.min_lr == 0
        assert ARGS.warmup_epochs == 0 and ARGS.lr_schedule == 'cosine'
        assert ARGS.token_cache is None and ARGS.init_stage2_checkpoint is None
        assert ARGS.metric_backend == 'original-rqvae' and ARGS.fid_real_split == 'train'
        assert ARGS.coeff_crps_weight == ARGS.geometry_loss_weight == 0
        assert ARGS.physical_pair_context and not ARGS.compound_tokens
        assert ARGS.stochastic_atom_temperature == POLICY['atom_temperature']
        assert ARGS.stochastic_atom_soft_target_variants == POLICY['variants']
        if ARGS.resume:
            ARGS.resume_checkpoint = checkpoint_io._checkpoint_upload_source(ARGS.resume_checkpoint)
        return ARGS
    p.parse_args = parse_args
    return p
training.build_parser = parser

native_source = training.source_image_dataset
def images(dataset, root, transform, *, split='train'):
    if dataset == 'imagenet' and split == 'train':
        result = EpochImageFolder(root/split,transform=transform,augmentation_seed=261001)
        assert len(result) == 1281167 and len(result.classes) == 1000
        return result
    return native_source(dataset, root, transform, split=split)
training.source_image_dataset = images
native_sampler_init = training.ExactGlobalBatchSampler.__init__
def sampler_init(self, dataset, *args, **kwargs):
    self.dataset = dataset
    return native_sampler_init(self, dataset, *args, **kwargs)
training.ExactGlobalBatchSampler.__init__ = sampler_init
native_sampler_epoch = training.ExactGlobalBatchSampler.set_epoch
def sampler_epoch(self, epoch):
    if hasattr(self.dataset, 'set_epoch'):
        self.dataset.set_epoch(epoch)
    return native_sampler_epoch(self, epoch)
training.ExactGlobalBatchSampler.set_epoch = sampler_epoch

native_build = training.build_model
def build(*args, **kwargs):
    global MODEL
    MODEL = PhysicalCompoundRQTransformer.from_scalar(native_build(*args, **kwargs))
    assert len(MODEL.body_transformer.blocks) == 42 and len(MODEL.head_transformer.blocks) == 6
    assert MODEL.config.embed_dim == 1536
    for name in ('body', 'head'):
        config = MODEL.config[name]['block']
        assert config.n_head == 24 and config.resid_pdrop == .1 and config.attn_pdrop == 0
    assert MODEL.config.embd_pdrop == 0
    return MODEL
training.build_model = build
def wrap(model, backend, device, world):
    assert world == 8 and backend == 'ddp'
    for block in model.head_transformer.blocks:
        block.attn.short_attention_backend = 'compiled'
    if COMPILE:
        for block in model.modules():
            if isinstance(block, AttentionBlock):
                eager = block.forward
                compiled = torch.compile(eager, fullgraph=True, dynamic=True)
                def forward(x, eager=eager, compiled=compiled, module=block):
                    return compiled(x) if module.training and torch.is_grad_enabled() else eager(x)
                block.forward = forward
    return DDP(model, device_ids=[device.index], broadcast_buffers=False,
        gradient_as_bucket_view=True,bucket_cap_mb=100)
training.wrap_distributed_model = wrap

native_load = torch.load
def load(path, *args, **kwargs):
    global CURSOR, LAUNCH_START_STEP, TRACKER, BEST_FID, BEST_IS
    payload = native_load(path, *args, **kwargs)
    if ARGS is not None and ARGS.resume and isinstance(path, (str, Path)) and Path(path).resolve() == ARGS.resume_checkpoint.resolve():
        if payload.get('original_rq_scratch_policy') != POLICY:
            raise RuntimeError('Only this fresh run with its exact target policy can be resumed')
        CURSOR = payload['global_step']
        LAUNCH_START_STEP = CURSOR
        metadata = recovery_metadata(payload)
        assert metadata['adam_step'] == CURSOR and metadata['rng_ranks'] == 8
        TRACKER = TrainingLossTracker(CURSOR, payload.get('train_loss_tracking'))
        BEST_FID, BEST_IS = payload.get('best_fid', []), payload.get('best_inception', [])
        record(VERIFY/('recovery-rank'+os.environ['RANK']+'.json'), metadata)
    return payload
torch.load = load
native_restore = training.restore_rank_rng_state
def restore(payload, device, seed):
    result = native_restore(payload, device, seed)
    if payload is not None:
        expected = payload['rng_state_by_rank'][dist.get_rank()]
        assert torch.equal(torch.get_rng_state(), expected['torch_cpu'])
        assert torch.equal(torch.cuda.get_rng_state(device), expected['torch_cuda'])
        record(VERIFY/('rng-rank'+os.environ['RANK']+'.json'),dict(step=payload['global_step'],restored_exactly=True))
    return result
training.restore_rank_rng_state = restore

native_teacher = pair_teacher.stochastic_soft_image_targets
teacher_audited = False
def teacher(aux, images, **kwargs):
    global teacher_audited
    if not teacher_audited:
        device = images.device
        with torch.random.fork_rng(devices=[device.index]):
            cpu, cuda = torch.get_rng_state(),torch.cuda.get_rng_state(device)
            first = native_teacher(aux,images[:8],**kwargs)
            second = native_teacher(aux,images[:8],**kwargs)
            changed_atoms = (first['atoms'] != second['atoms']).float().mean().item()
            changed_coefficients = (first['tokens'][...,1::2] != second['tokens'][...,1::2]).float().mean().item()
            assert changed_atoms > 0 and changed_coefficients > 0
            torch.set_rng_state(cpu)
            torch.cuda.set_rng_state(cuda,device)
            replay = native_teacher(aux,images[:8],**kwargs)
            for key in ('tokens','atom_ids','atom_weights','coefficient_probabilities'):
                assert torch.equal(first[key],replay[key]),key
            record(VERIFY/('stochastic-targets-rank'+os.environ['RANK']+'.json'),dict(
                both_inputs_stochastic=True,both_labels_soft=True,changed_atom_fraction=changed_atoms,
                changed_coefficient_fraction=changed_coefficients,rng_replay_exact=True,
                prefix_consistent_conditionals=True,policy=POLICY))
        teacher_audited = True
    return native_teacher(aux,images,**kwargs)
pair_teacher.stochastic_soft_image_targets = teacher

compiled_objective = torch.compile(physical_pair_objective_components, fullgraph=True, dynamic=True) if COMPILE else physical_pair_objective_components
def objective(a, c, atoms, probabilities, bins, weight, accumulation, **kwargs):
    global LOSS_SUM
    total, ce, crps = compiled_objective(a, c, atoms, probabilities, bins, weight, accumulation, **kwargs)
    with torch.no_grad():
        atom_entropy = sparse_atom_entropy(kwargs['target_atom_ids'], kwargs['target_atom_weights']).mean()
        coefficient_entropy = -(probabilities*probabilities.clamp_min(1e-30).log()).sum(-1).mean()
        values = torch.stack((total.detach()*accumulation, ce.detach(), crps.detach(), weight.detach(),
            atom_entropy, coefficient_entropy, ce.detach()-(atom_entropy+coefficient_entropy)/2)) * atoms.shape[0]
        values = torch.cat((values,values.new_tensor([atoms.shape[0]])))
        LOSS_SUM = values if LOSS_SUM is None else LOSS_SUM+values
    return total
training.physical_pair_objective = objective

native_step = torch.optim.AdamW.step
initial_step_checked = False
def step(optimizer, *args, **kwargs):
    global CURSOR, LOSS_SUM, LATEST, initial_step_checked
    WRITER.check()
    if UPLOADER is not None:
        UPLOADER.check()
    if not initial_step_checked:
        assert optimizer.defaults['betas'] == (.9,.95) and optimizer.defaults['weight_decay'] == .0001
        if CURSOR == 0:
            assert not optimizer.state and optimizer.param_groups[0]['lr'] == .0005
        else:
            assert len(optimizer.state) == len(list(MODEL.parameters()))
            assert {int(s['step']) for s in optimizer.state.values()} == {CURSOR}
        record(VERIFY/('startup-rank'+os.environ['RANK']+'.json'),dict(global_step=CURSOR,
            adam_state_count=len(optimizer.state),fresh_optimizer=CURSOR==0,lr=optimizer.param_groups[0]['lr'],
            model_parameters=sum(p.numel() for p in MODEL.parameters()),parameter_tensors=len(list(MODEL.parameters())),
            old_stage2_checkpoint_loaded=False,policy=POLICY))
        initial_step_checked = True
    result = native_step(optimizer, *args, **kwargs)
    CURSOR += 1
    dist.all_reduce(LOSS_SUM)
    values = (LOSS_SUM[:-1]/LOSS_SUM[-1]).tolist()
    samples = int(LOSS_SUM[-1])
    LOSS_SUM = None
    LATEST = TRACKER.update(*values[:4],samples,CURSOR)
    LATEST.update({'train/atom_target_entropy':values[4],'train/coeff_target_entropy':values[5],
        'train/target_to_model_kl':values[6]})
    if CURSOR in (1,20,21,40):
        assert {int(s['step']) for s in optimizer.state.values()} == {CURSOR}
        assert all(p.grad is not None for p in MODEL.parameters())
        finite = torch.stack([torch.isfinite(p).all() for p in MODEL.parameters()]
            +[torch.isfinite(s[k]).all() for s in optimizer.state.values() for k in ('exp_avg','exp_avg_sq')]).all()
        assert bool(finite)
        record(VERIFY/(f'step{CURSOR}-rank'+os.environ['RANK']+'.json'),dict(finite=True,
            global_step=CURSOR,adam_step=CURSOR,lr=optimizer.param_groups[0]['lr'],
            maximum_allocated_gib=torch.cuda.max_memory_allocated()/2**30,metrics=LATEST))
    stopping = torch.tensor(int(STOPPING),device=next(MODEL.parameters()).device)
    dist.all_reduce(stopping,op=dist.ReduceOp.MAX)
    if stopping.item():
        ARGS.max_optimizer_steps = CURSOR-LAUNCH_START_STEP
    return result
torch.optim.AdamW.step = step


def submit_upload():
    if UPLOADER is None:
        return
    checkpoints = ARGS.checkpoint_dir
    sources = [('last.pt',checkpoints/'last.pt')]
    for label, ranking in [('fid',BEST_FID),('is',BEST_IS)]:
        if ranking:
            sources.append((f'best-{label}-resume.pt',Path(ranking[0][1])))
    if any(not path.is_file() for _,path in sources):
        return
    slots = BASE/'upload-slots'
    slots.mkdir(exist_ok=True)
    for name, source in sources:
        checkpoint_io._replace_hard_link(checkpoint_io._checkpoint_upload_source(source),slots/name)
    UPLOADER.submit([slots/name for name,_ in sources],CHECKPOINT_EPOCH)


def save(payload, target):
    global BEST_FID, BEST_IS, CHECKPOINT_EPOCH
    BEST_FID, BEST_IS = payload.get('best_fid', []),payload.get('best_inception', [])
    CHECKPOINT_EPOCH = payload['epoch']
    payload = dict(payload, original_rq_scratch_policy=POLICY,train_loss_tracking=TRACKER.state_dict(),
        config=dict(payload['config'],architecture='gated_full_history_physical_compound_v2',
            compound_event_history=True,compound_event_order='raster_then_depth_atom_coefficient_pair'))
    if OFFICIAL is not None and OFFICIAL['global_step'] == payload['global_step']:
        payload['original_rqtransformer_metrics'] = OFFICIAL
    WRITER.wait()
    frozen = SNAPSHOTTER.freeze(payload)
    metadata = recovery_metadata(frozen)
    assert metadata['adam_step'] == metadata['global_step']
    def committed():
        record(EVIDENCE/'last-local-save.json',dict(epoch=payload['epoch'],step=payload['global_step'],
            target=str(target),bytes=target.stat().st_size,time=time.time()))
        submit_upload()
    return WRITER.submit(lambda:checkpoint_io.atomic_torch_save(frozen,target,on_commit=committed,collect_garbage=False))
training.atomic_torch_save = save
def snapshot(source, target):
    def alias():
        payload = source.resolve(strict=True)
        temporary = target.with_suffix('.alias.tmp')
        temporary.unlink(missing_ok=True)
        temporary.symlink_to(payload.relative_to(target.parent))
        temporary.replace(target)
        submit_upload()
    WRITER.append(alias)
training.snapshot_checkpoint = snapshot
training.remove_checkpoint = checkpoint_io.remove_checkpoint
training.upload_selected_checkpoint_files = lambda *args, **kwargs: []

def before_evaluation(model, optimizer, names, device, epoch, global_step,scheduler,
                      config,best_fid,best_inception,last_checkpoint):
    state,opt = training.full_checkpoint_states(model,optimizer,names)
    rng = training.gather_rank_rng_states(device)
    if dist.get_rank() == 0:
        save(dict(epoch=epoch+1,batch_idx=0,global_step=global_step,state_dict=state,
            optimizer=opt,scheduler=scheduler.state_dict(),config=config,
            rng_state_by_rank=rng,checkpoint_world_size=8,best_fid=best_fid,best_inception=best_inception),last_checkpoint)
        WRITER.wait()
    dist.barrier()
training.save_before_official_evaluation = before_evaluation

install_official_metrics(ROOT)
native_evaluate = training.evaluate_generation_metrics
def evaluate(*args, **kwargs):
    global OFFICIAL
    assert kwargs['metric_backend'] == 'original-rqvae'
    result = native_evaluate(*args, **kwargs)
    OFFICIAL = dict(global_step=CURSOR,fid=result[0],inception_score=result[1],inception_score_std=result[2],
        metric_backend='original_rqtransformer',generated_images=50000,real_split='train',
        real_images=1281167,inception_splits=10,seed=261001)
    if dist.get_rank() == 0:
        record(EVIDENCE/f'official-step{CURSOR}.json',OFFICIAL)
        WB.log({'train/global_step':CURSOR,'eval/fid_original_rqtransformer':result[0],
            'eval/inception_score_original_rqtransformer':result[1],
            'eval/inception_score_std_original_rqtransformer':result[2]})
    return result
training.evaluate_generation_metrics = evaluate

native_init = wandb.init
def init(*args, **kwargs):
    global WB,UPLOADER
    kwargs['resume'] = 'allow'
    kwargs['config'].update(original_rq_scratch_plan=PLAN,training_from_scratch=True,
        optimizer_from_scratch=True,stochastic_atom_supports=True,stochastic_coefficients=True,
        soft_targets_for_both_components=True,full_compound_event_history=True,
        source_stage2_checkpoint=None,parameter_ema_enabled=False,
        evaluation_protocol='Official RQTransformer FID/train reference, generated50k,10-split IS',
        original_bundle_config_sha256=PLAN['bundle_config_sha256'])
    WB = native_init(*args, **kwargs)
    native_log = WB.log
    def log(data, *args, **kwargs):
        if 'train/loss' in data and LATEST:
            data.update(LATEST)
        return native_log(data,*args,**kwargs)
    WB.log = log
    if ARGS.upload_checkpoints:
        UPLOADER = CheckpointUploader(BASE/'upload-snapshots',
            VerifiedCloudUpload('helloimlixin-rutgers/laser/'+WB.id,EVIDENCE/'cloud-checkpoint-receipt.json'),
            immutable_sources=True)
        submit_upload()
    native_finish = WB.finish
    def finish(*args, **kwargs):
        WRITER.close()
        submit_upload()
        if UPLOADER is not None:
            UPLOADER.close()
        record(EVIDENCE/'trainer-finish.json',dict(global_step=CURSOR,epoch=CHECKPOINT_EPOCH,
            stopped=STOPPING,no_further_updates=True,time=time.time()))
        return native_finish(*args,**kwargs)
    WB.finish = finish
    return WB
wandb.init = init

config_path = Path(sys.argv[sys.argv.index('--config')+1])
from src.training.options import options_to_argv
try:
    training.main(options_to_argv(yaml.safe_load(config_path.read_text())['options']))
finally:
    if WB is None:
        WRITER.close()
