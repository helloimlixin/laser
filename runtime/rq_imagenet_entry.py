"""Bootstrap a frozen runtime for either representation's shared driver."""
import os
from pathlib import Path
import shutil
import sys


SHARED_SOURCE_FILES = (
    'src/__init__.py','src/models/__init__.py','src/training/__init__.py',
    'src/rq_upstream.py','src/rqvae_metrics.py','src/stochastic_compound.py',
    'src/models/laser_rq_adapter.py','src/models/laser_tokenizer_adapter.py',
    'src/models/laser_pair_context_adapter.py',
    'src/models/laser_joint_tokenizer.py', 'src/models/laser_joint_execution.py',
    'src/models/geometric_joint_adapter.py',
    'src/training/laser_shared_joint_vocabulary.py',
    'src/training/laser_compact_joint_vocabulary.py',
    'src/training/laser_joint_recipe.py',
    'src/training/depth_weighted_ce.py',
    'src/training/rq_joint_sampling.py',
    'src/models/laser_packed_integer_tokenizer.py',
    'src/models/laser_pair_id_head.py',
    'src/training/laser_packed_integer_recipe.py','src/training/packed_pair_loss.py',
    'src/training/pair_id_linear_loss.py',
    'src/models/laser_catalog_integer_tokenizer.py',
    'src/training/laser_catalog_integer_recipe.py',
    'src/models/laser_native_compound_adapter.py','src/training/laser_compound_replay_recipe.py',
    'src/continuous_sparse_codec.py','src/models/continuous_coefficient_mixture.py',
    'src/models/continuous_compound_rq.py','src/training/continuous_compound_recipe.py',
    'src/training/compound128_recipe.py','src/training/rq_sdpa_backend.py',
    'src/training/laser_hard_ar.py','src/models/laser_compound_rq_adapter.py',
    'src/models/laser_native_support_order_adapter.py','src/training/laser_native_support_order_recipe.py',
    'src/models/laser_fine_joint_adapter.py', 'src/models/laser_fine_joint_tokenizer.py',
    'src/training/laser_fine_joint_recipe.py',
    'src/models/laser_signed_joint_adapter.py', 'src/training/laser_signed_joint_recipe.py',
    'src/scaled_atom_training.py', 'src/scaled_atom_rq.py',
    'src/training/rq_imagenet_shared.py','src/training/rq_imagenet_recipes.py',
    'src/training/rq_rng.py',
    'src/training/rq_sample_panels.py',
    'src/training/rq_sampling_override.py',
    'src/training/rq_imagenet_runtime.py','src/training/laser_rq_minimal.py',
    'src/training/stochastic_image_pairs.py','src/training/stochastic_omp_graph.py',
    'src/training/omp_joint_targets.py','src/training/exact_global_batch.py',
    'src/training/omp_math.py','src/training/omp_kernels.py',
    'src/training/k4_checkpoint_io.py',
)


def freeze_shared_runtime(base, project, *, include_stochastic_teacher=True):
    """Freeze only the shared driver and LASER adapters, for either recipe."""
    base, project = Path(base), Path(project)
    files = SHARED_SOURCE_FILES
    if not include_stochastic_teacher:
        excluded = ('src/stochastic_compound.py','src/training/stochastic_image_pairs.py',
            'src/training/stochastic_omp_graph.py','src/training/omp_joint_targets.py',
            'src/training/omp_math.py','src/training/omp_kernels.py')
        files = tuple(name for name in files if name not in excluded) + (
            'src/training/laser_hard_ar.py','src/training/laser_ar_factors.py','src/orthogonal_sparse_codec.py',
            'src/models/laser_compound_rq_adapter.py','src/training/laser_pair_ablation.py')
    for name in files:
        target = base/'source'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(project/name,target)
    (base/'runtime').mkdir(parents=True,exist_ok=True)
    for name in ('rq_imagenet_entry.py','amp_replay.py','rq_checkpoint_store.py'):
        shutil.copyfile(project/'runtime'/name,base/'runtime'/name)


def launch(kind):
    project = Path('/workspace/Projects/laser')
    variable = 'ORIGINAL_RQ_BASE' if kind == 'rq' else 'LASER_RQ_BASE'
    default = '/tmp/laser-original-rq-imagenet-20261014' if kind == 'rq' else '/tmp/laser-rq-minimal-imagenet-20261014'
    base = Path(os.environ.get(variable, default)).resolve()
    os.environ['RQ_UPSTREAM_ROOT'] = str(base/'upstream')
    os.environ['TORCH_HOME'] = str(base/'torch-cache')
    sys.path[:0] = [str(base/'source'),str(base/'upstream'),str(base),str(base/'runtime'),
                    str(project),str(project/'runtime'),str(project/'scripts/tools')]
    from src.training.rq_imagenet_shared import main
    main(kind,base)
