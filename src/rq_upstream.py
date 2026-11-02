"""Resolve the single vendored KakaoBrain implementation used by both priors."""
import hashlib
import os
from pathlib import Path
import sys

UPSTREAM_COMMIT = '341395e562ac347f5eb62db9f5f08b9f2cc42a60'
ROOT = Path(os.environ.get('RQ_UPSTREAM_ROOT',
    Path(__file__).resolve().parents[1] / 'third_party/rq-vae-transformer')).resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rqvae.models.rqtransformer.transformers import RQTransformer
from rqvae.models.rqtransformer.configs import RQTransformerConfig

for module_name in (RQTransformer.__module__, RQTransformerConfig.__module__):
    imported = Path(sys.modules[module_name].__file__).resolve()
    if not imported.is_relative_to(ROOT):
        raise RuntimeError(f'RQ module came from another codebase: {imported}')


CORE_FILES = (
    'rqvae/models/rqtransformer/transformers.py',
    'rqvae/models/rqtransformer/attentions.py',
    'rqvae/models/rqtransformer/primitives.py',
    'rqvae/models/rqtransformer/configs.py',
    'rqvae/optimizer/loss.py', 'rqvae/optimizer/optimizer.py',
    'rqvae/optimizer/scheduler.py', 'rqvae/img_datasets/transforms.py',
    'rqvae/models/rqvae/rqvae.py',
)


def provenance():
    return dict(upstream_commit=UPSTREAM_COMMIT,
        compatibility_patch='Python 3.12 dataclass default_factory',
        core_files={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                    for name in CORE_FILES})
