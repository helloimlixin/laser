import os
import warnings

# Mitigate MKL threading errors during test dataloading
# Always override to GNU because some environments export INTEL, which crashes
# PyTorch dataloader workers when linked against libgomp.
os.environ["MKL_THREADING_LAYER"] = "GNU"

# Quiet deterministic UMAP warning about forcing n_jobs=1 when random_state is set.
# Filter noisy deterministic UMAP warning regardless of module path.
warnings.filterwarnings("ignore", message=r"n_jobs value 1 overridden.*random_state")
