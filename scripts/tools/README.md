Legacy-adjacent helper utilities live here so top-level `scripts/` stays focused on the maintained launch path.

Maintained launch entrypoints:
- `scripts/profile.sh`
- `scripts/run.sh`
- `scripts/sweep.sh`

Utilities in this directory:
- `smoke_e2e.py`
- `cache.py`
- `compute_rfid.py` - post-hoc paper-style rFID: full validation split by default (`--max-samples 0`); supports ImageNet/image-folder datasets.
- `kmeans_quantize_sparse_codes.py`
- `prune_runs.py`
- `laser_sanity.py`
