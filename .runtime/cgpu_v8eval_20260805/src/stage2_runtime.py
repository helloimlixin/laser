"""Compat shim for the older stage-2 runtime import path."""

from src.s2 import (
    Batch as Stage2SampleBatch,
    Run as Stage2Runtime,
    build_stage2_sample_payload,
    default_generate_output_dir,
    default_stage2_sample_dir,
    generate_stage2_samples,
    load_stage2_runtime,
    resolve_stage2_device,
    resolve_stage2_nrow,
    save_stage2_sample_grids,
    save_stage2_sample_payload,
)
