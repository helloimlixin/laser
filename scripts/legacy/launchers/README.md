Legacy launchers and helper scripts live here.

Examples:
- `local_p4.sh`: older local two-GPU p4 workflow
- `submit_p4gwin.sh`, `submit_p4gg.sh`, `submit_pg.sh`: older SLURM submit wrappers
- `job_*.sbatch`, `run_*.sh`, `p4g.sh`, `pg.sh`, `slurm_env.sh`: launcher internals

These files are preserved for ad hoc reruns and debugging of older jobs. The
maintained workflow is:
- `scripts/profile.sh`
- `scripts/run.sh`
- `scripts/sweep.sh`
