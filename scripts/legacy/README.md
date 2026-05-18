Legacy scripts are split by purpose:

- `launchers/`: older local launchers, SLURM wrappers, and helper scripts
- `sweeps/`: older sweep generators

Maintained entrypoints stay at the top level:
- `scripts/profile.sh`
- `scripts/run.sh`
- `scripts/sweep.sh`

Files in this directory are preserved for reference and ad hoc reruns, but they
are not the primary path for current development.
