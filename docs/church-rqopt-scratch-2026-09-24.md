# Church RQ optimization from scratch, September 24, 2026

W&B run: https://wandb.ai/helloimlixin-rutgers/laser/runs/church-compound-rqopt-scratch-b2048-4h100-20260924

This starts a fresh stage-2 transformer using the configuration of
`helloimlixin-rutgers/laser/church-compound-rqopt-b2048-4h100-20260924`.
Transformer weights, optimizer, RNG trajectory, epoch, and global step start
fresh. The frozen stage-1 tokenizer and its validated 126,227-image Church
token cache are retained. No stage-2 checkpoint initializes this run.

The training configuration is
`configs/stage2/lsun-church-rqopt-scratch-4h100.yaml`.
Four H100 GPUs use batch 128 each and four accumulation steps, giving global
batch 2,048. Training runs for 300 epochs (18,300 optimizer updates), with
learning rate 0.0005 and cosine decay to zero without warmup. Physical
coefficient targets retain temperature 0.125 and atom loss weight 1.5.
Sampling retains atom top-k 250 and coefficient top-p 0.85.

Online W&B uploads use the fixed run-file names `last.pt` and
`best-fid-01.pt`. Both are full checkpoints with optimizer, scheduler and
per-rank RNG state. FID uses 50,000 generated images and the original RQ-VAE
Inception network/reference statistics every five epochs. The first best-FID
checkpoint is selected after epoch five. Recovery saves occur every 1,000
updates, with additional saves at every FID evaluation and every ten epochs.

The run directory is
`outputs/church-compound-rqopt-scratch-b2048-4h100-20260924`.
It contains the source run's online configuration, frozen source archive and
checksums, independent tokenizer/cache assets, package versions, resolved
recipe, test results, startup checks, logs, and launch receipt. The training
source matches the reference run. The supervisor's startup-report metadata
was adjusted to report the actual geometry setting.

The detached supervisor passed a two-update training/checkpoint check and a
generation check before launching production training. The full checkpoint
contains finite model/optimizer tensors, four RNG states, and scheduler step
2 of 18,300. Generation passed at 128 images per GPU, with peak allocated
memory of 56.32 GiB per GPU. Checkpoint handling and model tests passed (39
tests). The original RQ-VAE Inception forward pass and Church reference
statistics were also checked. Production started at scheduler step zero.

The subsequent autoregression audit verified the exact frozen runtime against
its manifest and the active launch configuration. All 13 causal/history tests
passed. Additional interventions confirmed that changing an earlier generated
coefficient changes subsequent atom and coefficient predictions, and that the
generation loop passes the selected atom's actual dictionary vector to the
coefficient head. Teacher forcing uses the same dictionary-vector conditioning.
The chain is `p(a_i | a_<i,c_<i) p(c_i | a_<=i,c_<i)` across spatial sites and
sparse depths. Reports and the reproducible check are in
`autoregression-audit/` inside the run directory. Training continues with the
verified architecture.

To inspect progress, read `training.log`, `status.json`, and the W&B run.
Once this new run has saved a production checkpoint, resume it with:

```bash
/mnt/laser-church/venv/bin/python \
  outputs/church-compound-rqopt-scratch-b2048-4h100-20260924/resume.py
```

The resume entry point refuses to run without a complete production
`last.pt`; it restores this new run's own training state. Credentials are
stored privately outside the repository and are excluded from run artifacts.
