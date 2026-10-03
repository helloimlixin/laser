The repair has not broken FID 10. Its best is **10.6070 at epoch 40**; the latest completed FID50k is **10.7077 at epoch 50**. The earlier Church checkpoint scored **9.6012** using the same official reference. This does not support a universal FID-10 limit for the tokenizer/model family.

The prior relaunch changed coefficient noise, dropout, and geometry loss, but retained one fixed cached atom sequence per image. It did not implement a residual-conditioned stochastic atom teacher. Calling those changes a resolution of the exposure-bias concern would be incorrect.

The old monitor used coefficient temperature 0.5. This audit checks that its overfitting signal is not merely a history-distribution mismatch: the same 300 training and 300 disjoint validation images are evaluated at epochs 10 and 40 using both nearest-bin histories and one fixed stochastic history drawn from the actual repaired target distribution. Weights load strictly from immutable checkpoints, with SHA256 identities recorded. The audit performs no encoding, refitting, gradient updates, or production changes.

| Actual repair history metric | Epoch 10 | Epoch 40 |
|---|---:|---:|
| Train atom NLL | 8.3991 | 5.0257 |
| Held-out atom NLL | 8.7663 | 10.0191 |
| Train coefficient KL | 0.5494 | 0.3910 |
| Held-out coefficient KL | 0.5923 | 0.6378 |
| Train first-atom accuracy | 7.90% | 69.04% |
| Held-out first-atom accuracy | 6.99% | 5.51% |

The held-out atom-NLL increase is +1.2528 nats, paired-image bootstrap 95% interval [1.2302, 1.2771]. All four atom depths worsen. Nearest-bin histories show the same direction, with held-out mean atom NLL 8.7326 to 10.0044. These intervals resample images under fixed histories; they do not cover training-seed or FID variation. The audit took 40 seconds and peaked at 2.26 GiB allocated GPU memory.

The divergence exists with teacher-provided histories and at the first atom within each spatial site. Therefore a problem confined to later within-site generated prefixes is not a complete explanation. This does not exclude spatial or within-site exposure bias, identify the best replacement objective, or turn token accuracy into an image-quality metric. FID improved substantially from epoch 10 to 40 even as held-out token likelihood worsened; early stopping at the minimum token NLL would have selected FID 28.53.

The stronger historical recipe had 16 stochastic OMP trajectories, conditional soft atom labels, different coefficient scales, batch 2048, a different schedule, and sampler top-k 700 / coefficient temperature 0.9. The repair uses a single fixed atom sequence, batch 1024, and sampler top-k 250 / coefficient temperature 1.0. These differences remain confounded. The bank is a leading training-target hypothesis, not a demonstrated cause; its existence is not permission to introduce refitting. Previous sampler confirmations evaluated the predecessor checkpoint, not this repaired epoch-40 checkpoint.

The next teacher experiment should first establish a no-refit encoding with acceptable held-out reconstruction and prefix-consistent targets. A controlled training comparison should hold the tokenizer, architecture, effective batch, optimizer schedule, image budget, and evaluation protocol fixed while changing that teacher. Token likelihood is a diagnostic, not a sufficient launch gate or FID success criterion. A sampler-only comparison, if undertaken, should use this frozen best checkpoint and independent 50k confirmation draws. Neither change should be reported as a fix before it produces a reproducible FID gain.

Production remains running and unchanged. Best and latest full checkpoints continue uploading. No additional training was launched for this audit. Results, per-image arrays, plot/PDF, checkpoint identities, source, and a snapshot of the production curves are published in the linked W&B audit run.

Evidence: `outputs/church-repair-plateau-audit-20260928/actual-histories.json`, `per-image.npz`, and the existing [matched FID evaluation](church-matched-official-fid-2026-09-27.md). The audit does not yet provide a demonstrated fix for the plateau.
