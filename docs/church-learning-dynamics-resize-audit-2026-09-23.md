The current run preserves the original September 23 A100 run's effective batch,
optimizer, and LR schedule on four H100s. It does **not** reproduce the learning
dynamics of the September 21 eight-H200 Church run that achieved FID 11.85.

| Setting | Earlier 8-H200 Church | Original 4-A100 Church | Current 4-H100 Church |
| --- | ---: | ---: | ---: |
| Effective batch | 1,536 | 256 | 256 |
| Optimizer updates per epoch | 82 | 493 | 493 |
| Peak learning rate | 0.0005 | 0.0005 | 0.0005 |
| Cosine horizon, epochs | 90 | 300 | 300 |
| LR after epoch 50 | 0.000206588 | 0.000466506 | 0.000466506 |
| FID50k at epoch 50 | 11.851648 | 16.398817 | 15.119682 |
| Best logged FID50k at audit | 11.851648 | 15.303452 | 13.271944 |

W&B runs:
[earlier Church](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-laser-ft3best-compound-nogeom90-h200x8-20260921),
[original A100](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-ffhq-compound-350m-4a100-20260923),
[current H100](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-ffhq-compound-350m-4h100-jointgeom-scratch-20260923).
The current epoch-70 evaluation subsequently reported FID 13.707084; the best
remains 13.271944 at epoch 55. These are generation FID with 50,000 samples.

At a matched image epoch, the current recipe makes approximately six times as
many updates and decays LR more slowly. Its epoch-50 LR is 2.26 times the
earlier run's. GPU type alone does not require an LR change when the effective
batch and optimizer schedule are preserved. However, preserving the A100
recipe is different from matching the earlier large-batch run. Logged LRs for
all three runs agree with their respective cosine formulas to below 1e-12.
The active trainer divides accumulated loss by the accumulation count and
advances the scheduler once per optimizer update. There is no evidence here
of an accidental LR reset or a missing gradient-accumulation divisor.

Other material differences prevent attributing the FID gap solely to LR.
The earlier Church recipe used physical-space coefficient targets with
temperature 0.125 and no geometry penalty. The current recipe uses normalized
targets with temperature 0.5 and corrected conditional geometry weight 0.05.
Coefficient scales also differ. Temperatures in these different units are
not directly comparable. The A100 run's final mutable W&B config includes the
geometry correction applied near its end; it does not describe the objective
used for most of that run's history. The old Church sampling sweep's 10.938236
result also used a different coefficient sampler (top-p 1 rather than 0.85).

The [FFHQ 8.174393 run](https://wandb.ai/helloimlixin-rutgers/laser/runs/ffhqcmp0804205803)
used effective batch 128, peak LR 0.0005, and a 200-epoch cosine horizon.
Its architecture motivated this Church implementation, but its numerical FID
is not directly comparable across datasets and real-reference distributions.
Copying the architecture does not establish optimal Church hyperparameters.

The FID resize audit found no bilinear/bicubic mismatch. The frozen production
evaluator uses float32 RGB in [0,1], resized from 256 to 299 by bilinear
interpolation, with `align_corners=False` and `antialias=False`, then normalized
to [-1,1]. The local Inception forward function has an identical syntax tree
to the [released RQ-VAE evaluator](https://github.com/kakaobrain/rq-vae-transformer/blob/main/rqvae/metrics/inception.py).
A fresh CPU probe passed identical synthetic pixels through both `real=True`
and `real=False` metric entry points. A hook before the first Inception block
verified bitwise-identical inputs for both paths and the explicit bilinear
calculation. The probe stops before feature extraction; it is not a full FID
recomputation.

Production FID loads precomputed real statistics. The active reference hash is
`809489d8316b9e6eb9dc3bc021b6d602f4b6d816cc80621c6b9c189a9253a7f6`,
matching the official Church reference previously verified byte-for-byte in
the [September 14 protocol audit](church-fid-protocol-audit-2026-09-14.md).
The earlier run's historical reference path is unavailable on this host; the
earlier audit records the same official hash. The published real-statistics
code uses the same Inception preprocessing as generated images. No full real
reference was regenerated in this audit. The separate resize-to-256 source
transform is bilinear plus center crop; the previous full dimension audit
found all originals already had a 256-pixel shorter side, making that resize
a no-op. See the [native-256 audit](church-native256-fid-2026-09-14.md).

Switching to bicubic would require regenerating real statistics and generated
features under the same new protocol. The current standard-FID comparison
should retain its verified bilinear preprocessing.

No live training settings were changed during this audit. A controlled
comparison matching the earlier Church dynamics can use effective batch 1,536
on four GPUs (for example, batch 64 per GPU and six accumulation steps), with
the earlier schedule and targets. Changing only LR mid-run would not reproduce
that baseline or isolate the contribution of target noise and geometry loss.

Evidence is saved in the current run's `dynamics-fid-audit/`: compact numeric
comparisons, the resize probe and its result, and this report. Local working
snapshots of W&B config/history are under
`/mnt/laser-church/dynamics-fid-audit-20260923`.
