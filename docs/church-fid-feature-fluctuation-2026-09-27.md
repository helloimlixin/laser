# Diagnosing Church FID fluctuations

The FID evaluator is behaving consistently. The original fixed-code run and the lower-LR continuation use the same released RQ-VAE Inception/Frechet implementation, 50,000 generated images per checkpoint, the same real reference statistics, and the same sampling settings and seed base. The reference covariance trace is exactly 101.62237 in every logged feature record. The training log verifies finite Inception features and records exactly 50,000 generated samples per evaluation. I found no sign of a changing reference, FID protocol, or broken evaluator.

![Matched-epoch FID and feature diagnostics](../outputs/church-fid-feature-analysis-20260927/fid-feature-decomposition.png)

The logged score decomposes as squared distance between real and generated Inception means plus a covariance-distance term. The covariance term is the larger component, contributing about 65% of total FID. It explains most checkpoint-to-checkpoint movement; generated feature-trace ratios add context but do not determine FID by themselves, since covariance orientation and eigenvalues also matter.

| Matched epochs 40–65 | Original LR | Lower LR |
|---|---:|---:|
| FID range | 10.549–12.233 | 10.665–11.210 |
| FID standard deviation across six checkpoints | 0.562 | 0.183 |
| Mean-term standard deviation | 0.180 | 0.065 |
| Covariance-term standard deviation | 0.398 | 0.130 |
| Generated/real covariance-trace ratio range | 0.822–0.886 | 0.839–0.855 |

These standard deviations describe movement across checkpoints, not measurement confidence or independent-seed FID variance. At epoch 45, the original run jumped from epoch 40 by **+0.423** in the mean term and **+0.922** in the covariance term. Over the same interval, the lower-LR branch moved only **+0.044** and **+0.022**. Across all six matched checkpoints, lower LR reduced FID variation, but it has not beaten the inherited best of **10.4322**; its best by epoch 65 was **10.6649** at epoch 55.

The feature trace ratio stays below 1, so generated features have less total variance than real features, by about 15% in the lower-LR checkpoints. This is a persistent feature-distribution mismatch. It does not by itself explain each FID jump: for example, the original epoch-45 ratio is lower than epoch 40 even as its covariance-distance term rises sharply.

Held-out token likelihood points to overfitting as an underlying training issue. From epoch 40 to 65, original-LR train-probe atom NLL fell **4.130 → 3.101**, while validation NLL rose **10.251 → 11.225** and the gap grew **6.121 → 8.124**. With lower LR, train NLL fell more slowly (**3.947 → 3.364**), validation NLL rose more slowly (**10.329 → 10.794**), and the gap grew **6.382 → 7.430**. That supports excessive fitting of the fixed training codes as a contributor. Lower LR slows the divergence; it has not eliminated it.

The most defensible conclusion is that the observed fluctuation is primarily movement in generated Inception covariance, alongside growing train/validation overfit, rather than an inconsistent FID reference or arithmetic path. A controlled repeated-seed evaluation of the same saved checkpoint would be needed to quantify 50k FID sampling uncertainty; the logged feature summaries do not include per-dimension covariance or raw features. The lower-LR run is active again from the verified epoch-65 checkpoint after its process received SIGTERM. No training settings were changed during this investigation.

## Compared with the earlier stochastic-OMP run

The earlier run the user identified is clearly the stronger observed result: **FID 9.8003 at epoch 114 / update 7068**, versus **10.6649 at epoch 55 / update 5570** for the lower-LR branch (gap **0.8645**). At almost exactly matched update counts, the earlier run scored **10.5456 at update 6820** and the lower-LR branch scored **11.0144 at update 6810** (gap **0.4688**). The lower-LR run's latest recorded FID is epoch 75 / update 6810.

The earlier recipe used stochastic OMP targets and atom top-k 700; the current run uses fixed hard codes and atom top-k 250. So these observed FIDs rank the actual runs, but they do not isolate a single cause. The reduced-LR result so far does not catch the earlier model.

Detailed matched rows and summary calculations are in [analysis.json](../outputs/church-fid-feature-analysis-20260927/analysis.json). The live run is [here](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-fixed-codes-joint-sequence-lr1e4-from-e35-b2048-h100x5-20260926).
