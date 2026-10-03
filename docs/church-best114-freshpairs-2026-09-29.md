# Church continuation with fresh paired sparse-code targets

The fixed16-trajectory cache averaged4.85 unique complete paired trajectories per latent site, and93% of sites had one first atom across all16 entries. This supports testing broader positive targets; it does not establish the cause of the FID floor.

The new continuation compares the original bank teacher with fresh full-vocabulary targets around the complete physical sparse vector. Both start from the preserved original epoch114/step7068 model and all517 Adam states. That checkpoint scored9.538903 under the matched50k protocol (historical reported FID9.800315). The preceding200-update loss-weight trials all regressed to10.93–11.56 and remain preserved separately.

The new teacher draws a paired anchor from the original cache and reconstructs its complete vector. It then updates one support–coefficient pair at a time, scoring each candidate against the residual after subtracting the other current pairs. Atom probabilities cover all16384 atoms except the other occupied supports, marginalizing both coefficient signs at the current magnitude. The conditional coefficient draw uses all2048 fixed physical bins. There is one warmup sweep and one final ascending sweep; the latter records soft labels at the time of each draw, so later updates cannot change an already emitted prefix. A single model forward pass learns these paired conditional targets. The finite procedure is not claimed to sample the equilibrium distribution exactly.

The tokenizer, dictionary, coefficient scales and quantization bins are frozen. There is no coefficient refitting, FID reward, geometry penalty or contrastive loss. The original normalized atom/coefficient CE weighting is1.5:1 in both arms. Teacher randomness is separate from model dropout and both RNG states are saved.

Physical temperature0.4204482076268573 was calibrated on512 cache sites, with512 independent check sites and four random draws per site. It matches the original coefficient-noise whole-vector squared perturbation to+3.17% on the fitting sites and+5.41% on the independent check sites. On the check sites29.50% of sampled atoms lie outside the union of all stored supports. No FID features or model training were used to select this temperature.

Both arms use a quarter of the original learning rate: resumed LR0.00008550778454279245, cosine base0.000125, original phase7068 and horizon18600. Adam moments and step counters are preserved. The multiplier is applied once at the historical bootstrap and never again on same-study resume. This is a conservative response to the original run's large adjacent-epoch FID oscillations, not evidence that0.25 is optimal.

The pilot uses four GPUs per arm, global batch2048, microbatch128 and accumulation4, for200 updates. The control/fresh comparison uses the same eight logical evaluation RNG streams, native sampler,50k images, image preprocessing and official Church reference. Samples are logged every200 updates; full LAST and best-FID model/optimizer/scheduler/RNG checkpoints are published to W&B. If fresh improves matched control FID by at least0.10, its full LAST continues on eight GPUs; otherwise control continues. This practical threshold is not a significance test. Heldout loss is not a veto. The original best checkpoint remains best until a lower matched FID is measured.

Validation includes15 focused CPU tests, an independent enumeration of a192-state signed-Gibbs example, two-rank real-model forward/backward/optimizer smoke tests, finite gradients for all517 parameters and matching model RNG fingerprints. Full microbatch128 smoke tests and the launch state are recorded in the deployment receipts. The earlier shared-GPU microbatch64 smoke tests are preserved separately and are not evidence of uncontended throughput.

Runtime: `/tmp/laser-church-best114-freshpairs-20260929`.
Durable records: `outputs/church-best114-freshpairs-20260929`.
Implementation: `src/training/fresh_paired_teacher.py`; tests: `tests/test_fresh_paired_teacher.py`.

## Verified launch

Both runs launched on September29,2026 around00:45UTC. All eight ranks restored the original7068 checkpoint and517 Adam states, used LR8.550778454279245e-5, and completed training updates. Full microbatch128 smoke tests passed before production. Peak allocated memory was36.18GiB for control and40.56GiB for fresh. Corresponding ranks across both pilots have identical model-dropout RNG fingerprints while teacher targets differ. W&B confirms both runs are running and each source artifact contains67 files.

- Fresh paired teacher: https://wandb.ai/helloimlixin-rutgers/laser/runs/church-best114-freshpairs-20260929-fresh
- Original bank control: https://wandb.ai/helloimlixin-rutgers/laser/runs/church-best114-freshpairs-20260929-control

Early four-GPU throughput is approximately1.38seconds/update for fresh and0.96seconds/update for control; these are initial observations, not long-run averages. The first samples and matched50k FID are scheduled at step7268. No new FID result was available at launch.

The outgoing equal-weight continuation was captured at completed step8368 with every rank's update intent equal to8368, then retired. Its complete LAST/BEST and optimizer state are preserved separately from the original7068 bootstrap; W&B publication runs asynchronously. The controller selects a pilot after both50k evaluations and verified full LAST/BEST uploads, then continues its full LAST on all eight GPUs.

## Matched pilot result at step 7268

Both arms completed 200 updates and matched 50,000-image FID evaluation. Metrics and sample grids were verified through the W&B API.

| Model | FID | Mean term | Covariance term |
|---|---:|---:|---:|
| Original parent, step 7068 | 9.538903 | 3.027131 | 6.511772 |
| Historical bank control, step 7268 | 10.429794 | 3.318565 | 7.111229 |
| Fresh paired teacher, step 7268 | 9.631140 | 3.066115 | 6.565024 |

Fresh improves FID by 0.798654 and covariance distance by 0.546205 relative to the matched control, satisfying the continuation rule. It remains 0.092236 FID and 0.053252 covariance distance above the starting checkpoint. This comparison shows less regression during continuation; it does not establish that the original FID floor is resolved. The original parent remains the best checkpoint.

Both arms' complete LAST/BEST checkpoint pairs were verified on W&B as their `selected-checkpoints:v1` artifacts. The controller continues the fresh full LAST after durable publication and a clean pilot exit.

The selected fresh run resumed successfully from full step 7268 on all eight GPUs. Adam state and cosine phase are preserved, with no second learning-rate reduction. Early measured steady update time is approximately 0.71 seconds at global batch 2,048; the next scheduled matched FID is at step 7440.
