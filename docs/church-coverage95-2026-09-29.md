The Church sampler now enforces at least 95% of the full joint-token probability mass at every prediction in all four RQ depths. It retains the original top 1,400 atom–coefficient pairs (including boundary ties), adds the probability-ranked prefix needed to reach the floor, and samples from their union. There is no maximum token count. Temperature remains 1.0; each sampled integer still denotes one complete atom–coefficient vector.

The user explicitly prioritized coverage at all depths. This is a sampling constraint, not a claim that the model probabilities are calibrated. Coefficient-bin diagnostics are recorded separately; no equal-bin quotas, refitting, reward, or training-objective changes were introduced.

The corrected implementation passed an audit of all 12,800,000 conditional distributions in a 50,000-image generation. Every depth had 3,200,000 observations and zero coverage violations:

| RQ depth | Original top-k mean coverage | New mean coverage | New minimum coverage |
|---|---:|---:|---:|
| 1 | 98.798% | 98.961% | 95.000970% |
| 2 | 92.485% | 96.127% | 95.000929% |
| 3 | 76.342% | 95.084% | 95.000887% |
| 4 | 56.818% | 95.005% | 95.000899% |

The matched checkpoint is step 3,100, after 50 completed epochs. FID uses the same eight RNG streams, frozen decoder, continuous pixels, official Inception/reference, and 50,000 images for each policy:

| Policy | FID50k | Mean term | Covariance term |
|---|---:|---:|---:|
| Original top-k 1,400 | 12.010617 | 4.371587 | 7.639029 |
| Top-k floor plus 95% coverage | 12.645119 | 4.806802 | 7.838316 |

The coverage constraint increased FID by 0.634502 at this checkpoint. Adoption follows the user's coverage requirement. The earlier best FID 10.584468 at step 2,480 remains preserved in the parent run; it is a different checkpoint and is not the matched baseline above.

The first implementation missed the original numerical tolerance in one of 12.8 million rows (retained mass 0.9499988556). That attempt was rejected and archived. The corrected implementation targets an internal cutoff of 0.95001 and uses the full vocabulary if the retained normalizer is still below the requested floor. The audit tolerance remains 1e-6. All 22 CPU tests passed, including broad-tail production-vocabulary distributions and an injected undercoverage fallback; native top-k GPU parity passed on all eight ranks.

The exact continuation starts from the full step-3,100 checkpoint (SHA256 `b07966ad06f645d722bda0e372c61af4aa05c9a24168df4c78be98e2c2006c14`). All eight ranks verified restored model bytes, scheduler and RNG state; rank 0 also verified every Adam tensor and parameter group. Training retains global batch 2,048, microbatch 256 on eight GPUs, and the original 18,600-update schedule for 300 total epochs. Previews remain every 200 updates; FID remains every 620. The continuation maintains its own BEST under the new sampling policy and uploads full LAST/BEST checkpoints.

[Active continuation](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-coverage95-20260929) · [Matched sampling comparison](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-coverage95-sweep-20260929-r2) · [Preserved parent run](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-joint65k-20260929)

Runtime: `/tmp/laser-church-coverage95-20260929`. Durable source, recovery assets, tests, sweep evidence, and rejected numerical attempt: `outputs/church-coverage95-20260929`.

Launch verification: all eight GPUs resumed, with step 3240 observed. W&B source/recovery manifests are verified; initial full LAST and BEST are committed as `helloimlixin-rutgers/laser/church-coverage95-20260929-selected-checkpoints:v0`. Runtime evidence is in `launch-verified.json` and the step-3,100 publication receipts.
