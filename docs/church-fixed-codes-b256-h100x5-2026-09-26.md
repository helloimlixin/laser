# Fixed-code Church duplicate: batch256 on five H100 NVL GPUs

This run was subsequently preserved at epoch 5/update 2,470 and superseded by
the [batch2,048 throughput continuation](church-larger-batch-throughput-2026-09-26.md)
at the user's request. The original model and optimizer state were retained.

The user requested a fresh duplicate of
[the fixed-code source run](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-fixed-codes-joint-sequence-scratch-b2048-h200x3-20260926)
and explicitly selected the **pasted recipe's effective batch of 256**. The
source run used 2,048. The new run is
[church-fixed-codes-joint-sequence-scratch-b256-h100x5-20260926](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-fixed-codes-joint-sequence-scratch-b256-h100x5-20260926).

The frozen executable archive was recovered from the source run's W&B provenance
artifact. The 420,365,312-parameter LASER RQ backbone and its two causal pair
sequence decoders are retained. Stage2 starts from fresh random weights; none of
the throughput benchmarks' trained weights initialize production.

| Setting | New run |
| --- | --- |
| Dataset | All 126,227 cached LSUN Church training images, unchanged |
| Hardware | Five NVIDIA H100 NVL GPUs, 95,830 MiB each |
| Effective batch | Exactly 256, split 52/51/51/51/51 |
| Accumulation | One; per-image weighting corrects DDP's rank average |
| Final epoch batch | 19 images, split 4/4/4/4/3; no dropping or padding |
| Updates | 494 per epoch; 148,200 over 300 epochs |
| Optimizer | Fused AdamW, LR 0.0005, betas 0.9/0.95, weight decay 0.0001 |
| Schedule | Cosine to zero across 148,200 updates; no warmup |
| Gradient clipping | Global norm 1 |
| Precision / dropout | BF16 training and FP32 loss; residual dropout 0.1 |
| Sampling | Atom top-k 250/top-p 1 per pasted recipe; LASER coefficient temperature 0.9/top-p 0.85 |
| Evaluation | Official RQ FID50k at epoch 1 for verification, then every 5 epochs |
| Checkpoints | Full last/best at evaluations, periodic save every 10 epochs; asynchronous W&B uploads |

The fixed hard atom/coefficient labels are inherited from the named source run.
This duplicates that LASER experiment with the requested optimizer dynamics;
it does not implement the original RQ-VAE stochastic soft-target teacher. The
coefficient sampling parameters have no direct counterpart in the RQ YAML.

The cache SHA256 is
`4c03555ebedca1bf7b74f1db30e91752586889337a58547e856416c3aeeae586`;
the frozen tokenizer SHA256 is
`762c51a10267ed6fa55709ff0d6cf997940d21056b16c7a0a0399b3ebf93868d`.
Both match the source. The exact source real FID reference was recovered from
`church-laser-consistent-rq32k-latest:v0` and matches SHA256
`ad3b5a341831d877110afdff37d6fff4be855612053e3eb9e2e7119f5593d364`.

Seven complete five-GPU training benchmarks compared accumulation and DDP
communication settings, retaining FP32 gradient reduction and batch 256. The
fastest measured configuration used 25 MiB buckets and gradient bucket views,
without static-graph mode, at **516.27 images/sec**. Other configurations ranged
from 410.08 to 511.81 images/sec. These are measured steady-state training rates,
excluding FID and checkpoint overhead, not a claim of a theoretical maximum.
The data/cache and checkpoints use local storage to avoid shared-storage latency.

Thirteen tests against the frozen runtime passed for sequence causality, pair
autoregression and distributed batching. Additional checks confirmed exact
full-epoch coverage and gradient equivalence for batch 256, including the final
19 images, with one and two accumulation steps. Fixed targets replay exactly and
consume no sampling RNG. GPU generation and actual distributed resume receipts
are recorded in the runtime and mirrored output directory. The five-GPU resume
restored all 582 optimizer parameter states and all five rank RNG states, then
advanced optimizer and scheduler together from step 30 to step 32.

Generation batches 2,048, 3,072, 4,096, 5,120 and 6,144 all passed with an additional 8 GiB
allocation reserved for evaluation headroom. Batch 4,096 was fastest at 113.67
images/sec/GPU for generation plus decoding; Inception extraction is additional
work. The larger 6,144 batch was slower at 109.28 images/sec/GPU. Across all 256
events, cached versus dense logits differed by at most 1.34e-5.

Production launched under PID 7301 and completed epoch 1 at update 494. The full
5,045,165,893-byte checkpoint passed model/optimizer finiteness checks, contained
582 Adam parameter states at step 494, and preserved all five rank RNG states.
The scheduler was also at step 494. Production's initial state hash and step-10
losses exactly matched the selected benchmark's fresh initialization.

The initial official FID50k was 123.2651; the full evaluation took 134.3 seconds.
This is an early training measurement. Training continues for the requested 300
epochs. Checkpoint upload completion is recorded in `production-accepted.json`
and `train/checkpoint-upload.json`.

The first full last/best checkpoint artifact is committed online as
`church-fixed-codes-joint-sequence-scratch-b256-h100x5-20260926-selected-checkpoints:v0`.
Its remote sizes and digests were verified. At handoff, production had passed
update 860 and sustained about 521 images/sec; the 300-epoch run remains active.

Runtime: `/mnt/laser-church/duplicate-20260926/run`.
Shared records: `outputs/church-fixed-codes-b256-h100x5-20260926`.
Logs: `/mnt/laser-church/duplicate-20260926/run/train.log`.
Full local checkpoints: `/mnt/laser-church/duplicate-20260926/run/checkpoints/train`.

To resume after stopping the existing process:

```bash
python /mnt/laser-church/duplicate-20260926/run/launch_h100.py --resume
```

The launcher rejects a duplicate active production process. It reads the W&B
credential from the environment or a mode0600 file outside the repository;
credentials are not included in the recipe or source archive.
