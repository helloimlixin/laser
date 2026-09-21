# Sampling comparison — 2026-09-15

The **50,000-sample confirmation is complete**, evaluated against all **126,227
training images** on the fixed epoch-137 checkpoint. Temperature 1/top-p 0.98 with
no top-k cap scored **11.9058**, compared with **12.2517** for temperature 1/top-k
1400/top-p 1: a reduction of **0.3459**. This uses one independent 50k seed per
setting, not a significance test. The original preprocessing is preserved.

| Setting | Generated samples | Real training images | FID50k |
|---|---:|---:|---:|
| baseline: temperature 1, top-k 1400, top-p 1 | 50,000 | 126,227 | 12.2517 |
| nucleus: temperature 1, no top-k, top-p 0.98 | 50,000 | 126,227 | **11.9058** |

The [full results and protocol](../outputs/church-sampling-sweep-20260915/fid50000-seed73000/results.json)
and [sample accounting check](../outputs/church-sampling-sweep-20260915/fid50000-seed73000/sample-integrity.json)
record the checkpoint, seed 73000 + rank, reference hashes, and exactly one
occurrence of every generated sample index. This checkpoint predates the code
repairs; these scores measure the sampling change.

Checkpoint availability was checked on 2026-09-15 following the request to
evaluate the best historical checkpoint. The current run's best diagnostic
FID4096 is 14.223 at epoch 40. Its best periodic FID50k is 12.4884 at epoch 100.
Neither set of weights was retained: the running driver writes only `last.pt`,
including when called with a best-checkpoint filename. `best-fid.json` contains
the historical score, not the corresponding weights. No backup was found in
the local checkpoint search, either run's W&B uploads, or matching project model
artifact collections. The older reference run's best checkpoint was also
removed in the earlier cleanup; its remaining checkpoint is epoch 76 with a
different tokenizer. Epoch 137 remains the retained checkpoint with the lowest
verified FID50k among the current run's periodic evaluations and this sweep.
The [search receipt](../outputs/church-sampling-sweep-20260915/best-checkpoint-search/summary.json)
records the evidence and verifies its checkpoint hash. No new sampling run was
launched; evaluating epoch 40 or 100 requires a backup of those exact weights.

Pure nucleus sampling with **temperature 1.0, top-p 0.98, and no fixed top-k cap** was the best tested candidate. It scored 14.317 versus 14.549 for the baseline in the initial screen, and 14.390 versus 14.549 in an independent repeat. Mean FID4096 was **14.354 versus 14.549**, a reduction of **0.195**.

The gain is modest. This is a 4,096-sample screening result on one checkpoint, with two seeds for the selected candidate and baseline. It is not a FID50k result or a statistical-significance claim. The other candidates have only one evaluation each.

Reported FID requires **50,000 generated samples against the entire training set: 126,227 real images**. The confirmation is recorded in its [W&B run](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-sampler-ep137-fid50k-20260915), [status](../outputs/church-sampling-sweep-20260915/fid50000-seed73000/status.json), and [launch record](../outputs/church-sampling-sweep-20260915/fid50000-launch.json). The table below remains historical diagnostic screening evidence.

| Setting | Temperature | Top-k | Top-p | FID4096, seed 71000 | Repeat, seed 72000 |
|---|---:|---|---:|---:|---:|
| baseline | 1.0 | 1400 | 1.00 | 14.549 | 14.549 |
| wider_2800 | 1.0 | 2800 | 1.00 | 14.593 | — |
| wider_5600 | 1.0 | 5600 | 1.00 | 14.822 | — |
| cooler | 0.9 | 1400 | 1.00 | 14.487 | — |
| warmer | 1.1 | 1400 | 1.00 | 15.028 | — |
| nucleus_95 | 1.0 | full vocabulary | 0.95 | 14.533 | — |
| nucleus_98 | 1.0 | full vocabulary | 0.98 | 14.317 | 14.390 |
| depth_dependent_k | 1.0 | [1400, 1400, 2800, 2800] | 1.00 | 14.700 | — |

![Sampling FID comparison](../outputs/church-sampling-sweep-20260915/sampling-fid4096.png)

All settings used the frozen epoch-137 / optimizer-step-8,494 checkpoint, the same 32,769-token codebook and tokenizer, and the real-image reference rebuilt from the original training loader. Each evaluation used exactly 4,096 generated images, two GPU ranks, sampling batches of 100 per rank, decoder/Inception batches of eight, and a reset seed of 71000 + rank or 72000 + rank. Decoder outputs were mapped directly to [0,1] and clipped; no additional image resizing or cropping was applied. The original Inception preprocessing remained identical to that used for the real reference. Sampling used the released cached model with FP16 autocast; decoding/Inception used FP32, moments used FP64, and TF32 was disabled.

Increasing top-k alone did not improve the first screen. The adaptive nucleus rule is more consistent across depths: on generated prefixes from the first batch of the initial run, its average candidate counts were approximately **255, 809, 1,693, and 2,049**. Thus it narrows early-depth distributions while retaining more possibilities at later depths. These diagnostics describe generated contexts specific to each sampler; they do not isolate which depth caused the FID change.

The repeat baseline FID was unusually close to the original baseline. Saved token sequences and feature statistics were checked: there were no identical aligned full code sequences, and mean/covariance statistics differed. The repeat was not a reuse of the first sample set. Every initial-screen setting had 4,096 distinct full token sequences, with every global sample index represented exactly once. This checks sample accounting, not freedom from visual mode collapse.

CPU preflight checks confirmed that diagnostic instrumentation preserved the released sampler’s token draws and RNG states for scalar top-k, full-vocabulary nucleus filtering, and depth-specific top-k. A GPU preflight exercised the full evaluation path. The ongoing training job continued throughout; its sampler and historical metrics were not changed. The evaluator stays available as a separate reproducible tool.

The 50,000-sample comparison supports p=0.98 as the preferred setting among the two finalists on this checkpoint. The new evaluator defaults to 50,000 generated samples, verifies that the real reference covers the entire audited training set without padding or dropped images, and requires `--diagnostic` to allow a different generated count. Results and W&B configuration record the generated and real sample counts separately. The original real-data loader, resize/crop, and Inception preprocessing are preserved.

W&B: [eight-setting screen](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-sampler-ep137-20260915), [independent repeat](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-sampler-ep137-repeat-20260915).

Artifacts: [summary](../outputs/church-sampling-sweep-20260915/summary.json), [CSV](../outputs/church-sampling-sweep-20260915/comparison.csv), [plot PDF](../outputs/church-sampling-sweep-20260915/sampling-fid4096.pdf), [baseline grid](../outputs/church-sampling-sweep-20260915/screen4096/baseline/samples.png), [p=0.98 grid](../outputs/church-sampling-sweep-20260915/screen4096/nucleus_98/samples.png), [full protocol and launch records](../outputs/church-sampling-sweep-20260915/README.md), and [evaluator](../scripts/tools/evaluate_church_sampling_sweep.py).
