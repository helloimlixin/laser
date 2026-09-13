# ImageNet tokenizer fidelity correction and eight-level refinement

**Current status:** The user subsequently requested eight levels because the
sixteen-level model is too large, and asked to improve eight-level fidelity.
The sixteen-level workflow was prepared and tested but **never launched**.
Its earlier acceptance is superseded. Eight-level coefficient calibration is
being tested with the same frozen checkpoint and matched evaluation protocol.
The sixteen-level preparation details below are retained as historical evidence.
The completed eight-level refit measures **4.407900 rFID-50k**; see
[the current eight-level run notes](imagenet-rq8-refit-stage2-2026-09-13.md).

The accepted mapping uses **sixteen signed coefficient levels**. On the same
50,000 ImageNet validation images, against the same original validation feature
statistics, the frozen source OMP tokenizer measures **4.215117 rFID** and the
sixteen-level RQ mapping measures **4.362995**, a difference of **+0.147878**.
The user accepted 4.363; the replacement workflow uses a **+0.15** drift gate.

Replacement run:
https://wandb.ai/helloimlixin-rutgers/laser/runs/imagenet-rfid421-rq16-480m-fidelity015-20260913

Output: `outputs/imagenet-rfid421-rq16-fidelity015-20260913`.

## Corrected evaluation

The initial eight-level report incorrectly compared 6.726 against the source
checkpoint's 4.21. The 6.726 value used the released **training-reference**
statistics; the source checkpoint's rFID compares reconstructed validation images
against the **original validation images**. Those scores are not comparable.
The original workflow was stopped during caching, before any stage-2 updates.
The metric correction is recorded in that run's W&B summary and in
`outputs/imagenet-scaled-rq-stage2-20260913/metric-protocol-correction.json`.

The corrected full evaluation uses the released Inception implementation,
50,000 original validation images, matching resize/center-crop transforms, FP32
encoder/decoder/quantizer computation, and FP64 accumulated feature moments.
All coefficients are fitted on training images only. The original validation
statistics, each reconstruction's statistics, selected indices, and complete
results are retained in
`outputs/imagenet-tokenizer-fidelity-20260913/full-matched-first`.

| Frozen tokenizer | Matched rFID-50k | Increase over original |
| --- | ---: | ---: |
| Original OMP4 | 4.215117 | — |
| Eight-level scaled RQ | 4.492764 | +0.277647 |
| **Sixteen-level scaled RQ** | **4.362995** | **+0.147878** |

The original result reproduces the source run's recorded 4.210914 within 0.0043.
A separate 4,096-image comparison screened pursuit geometry and coefficient
precision, but those subset FIDs were not used to approve production. A 32-level
full-validation comparison was stopped when the user accepted the sixteen-level
result; it supplies no completed full-validation result.

## Mapping and quality gate

The mapping remains compatible with the Church-style expanded RQ codebook:

```text
0                          = zero vector
1 + atom_id * 16 + bin      = levels[bin] * dictionary[:, atom_id]
```

The vocabulary has 262,145 IDs, stored as uint32 in the cache. Encoder, original
16,384-atom dictionary, and decoder remain frozen. Four residual codewords are
selected at each 8×8 latent site; earlier contributions stay fixed. Both released
transformer conditioning paths use the decoded physical codeword vectors.

The selected codebook uses the exact fitted tensors evaluated in the full study.
Its SHA256 is
`75ebbf03b9ca97f7d61203bf75a3448f786ed3bf1f0a10f1baef4762b38cc8eb`.
The source checkpoint SHA256 remains
`dd28db9306d526bdc9fbf8016403e106c4f317fd8f5c04792f0953e53310fdab`.

`fidelity-gate.json` binds the accepted full-validation result to both hashes and
the original-validation statistics. Production refuses a different checkpoint or
codebook, a subset-only result, unmatched references, nonfinite metrics, failure
to reproduce the source baseline within 0.02, or a drift above the configured
limit. The launcher and training driver both check the gate. After building the
validation token cache, the production decoder repeats the matched rFID audit
and must pass before it proceeds to the training cache.

The default gate remains +0.1. This accepted run explicitly supplies +0.15. A test
verifies that its measured result fails +0.1 and passes +0.15.

## Training and cache

The smallest released ImageNet configuration is retained: width 1536, 12 spatial
layers, four depth layers, 24 heads, and 1,000 conditioning labels. Enlarging the
output vocabulary gives **858,655,745 parameters**. The original optimization
recipe remains 100 epochs, effective batch 2,048, AdamW at LR 0.0005, betas
(0.9, 0.95), weight decay 0.0001, clipping 1.0, cosine decay, and no warmup.

Four H200 GPUs use microbatch 64 and eight accumulation steps. SDPA and fused
AdamW retain the released attention semantics and optimizer grouping. Model
computation uses FP16 autocast and GradScaler; tokenizer geometry and exact
full-vocabulary soft cross entropy use FP32. Stochastic codewords and soft
targets are regenerated at every visit, with sampled-prefix conditioning.

The sixteen-level mapping's target temperature is **0.0625**. Calibration on 128
held-out training images measured a sampled/hard latent-MSE ratio of 1.00964,
versus 1.00891 for the released ImageNet RQ tokenizer at temperature 0.5. The
previous temperature 0.125 exceeded the original-control-plus-0.02 criterion.

Training retains the two precomputed random crop/flip views per image, one
selected per epoch. Previously completed encoder-cache segments are reusable
because the frozen encoder and image transforms are identical. Reuse verifies
the source checkpoint, manifests, seed, precision, rank layout, and view count;
it copies only flushed segments and regenerates the sixteen-level token IDs.
It does not reuse the rejected eight-level IDs or modify the old cache.

Sampling, validation, FID checks, checkpointing, and restart behavior follow the
initial driver. The training configuration and source snapshot are saved beside
the run. The production process is detached and automatically proceeds from cache
construction to fresh transformer training.

## Verification and operation

Thirteen focused tests passed, including gate failures and the accepted +0.15
case. The full 859M model completed three production-sized updates, strictly
reloaded its checkpoint, verified finite model/optimizer states, checked class
conditioning and full-versus-chunked CE, and decoded valid sampled IDs. No AMP
updates were skipped. The final two benchmark updates processed about 352
images/second, with peak allocated memory of 58.26 GiB per GPU. These are short
preflight measurements, not a full-epoch throughput guarantee.

Initial stage-2 weights SHA256:
`3e8e84bc49b031968fa5dba51c9231537393d98887d34d1e9f0cfbf8523e9759`.
Production does not load the preflight weights.

Inspect the live workflow:

```bash
cat outputs/imagenet-rfid421-rq16-fidelity015-20260913/train/status.json
tail outputs/imagenet-rfid421-rq16-fidelity015-20260913/production.log
```

Resume after it has stopped:

```bash
.venv-imagenet-stage2/bin/python scripts/tools/launch_imagenet_scaled_stage2.py \
  --base outputs/imagenet-rfid421-rq16-fidelity015-20260913 \
  --run-id imagenet-rfid421-rq16-480m-fidelity015-20260913 \
  --max-rfid-drift 0.15 \
  --reuse-cache outputs/imagenet-scaled-rq-stage2-20260913/cache \
  --resume
```
