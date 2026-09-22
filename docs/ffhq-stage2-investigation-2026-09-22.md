The FFHQ stage-2 investigation keeps the selected epoch-47 tokenizer frozen
and preserves the original epoch-50 best and epoch-75 last prior checkpoints.
The original best generation FID is 29.4796168 with 50,000 samples against
RQ-Transformer's released FFHQ training reference. This is a comparison using
the same reference statistics, not identical training splits or architectures.
Both original checkpoints were verified online in artifact version 12.

Run directory:
`outputs/ffhq256-var341-scratch-20260921/investigations/stage2-20260922/`.
All large outputs reside on the local `/tmp` filesystem through the existing
run-directory symlink. The diagnostic and pilot runtime snapshots are separate
from the completed production runtime.

The prefix diagnostic forces a contiguous prefix of validation codes and
samples everything after it. Unit checks ensure future target codes cannot
affect the generated suffix. These conditional experiments are not eligible
for unconditional FID comparisons or checkpoint promotion. The diagnostic
uses 2,000 unique validation examples, the same published reference statistics,
three ranks, batch 64 per rank, an FP32 decoder, continuous [0,1] pixels,
and released pytorch-fid Inception. Real-code rows use the same deterministic
validation subset. All image previews use an 8-by-8 grid.

| Real-code prefix | Diagnostic FID, 2,000 outputs |
| --- | ---: |
| None: unconditional baseline | 37.6209 |
| 1x1 | 37.0817 |
| Through 2x2 | 28.7466 |
| Through 4x4 | 23.1966 |
| Through 8x8 | 17.6178 |
| All scales: reconstruction | 13.4288 |

These results show how much supplying real spatial context helps this model.
They do not isolate a causal defect in parallel sampling, and FID differences
cannot be added or assigned to scales as independent error contributions.
The 2,000-output reconstruction score cannot be compared directly with the
tokenizer's 10,000-image reconstruction FID of 4.84459.

The first unconditional sampling sweep keeps the weights and all seeds fixed.
Cooling or warming coefficient logits did not improve FID. Changing atom
temperature from 1.0 to 0.6 improved the 2,000-sample diagnostic from 37.6209
to 33.8164. Temperature 0.5 scored 35.2513, 0.7 scored 34.526, and 0.85 scored
35.3677. Applying the original VAR atom filter (top-k 900, top-p 0.96) worsened
FID to 49.1473. These are screening scores; a fresh-seed, larger evaluation
is required before promoting an inference policy.

Sampler changes add optional atom/coefficient temperatures, coefficient
nucleus filtering, and a diagnostic-only teacher prefix. Defaults preserve
the original inference settings. Normal generation configuration permits
only unconditional sampling controls; it cannot enable teacher forcing.
Both the periodic diagnostic and RQ-reference evaluator record their sampling
settings in their result files.

The isolated training pilot tests normalized per-position scale loss weights
`[1, 8, 4, 2, 1]` for scales `[1, 2, 4, 8, 16]`. The final scale's share of the
loss changes from 256/341 (75.1%) to 256/481 (53.2%), giving more weight to
the earlier spatial scales. Architecture, tokenizer, data, stochastic cache,
batch 384, and sampling settings are preserved. It fine-tunes the epoch-50
weights for ten epochs at learning rate 0.00003, transferring the optimizer
and RNG states. This is an explicit objective-change experiment, not an
unchanged resume; the original and new contracts and parent checkpoint hash
are stored in `scale-weighted/fine-tune-initialization.json` and checkpoint
provenance. The baseline best-FID checkpoint remains selected until beaten.

The pilot uses a separate online W&B run and uploads the full last, best-FID,
and best-validation checkpoints. Its supervisor retains the existing bounded
retry/progress-watchdog behavior and verifies committed online artifacts.

Tests passed for teacher-prefix isolation, full teacher-forcing roundtrip,
temperature validation, normalized scale weighting, gradient finiteness,
checkpoint compatibility, prevention of teacher forcing through production
sampling settings, continuous evaluation pixels, and the existing pipeline
and supervisor checks. The most recent focused run passed 22 tests.

The ten-epoch training pilot completed. Its periodic FID improved to 37.3384
at epoch 60, compared with the original best 37.8654 and the unmodified
epoch-60 continuation at 38.9212. This is the original uint8 diagnostic
against 10,000 held-out images, with 2,000 generated images.

Fresh-seed confirmation uses 10,000 generated images per case, seed 173000,
and the published RQ reference with an FP32 decoder and continuous pixels.

| Weights | Atom temperature | FID10k |
| --- | ---: | ---: |
| original | 1.0 | 30.3149 |
| original | 0.6 | 26.4843 |
| pilot-last | 1.0 | 29.8038 |
| pilot-last | 0.6 | 25.2518 |

The combined pilot weights and temperature 0.6 completed 50,000-sample
confirmation: **FID 24.6127001**, down from **29.4796168** (4.8669 points,
16.51%). Both results use RQ-Transformer's released `ffhq_256_train.npz`,
representing its complete 60,000-image FFHQ training set. The reference SHA256
is `7f8f54ad5eee50c5b1fc1583e9d65c5a46652c3d60ba031965c9f46c9f2fa12b`.
The 2,000- and 10,000-image scores above are screening and confirmation checks;
24.6127001 is the final unconditional FID50k result. Both 50k evaluations use
seed 73000, three ranks, batch 64 per rank, the same Inception implementation,
FP32 decoding, and continuous pixels without uint8 rounding. Our training
images follow the contiguous FFHQ split, while RQ uses a shuffled split, so
this remains a comparison using the same evaluation reference rather than an
exact training-data reproduction. RQ-Transformer's reported 10.38 remains
substantially better.
The subsequent transform audit additionally found LANCZOS downsampling in our
training-data builder versus BILINEAR in RQ's published real-image evaluation
transform. Feature-extractor preprocessing matches. The completed follow-up
measures its effect directly and establishes an exact training-image reference;
see `ffhq-fid-consistency-2026-09-22.md`.

The 10k ablation attributes most of the observed gain to cooling atom sampling:
30.3149 to 26.4843 with unchanged weights. The reweighted training adds a further
1.2325-point gain at the selected temperature. This supports both interventions
but does not establish a unique root cause. Real prefixes helping generation
does not prove that parallel prediction is defective: original VAR also
predicts positions within each scale in parallel. Our atom/coefficient token
representation and five-scale schedule differ from VAR's VQ tokens and
ten-scale schedule; comparisons with VAR's class-conditional ImageNet results
cannot explain the FFHQ gap quantitatively.

The selected epoch-60 checkpoint SHA256 is
`9003ecd29cee999bed00276a2be355be2bdb6c2f4668fa8e17af56b98d8bbbcf`.
The original epoch-50 weights remain preserved. The training resume contract
now includes any explicit sampling policy, so best-FID scores cannot silently
be compared across different sampling settings on resume.

The archived epoch-60 inference profile is
`promoted/inference-config.yaml` in this investigation directory, with atom temperature
0.6, coefficient temperature 1.0, atom top-k 250, both top-p values 1.0, and
CFG 0.0. It selects the frozen epoch-47 tokenizer and epoch-60 prior. Local
last/best-FID hardlinks, the 8-by-8 grid, result records, and inference profile
are in the investigation's `promoted/` directory. The separate training recipe
and objective-change receipt remain in `scale-weighted/`; the inference
profile is not an unchanged training-resume recipe.

Reproduce the selected 50k evaluation from this repository with the saved
runtime and evaluation plan (the W&B API key must already be in the environment):

```bash
work=/workspace/Projects/laser/outputs/ffhq256-var341-scratch-20260921/investigations/stage2-20260922
CUDA_VISIBLE_DEVICES=0,1,2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NCCL_NVLS_ENABLE=0 \
TORCH_HOME=/workspace/tmp/official-rqvae-eval-cache \
/tmp/laser-resume-env/bin/python -m torch.distributed.run --standalone --nproc_per_node=3 \
  "$work/evaluation-runtime/scripts/tools/investigate_ffhq_var_sampling.py" \
  --config "$work/promoted/inference-config.yaml" \
  --checkpoint "$work/promoted/prior-best-fid.pt" \
  --output "$work/reproduce-final-50k" --plan "$work/final-plan.json" \
  --reference /tmp/laser-rqvae-reference/ffhq_256_train.npz \
  --samples 50000 --seed 73000 --run-id ffhq256-var341-fixed-reproduction --online
```

The pilot's full last and best-FID checkpoints are both epoch 60 and have the
same bytes. Their committed online source artifact is
`helloimlixin-rutgers/laser/ffhq256-var341-stage2-scale-weighted-20260922-checkpoints:v1`.
The selected inference bundle references those immutable online blobs and the
frozen tokenizer, and includes the sampling policy and evaluation evidence.
Bundle `helloimlixin-rutgers/laser/ffhq256-var341-stage2-fixed-20260922-checkpoints:v0`
is committed online with aliases `latest`, `last`, and `best-fid`. Verification
checked local SHA256/MD5 hashes, source-file sizes, and the bundle's immutable
reference targets; the receipt is `promoted/online-checkpoints-verified.json`.

[Final 50k evaluation](https://wandb.ai/helloimlixin-rutgers/laser/runs/ffhq256-var341-fix-final-50k-20260922),
[training pilot](https://wandb.ai/helloimlixin-rutgers/laser/runs/ffhq256-var341-stage2-scale-weighted-20260922),
[selected inference bundle](https://wandb.ai/helloimlixin-rutgers/laser/runs/ffhq256-var341-stage2-fixed-20260922).
The reference protocol and published comparison come from the
[official RQ-VAE repository](https://github.com/kakaobrain/rq-vae-transformer);
the VAR implementation is available in the
[official VAR repository](https://github.com/FoundationVision/VAR).
