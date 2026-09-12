# Matched fresh MDCTCodec RVQ–LASER experiment

**Stopped and superseded on September 12.** The initialization-only LASER
coefficient bound did not follow the evolving encoder scale; training diagnostics
eventually showed 100% clipping. Both arms and the reporter were stopped. The
last saved LASER checkpoint contains 16,188 generator updates (19 epochs); RVQ
contains 17,040 (20 epochs). Latest plus the best three saved checkpoints were
preserved online in `model-4ahidv8n-selected-checkpoints:v3` and
`model-3lqdow5z-selected-checkpoints:v3`. See the
[range fix and restart](mdctcodec-range-fix-2026-09-12.md). The text below records
the original protocol, whose fixed-range choice is superseded.

This experiment tests the training-history confound behind the gap to the released
MDCTCodec checkpoint. Both arms start from identical freshly initialized encoder,
decoder, discriminator and identity adapters. Neither resumes a trained model.
The released checkpoint remains a separate evaluation reference.

Launched September 12 at 03:43 UTC on the existing two H200 GPUs:

- [LASER training](https://wandb.ai/helloimlixin-rutgers/laser/runs/4ahidv8n), GPU 0, PID 62200.
- [RVQ training](https://wandb.ai/helloimlixin-rutgers/laser/runs/3lqdow5z), GPU 1, PID 62201.
- Automatic final reporter PID 62202; it waits for both successful budget completions.

Both passed their first 100 production updates with finite losses and active
spectral objectives. Initial weights, protocol, source archive and environment
were uploaded to each run before optimization. This is a launch record; quality
results require completed training and the final locked-test report.

The frozen protocol and initialization hashes are in
`outputs/mdctcodec_matched_6kbps/protocol.json`. The encoder, decoder and
discriminator initial state hashes are checked before each run starts; seed alone
is not relied on because constructing different bottlenecks consumes different
amounts of random state.

| Setting | Both fresh arms |
|---|---|
| Rate / waveform | 6 kbps, mono 48 kHz, original levels |
| Training data | 40,936 mic2 recordings, 100 speakers |
| Budget | Exactly 200,000 generator and 200,000 discriminator updates |
| Batches | 48 crops of 7,960 samples, drop incomplete batch |
| Data order | Same per-epoch permutation and per-file crop offset |
| Optimizer | Fresh AdamW, LR 0.0002, betas 0.8/0.99, weight decay 0.01 |
| LR schedule | Multiply by 0.999 per completed epoch |
| Precision | bf16 mixed training; FP32 serialized validation/test |
| Shared audio losses | MDCT MSE 250; mel MAE+MSE 45; adversarial 0.1; feature matching 0.1 |
| Update order | Discriminator first, then generator using updated discriminator |
| Validation | 128 full recordings, 16 per held-out speaker, every epoch |
| Test | 200 separate full recordings, 25 per held-out speaker |
| Checkpoint selection | Highest validation ViSQOL audio at 48 kHz |
| Uploads | Latest plus top three, every five completed epochs and successful exit |

There are 852 full training batches per epoch. The budget therefore covers 234
complete epochs plus 632 updates in the last epoch. Lightning counts the two
optimizers separately; `max_steps=400000` and an independent generator counter
enforce the intended 200,000 updates per arm. These are all updates at 6 kbps.

RVQ uses the authors' unchanged quantizer source at commit
`1aff8b6287f4e2e66511bd4bc46f3a6bcf96edaf`: four 1,024-entry codebooks,
32-dimensional projected codes, no quantizer dropout. Its commitment coefficient
is 2.5 and codebook coefficient is 10, as in the public training code. The paper
and code differ on the written commitment coefficient; this experiment records
the implemented value explicitly.

LASER uses 8,192 atoms, two selected atoms per frame, and signed127 coefficients:
two times (13 index bits plus 7 coefficient bits) at 150 frames/second. It retains
the recovered alternating residual dictionary update (ridge 0.05, relaxation 0.1,
eight-batch accumulation). Commitment coefficient is 2.5. Dictionary loss is
optimized by its explicit update instead of an additional gradient loss.
Coefficient quantization is active from the first update. Its fixed bound,
0.9775147438049316, was calibrated at p99.9 on 1,024 crops from the training split
using the fresh shared encoder and fresh LASER dictionary. This bound belongs to
these fresh weights, not the previous trained checkpoint. Clipping is monitored;
no test-dependent recalibration is allowed.

The new test excludes all source recordings reserved by the earlier recovered
benchmarks and corrected comparison, including the sources of concatenated long
segments. The validation manifest extends the earlier test-disjoint 116-file
validation set to 128. All evaluation source files have recorded SHA-256 hashes.
The manifest SHA-256 is
`d085b127d5978ec0f2639de19dae3b2ad709f29d822e6ef642f5a7b04b1d7b31`.

Each epoch records a SHA-256 digest of every filename and crop offset. The final
report refuses to claim matched examples if the two complete data-order logs
differ. Crop selection is independent of model random draws, worker scheduling,
and GPU speed. CUDA reflection-padding backward is not guaranteed deterministic
by this PyTorch version; identical data and starting weights do not imply
bitwise-identical floating-point training or seed-robust conclusions.

Validation and test decode actual five-byte-per-frame payloads. Final reporting
compares both validation-selected fresh checkpoints with released MDCTCodec on
the locked test. Metrics are ViSQOL audio48k, ViSQOL speech16k, PESQ-WB16k and
STOI16k, with 5,000 paired speaker-bootstrap draws. Nominal 6 kbps and measured
packed payload rates are both retained; padding counts toward payload rate.
Model parameters, fixed bounds, original length and headers are outside that rate.

## Validation and operation

Fifteen focused tests pass. They verify equivalence of the RVQ adapter to upstream outputs, code IDs,
losses and gradients; decoded-code equivalence; LASER gradient and bitstream
behavior; data/crop reproducibility; and upload cadence. Both real GPU integration
runs exercise six one-batch epochs and actual online epoch-five uploads. The
first integration attempt exposed missing metadata in the new loader and is
explicitly marked invalid in W&B. The corrected loader checks that MDCT, mel and
feature-matching losses activate before any full experiment starts.

The online epoch-five artifacts are verified COMMITTED with latest and three
ranked checkpoints. Remote checkpoint restore tests preserve both optimizer
states, update counts and the expected next batch. Evidence is saved in
`preflight_verified.json` and `restore_verified.json` under the experiment root.

Entry points:

```bash
python scripts/train_mdctcodec_matched.py --prepare
CUDA_VISIBLE_DEVICES=0 python scripts/train_mdctcodec_matched.py --arm laser
CUDA_VISIBLE_DEVICES=1 python scripts/train_mdctcodec_matched.py --arm rvq
python scripts/report_mdctcodec_matched.py --pids LASER_PID RVQ_PID
```

Resume an arm using its existing output directory and `--resume /path/to/last.ckpt`.
The `--prepare` command reuses a completed preparation and does not overwrite it.
Reduced-budget options require `--smoke` and use separate output directories.
The final reporting process waits for both jobs and requires successful budget
completion. Logs and run URLs are in each arm's output directory and `launch.json`.

This is one matched-seed comparison of two complete bottleneck training methods.
Their parameter counts, bottleneck initialization, auxiliary objectives and
dictionary-learning algorithms differ. It is not a pure estimate of sparse
representation alone, a strict reproduction of every upstream training operation,
or a comprehensive SOTA claim. If both fresh arms trail the release, reproducing
the authors' complete training path remains the next diagnostic.
