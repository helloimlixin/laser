# Longer-context hard-6kbps continuation

The user requests substantially more training to reduce artificial-sounding
audio. Continue both completed K4/4096-atom LASER and four-book RVQ codecs from
their 200000-update final states to **600000 total updates**. Preserve both
optimizers, learned coefficient centers, dictionaries and the three best existing
validation checkpoints. The finished 200k comparison remains intact.

Both continuation arms use 31960-sample crops (666 ms), batch 12, in place of
7960 samples (166 ms), batch 48. This changes audio samples/update from 382080
to 383520 (+0.38%), while quadrupling temporal context. A 16-byte packet header
uses about 13% of the short crop's budget but about 3% of the longer crop's budget.
LASER trains with 67 coded frames per longer crop (about 100.6 Hz) rather than
15 (90.5 Hz); full validation averages about 104.3 Hz. RVQ moves from about
126.6 to 144.2 Hz, closer to full validation's 148.8 Hz. This reduces the observed
training/inference frame-rate mismatch. It does not establish that short crops
were the only cause of audible artifacts.

The packet format, coefficient vocabulary, K4, 4096 LASER vectors, RVQ 4x1024,
backbones and losses are unchanged. Both use the same continuous cosine LR from
their saved 0.000158253834 to 0.00002 over 400000 additional generator updates.
The data order remains deterministic. The supervisor compares both arms' epoch
audits and stops the pair on a discrepancy or incomplete/failed worker.

Each extension has a **24 assigned-GPU-hour ceiling**, 48 GPU-hours for the pair;
parent compute is recorded separately. Graceful stopping preserves a restartable
checkpoint. The update target is not guaranteed if the compute ceiling is hit.
Latest and the best three validation-ViSQOL checkpoints, audio previews, waveforms,
log-mel and spectral figures upload every five completed epochs. With batch 12,
each continuation epoch contains 3411 batches. Every full validation still
decodes all 128 fixed recordings from packets and checks their maximum bitrate.

The longer clips form a new training curriculum. This is an equally trained
paired continuation, not an ablation of update count alone. Additional training
may improve naturalness; listening quality is not guaranteed by ViSQOL or update
count. The completed 200k test results do not influence checkpoint selection or
coefficient fitting. The old K2 stage-2 queue stays held for compatible caches.

```bash
python scripts/tools/continue_mdctcodec_hard6k.py --prepare
CUDA_VISIBLE_DEVICES=0 python scripts/tools/continue_mdctcodec_hard6k.py --arm laser --smoke
CUDA_VISIBLE_DEVICES=1 python scripts/tools/continue_mdctcodec_hard6k.py --arm rvq --smoke
python scripts/tools/run_mdctcodec_hard6k_pair.py --root outputs/mdctcodec_k4_a4096_hard6k_long_20260913 --evaluate
```

The supervisor requires recorded preflight/restore checks before launching.
After both update budgets complete, it evaluates the validation-selected pair on
the existing fixed test set and uploads the report, within the remaining compute
ceiling. Partial or failed training does not trigger this evaluation.
Results live in `outputs/mdctcodec_k4_a4096_hard6k_long_20260913`.
