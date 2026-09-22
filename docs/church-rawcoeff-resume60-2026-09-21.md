# Raw-coefficient Church continuation from epoch 60

The same online run is resumed:
[church-laser-ft3best-stochastic-rawcoeff90-h200x5-20260921](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-laser-ft3best-stochastic-rawcoeff90-h200x5-20260921).

The original best checkpoint is epoch 60 / optimizer step 7860, with FID50k
10.6884632054. Later original evaluations regressed to 10.9360111213 at epoch
70 and 11.1957635089 at epoch 80. The previous process failed during a network
checkpoint write after epoch 87. These observations motivate a reduced LR;
they do not prove that LR was the cause of the FID regression.

This continuation restores the entire epoch-60 model, AdamW moments and step
counters, and all five per-rank RNG states. The original tokenizer, physical
coefficient cache, stochastic targets, batch 192 per GPU (960 global), loss,
and sampling settings remain the same. The original epoch-60 checkpoint and
previous run files are preserved.

Training continues through epoch 90 (30 additional epochs / 3930 updates).
The initial LR is 3e-5, down from the checkpoint's 1.25e-4. A new cosine
schedule decays toward 1e-6. Three consecutive FID evaluations without an
improvement exceeding 0.02 halve the component above the LR floor, with one
evaluation of cooldown. Every strictly better FID is eligible for the best
checkpoint; the 0.02 threshold only controls LR reductions. The complete
controller state is serialized into every full checkpoint, and resumes
restore it without restarting patience or the cosine schedule.

FID uses exactly 50,000 samples at **every epoch, 61 through 90**, with the
original real-reference statistics, sampler, five GPU streams, and seed
20261921 + rank. The evaluator restores training RNG state. Rank zero's FID
is broadcast so all ranks make the same LR decision. The original online
history is retained; continuation-specific FID is also logged against
`continuation/epoch` so the epoch-60 rewind is explicit.

The frozen training runtime is reused without modification. The wrapper and
controller live in `outputs/church-stochastic-rawcoeff90-resume60-20260921/`.
A detached supervisor runs twelve isolated preflight updates across all five
GPUs before launching production from the untouched epoch-60 source.

Working full latest/best checkpoints live under
`/tmp/laser-rawcoeff-resume60/`, linked from the experiment directory. This is
an explicit exception to the frozen trainer's workspace-only storage rule:
the preceding network write failed, while local disk has ample space. W&B
provides durable full checkpoint artifacts, with checksums and sizes verified
against the committed remote manifest. Local working files are ephemeral;
the original epoch-60 checkpoint remains under `/workspace`. Uploads use a
bounded queue of immutable local copies, retain the latest pending state,
and drain the final upload before marking the run complete.

Receipts in the experiment directory:

- `source-verification.json`: local epoch-60 state matched to committed W&B artifact.
- `validation.json`: scheduler and uploader test results.
- `preflight/complete.json`: isolated real optimizer updates and full saved state.
- `status.json`: detached supervisor status.
- `train/evaluations.jsonl`: each epoch's FID and LR decision.
- `train/adaptive-lr.json`: current LR controller state.
- `train/checkpoint-upload.json`: most recent verified full last/best artifact.
- `complete.json`: written only after epoch 90 and the final verified upload.

Launch verification passed: the epoch-60 source matches the committed online
checkpoint's size, MD5, and full state. Five scheduler/uploader tests passed.
The isolated preflight completed twelve updates and three sample previews,
then saved step 7872 with optimizer and scheduler counts verified. Production
started again at step 7860; its first logged loss and LR exactly match the
preflight at step 7870. Steady throughput is approximately 3015 images/second.
The same W&B run is verified running, with `fid_every=1` online.

The first completed continuation epoch, 61, scored FID50k **11.06934397**.
The best remains the epoch-60 source, **10.68846321**. A full recovery
checkpoint at step 8000 contains the new scheduler state, the first FID
observation, AdamW step 8000, and all five rank RNG states. Subsequent FID
observations and verified checkpoint uploads are recorded in the live
receipts listed above.

The epoch-61 full latest state and epoch-60 full best state were verified
committed online in `selected-checkpoints:v13`, including matching file MD5
and byte counts. Epochs 62 and 63 scored 11.12238102 and 11.14139533. After
those three non-improving evaluations, the controller reduced LR from
2.92903195e-5 to 1.51451597e-5; epoch 64 scored 11.04010022. The original
best remains preserved. The accompanying read-only sparse-code investigation
is documented in [the audit](church-sparse-code-audit-2026-09-21.md).
