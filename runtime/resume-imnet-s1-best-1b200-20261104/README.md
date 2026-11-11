# ImageNet joint-token stage-one continuation

Source: `helloimlixin-rutgers/laser/imnet-s1-laser-s4-k2-j4096-680-scratch10-2h100-20261102`.
The complete 50k-image evaluation history selects epoch 4, step 40036, rFID
2.9342634986184635. Epoch 5 scored 3.1663600053379355. The recovered full-state
artifact is the source run's `-tokenizer:v47`, SHA256
`8d113f210de88a2f3a4920f5c76496200bdc9bd7fbaad67ae4c5f3df69a9eeb6`.

This deliberately continues the named four-scale, 4096-joint-symbol model,
including its 1024 directions and four coefficient levels. The source run's
summary records that it was superseded by a separate continuous-OMP experiment;
this continuation does not switch architectures.

The generator, discriminator, both optimizers and rank-zero RNG were restored.
Optimizer ages were verified as 40036 and 30036. Topology changed from four H100s
to the available single B200. The next three global image batches were verified
identical, but changing topology is not bitwise trajectory replay.

Both private five-update execution trials passed checkpoint, model and RNG
round trips, ending at optimizer ages 40041 and 30041. Batch 64 with accumulation
2 measured 87.58 images/sec and 122.59 GiB peak allocated memory; batch 32 with
accumulation 4 measured 85.05 images/sec and 62.54 GiB. The selected effective
batch remains 128. Benchmark updates are discarded before production.

Both learning rates use cosine decay from 5e-5 at step 40036 to 1e-5 at the
original epoch-10 endpoint, step 100090. All optimizer moments and hyperparameters
other than the learning rates are retained. The original objective, architecture,
image transforms, LPIPS weights and FID backend are unchanged.

Training data comes from the existing verified archive under
`/workspace/Projects/data/imagenet/archives/`, staged on local disk. Its MD5 is
`1d675b47d978889d74fa0da5fadfb00e`. The shared validation archive contains zero
sections and fails MD5. A separate local copy was restored from
`satabios/imagenet_val/ILSVRC2012_img_val.tar` and verified against the official
MD5 `29b22e2961454d5413ddabcf34fc5622`. Shared data was left intact.

The original full 50k-image baseline is recomputed before training, with a
0.03-rFID tolerance for hardware/batching differences. Full evaluation follows
at step 40536, every 2500 continuation updates, and every epoch endpoint.
Checkpoints are published to W&B every 1000 updates and before evaluation.
Each new best is preserved separately as `tokenizer-best-rfid.pt`, with the
associated remote artifact given the `best-rfid` alias.

Live output:
`/workspace/Projects/laser/outputs/imnet-s1-laser-best2934-continue10-1b200-20261104/`.

Local runtime and detailed log:
`/tmp/laser-s1-best-b200/frozen/runtime/` and
`/tmp/laser-s1-best-b200/training.log`.

`supervisor-status.json`, `status.json`, `metrics.jsonl`, `baseline-verification.json`,
`checkpoint.json`, `best-reconstruction.json`, and per-step evaluation records
are mirrored to the persistent output directory. Full latest and best checkpoints
are also mirrored. W&B is a new continuation run, preserving the source history.

To request a checkpointed stop, create `/tmp/laser-s1-best-b200/run-output/STOP`.
Training requires this host to remain running. Inspect its latest checkpoint
before restarting; the supervisor refuses to overwrite an existing production
launch.

The source artifact and frozen-runtime checksum verification provide the original
training implementation. `prepare.py` applies the small LR, logging, evaluation
frequency and checkpoint-preservation changes. `continuation.py` supplies those
hooks; `supervise.py` selects the verified execution and launches the detached job.
Credentials are stored only in the root user's private credential file, not in
the runtime, configuration, output artifacts, or this directory.
