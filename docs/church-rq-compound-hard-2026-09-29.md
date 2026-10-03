# Deterministic hard-target compound baseline

This experiment removes stochastic encoding and physical-vector soft targets from the paired-head Church model. It changes the training targets while retaining the model, initial seed, optimizer, schedule, and evaluation protocol. It starts stage 2 and AdamW from scratch.

At each of four depths, choose the atom and coefficient bin jointly to minimize the residual squared error, subtract their exact physical contribution, and continue. For each atom, evaluate its nearest existing FP32 bin to the optimal scalar coefficient, then compare the quantized pair error across atoms. The dictionary and shared 2,048 physical bins are frozen; repeated atoms are allowed, and previous coefficients are never refitted. Cache the resulting [N,8,8,4,2] integer pairs once. The existing OMP cache is not reused because it has different residual and support rules.

Train with the sum of hard atom cross-entropy and hard coefficient cross-entropy conditioned on the observed current atom. No soft probabilities or stochastic teacher are constructed during training. The legacy teacher RNG field is retained as an inert checkpoint compatibility field; it is not used for target generation. The model's ordinary dropout remains unchanged.

The 392,392,704-parameter architecture is identical to the preceding compound run: 20 plane-encoder blocks, six causal decoder blocks, individual earlier-plane cross-attention memory, and a direct same-site physical prefix. Keeping the prefix is deliberate experimental control, not another architecture change. Initialization, data order, batch size, AdamW parameters, and sampling seeds match the preceding run.

Train for 300 epochs (18,600 updates) on eight GPUs at global batch 2,048, microbatch 128 and accumulation two. Preview 64 images every 200 updates. Evaluate 50,000-image FID every 620 updates. Sampling remains stochastic over all 16,384 atoms followed by all 2,048 conditional coefficient bins at temperature one. Publish full LAST at step 200, then LAST/BEST at FID and final boundaries; retain durable copies and encoding provenance.

Validation covers a small dense joint-argmin reference, exact residual subtraction and deterministic caching, hard likelihood equivalence and gradients, production-size distributed training, unchanged model causality/cache checks, a fixed-subset learning check, generation capacity, and checkpoint continuation. Diagnostic weights are discarded before production. Numeric continuation is assessed with the same predeclared tolerances as the preceding run, without claiming bitwise equality.

The preceding soft-target run is preserved at a completed checkpoint with its measured BEST and published recovery assets before its identified workers are retired. The original request to wait for epoch 30 was fulfilled earlier in the experiment lineage.

Runtime: `/tmp/laser-church-rq-compound-hard-20260929`.

W&B: https://wandb.ai/helloimlixin-rutgers/laser/runs/church-rq-compound-hard-20260929

## Validation results

All 34 CPU tests passed. The deterministic cache contains 126,227 images, shape [126227,8,8,4,2], int32, SHA256 `61427c0fecffd9c5bf3b7e5252427c5c23d0d4cae91e42682b5a213c349a3d53`. Construction took about 20.9 seconds across eight GPUs. An independent FP64 reference matched all 64 sampled pair decisions from 16 sites across the dataset, with zero objective regret. All cache, source-latent, dictionary, and physical-bin identities were verified.

All eight ranks passed training at the production batch size, with 503 finite nonzero gradient tensors and identical model/Adam states. Steady smoke updates took 0.660 seconds, with peak allocated memory 49.66 GiB. The unchanged fixed-64-image, 128-update learning probe reduced hard joint NLL from 17.7513 to 0.03823. Every depth learned, including the first plane without cross-memory; shuffling earlier-plane context worsened later-plane NLL. These diagnostic subset results do not establish generalization or FID improvement.

Sampling capacity passed at 1024 images per GPU (3182 images/second aggregate generation only). Continuation from a three-update checkpoint versus four uninterrupted updates passed the existing numerical gate: maximum FP32 model difference 5.96e-8 and Adam-moment difference 2.33e-10. Loaded states, metadata, cursor, sources, and rank RNG streams matched exactly; continuation is not bitwise identical.

The preceding stochastic-target run retired at step 1000, preserving full LAST SHA256 `382f08d97dc9adc439fe6f526cd4f62208b601e103295b9dd73a053cb0c6160c` and BEST at step 620 (FID 77.07835). There was no scheduled FID at step 1000. W&B selected-checkpoints:v2 and the hard-target retirement lineage preserve this transition.

## Production launch

Launched fresh on all eight GPUs on 2026-09-29 at 18:57 UTC. Controller PID 310069, initial torchrun PID 310079. Source artifact `church-rq-compound-hard-20260929-source:v0` is verified online. Initial live updates are approximately 0.65 seconds at global batch 2048; all 503 parameter tensors receive gradients. The controller continues toward 300 epochs with bounded restart from this run's own full checkpoints.

Independent production audit passed: all eight ranks began with empty Adam state and initial weights identical to the preceding run. The first step-200 image grid is confirmed in remote W&B history. Full LAST200 (4,709,417,302 bytes, SHA256 `4aaa45d26a811f17701dabb1e2f852ca4fac858c3e4341f4548f6ab10a7f9162`) is verified in selected-checkpoints:v0 and durable storage. It includes 503 optimizer states, scheduler/cursor, and all rank RNG streams. Recovery:v0 contains the deterministic pair cache, encoding metadata, and frozen source assets.

Median steady training time over 23 logged full updates was 0.65065 seconds; partial final-epoch batches were excluded. This is about 37% less update time than the prior soft-target smoke result of 1.036 seconds. First FID remains scheduled at step 620; no image-quality improvement is claimed from the training or subset diagnostics.
