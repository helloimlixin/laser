# Cached plane memory for compound-token RQ

The user requested the new model start only after the preceding global-depth model completes 30 epochs, including its scheduled FID evaluation. The boundary is step 1,860. The detached watcher requires a complete full checkpoint, a 50,000-image result and all eight evaluation receipts before preserving LAST and every selected BEST, then terminating the identified parent processes.

Parent run: https://wandb.ai/helloimlixin-rutgers/laser/runs/church-rq-globaldepth-20260929

New run: https://wandb.ai/helloimlixin-rutgers/laser/runs/church-rq-planememory-20260929

Runtime: `/tmp/laser-church-rq-planememory-20260929`.

Durable source and receipts: `/workspace/Projects/laser/outputs/church-rq-planememory-20260929`.

## Dependency graph and implementation

For raster location `i` and sparsity level `d`, predict one complete atom/coefficient token from every individual token at earlier levels and every previous location at the current level:

`p(Q) = product_d product_i p(q[i,d] | q[:,<d], q[:i,d])`.

The model uses a 20-block bidirectional encoder and a six-block causal self/cross-attention decoder, both width 1,024 with 16 heads. Parameter count is 420,218,881 across 488 tensors, compared with 420,205,569 for the preceding flat model. The shared output head remains a single 65,537-way categorical classifier.

Each completed 64-token plane is encoded independently once. Its 64 individual physical pair slots retain spatial/depth positions and a direct pair-embedding residual; they are appended to immutable cross-attention memory. No current or future plane enters this memory. Current-plane decoder self-attention uses shifted tokens with a causal mask. At a plane boundary, its self-attention cache resets while previous cross-attention keys and values stay intact. Sampling uses preallocated BF16 caches and single-query SDPA. Three completed planes are encoded to generate all 256 target tokens.

The first plane has learned empty-context memory and relies on the six-block decoder. This is a capacity tradeoff that requires explicit testing. Moving the heavy encoder outside the token-by-token loop is a speed hypothesis, not a demonstrated FID improvement or a solution to exposure bias.

## Preserved training and sampling recipe

Stage 2 and AdamW initialize from scratch. The stage-1 tokenizer, encoder-latent cache, dictionary, physical coefficient levels, and joint teacher are unchanged. No refitting, reward, contrastive term or extra geometry loss is introduced.

Each atom has four existing physical coefficient levels shared across sparsity levels. The nonzero joint ID is `1 + 4 * atom_id + coefficient_level_id`; zero is a separate token. The teacher samples complete pairs over the full vocabulary at temperature 0.125 using squared physical residual distance, subtracts the exact sampled vector before the next level, and supplies full joint soft labels. The objective remains joint soft cross entropy. Model and teacher RNG streams are isolated on each rank and saved separately.

Eight H100 GPUs train with global batch 2,048 for 300 epochs (18,600 updates). AdamW uses learning rate 0.0005, betas `(0.9,0.95)`, epsilon `1e-8`, weight decay `0.0001`, clipping at 1, zero warmup and cosine decay. Training microbatch is chosen by an eight-GPU comparison of 64 and 128; 128 is selected only if throughput improves at least 5% and peak allocation is at most 70 GiB. Accumulation preserves global batch 2,048.

Sampling keeps the previous temperature-1 policy: union of the top 1,400 complete pairs and a nucleus retaining at least 95% of full-vocabulary probability mass. There are 64 preview images every 200 updates, and official continuous-pixel 50,000-image FID every 620 updates and at completion. Generation batch 1,024 per GPU requires a successful capacity probe; decoder and Inception batches are 32 in FP32. The official reference and eight logical evaluation streams are unchanged.

Full local checkpoints are saved every 200 updates and at FID boundaries. The first full LAST is uploaded at step 200; subsequent FID boundaries publish full LAST and measured BEST, with durable copies. Source and all five recovery assets are uploaded and verified. Resume accepts only this run's source, plan, data, optimizer/schedule, distributed layout, cursor, and all 16 RNG streams.

## Gates and handoff

CPU validation passed ten architecture tests, three recovery tests, the sampler/protocol checks and eight handoff/retirement tests. All 256 cached CPU predictions agree with dense forward predictions to maximum absolute error `5.96e-8`. Tests cover independent-plane encoding, absence of current/future leakage, influence from individual earlier pairs, distinction between different pairs with the same physical sum, cache appending without rewriting earlier slots, and all parameter gradients. An independent review found no outstanding issues after tightening resume-report validation against missing sections and stale outputs.

GPU validation runs only after the epoch-30 handoff. It checks every rank's full-size model gradients and optimizer state, FP32/BF16 dense/cache parity, a fixed-64-image learnability probe, actual generation capacity, and three-plus-one versus four-step continuation. The learnability probe separately requires meaningful first-plane KL improvement and worse later-level cross entropy when earlier-level memories are shuffled while current-level history stays intact. These checks assess a small fixed training subset, not generalization or FID.

Exact resume comparison is preserved even when it fails. A separate numerical check permits only finite FP32 parameter differences up to `3e-7` and Adam moment differences up to `1e-9`; loaded tensors, all RNG streams, metadata, schedule and cursor must match exactly. Production layout is bound to the measured smoke-test layout before launch, with the original source and plan archived in `prelaunch64`.

## Completed validation and launch

The parent completed epoch 30 with FID **19.23521253**, comprising mean distance **8.34630299** and covariance distance **10.88890954**. Earlier FIDs were 100.3636 and 33.5002 at epochs 10 and 20. Its full LAST and BEST both correspond to step 1,860 and SHA-256 `42f113a9629d32e32b5e2aba5a2e113f351fea60f12d6ab4fcc528731eb81e0f`. Both are verified in W&B artifact `church-rq-globaldepth-20260929-selected-checkpoints:v3` and durable storage; retirement lineage is `church-rq-globaldepth-20260929-planememory-retirement:v0`. The old run is marked retired after 30 epochs, not completed through its original 300-epoch target.

All eight GPUs passed production-size forward/cache parity, no-leakage, and gradient/state checks. Worst FP32 cache error was `6.20e-6`; worst BF16 KL difference was `3.66e-6`. All 488 parameters had gradients and optimizer states, and model/Adam hashes matched across ranks.

| Training microbatch per GPU | Accumulation | Slowest-rank steady update | Peak allocated GiB |
| --- | ---: | ---: | ---: |
| 64 | 4 | 1.46619 s | 35.6995 |
| 128 | 2 | 1.35945 s | 62.4872 |

The selected production layout is **128 × 8 GPUs × 2 accumulation = 2,048**, giving a measured 7.85% throughput improvement over microbatch 64 for this model.

At generation batch 1,024 per GPU, all eight GPUs generated 8,192 sparse-code grids in 6.0641 seconds: **1,350.90 images/second**, about 52% more than the preceding flat model's 889.02 images/second at the same batch and coverage policy. This is a generation-only benchmark, not full FID turnaround. Peak allocated memory was 10.8589 GiB, including 6.0234 GiB of persistent cache. All 256 events and exactly three plane encodings were verified. Each rank also decoded 32 images through the FP32 tokenizer decoder and produced finite outputs.

The fixed-64-image learning probe reduced mean KL from 10.68968 to 0.24092. First-plane KL fell from 11.22374 to 0.14401. Shuffling lower-plane memories while preserving current-plane histories worsened later-level cross entropy by 0.36319, 0.30844, and 0.24729 nats. All three learning gates passed. This remains a memorization/conditioning check and does not establish held-out or FID improvement; the shuffle includes same-site earlier pairs.

Resume loaded the selected step-3 model and Adam tensors exactly. Three-plus-one versus four uninterrupted updates preserved all 16 RNG streams and all nonfloating state exactly, but was not bitwise identical in subsequent weights: maximum model difference `3.72529e-9`, maximum Adam moment difference `5.82077e-11`. The raw bitwise report is retained and the separate numerical gate passed.

Production launched fresh on all eight GPUs after these checks, with controller PID 284127 and torchrun PID 284193. Only the measured microbatch/accumulation and descriptive status/batch notes changed from the archived pilot plan; the model, teacher, loss and sampler source stayed identical. `diagnostics/final-configuration-gate.json` binds the final plan and source to the tested layout. Runtime receipts and W&B provide ongoing progress and publication status.


## Production publication and training-speed comparison

The step-200 64-image Church grid is verified on W&B by downloading the remote image and matching its SHA-256. The full step-200 LAST (5,043,315,070 bytes, SHA-256 `a6f1e17b188cafe3f1b09bfa6a42df9c57ebee6e7fa2bd51aa10643a5b57be79`) is committed in `church-rq-planememory-20260929-selected-checkpoints:v0`; its manifest matches the independently checked local payload. All 488 Adam states, scheduler 200, and eight model plus eight teacher RNG streams passed recovery checks. Source and all recovery assets are also verified online.

Full-update production medians on eight GPUs at global batch 2,048 are 0.68692 s for DC-shared, 1.36929 s for preceding flat joint RQ, and 1.35534 s for this model. Initial ten updates and partial epoch-end batches are excluded. DC and this run both use microbatch 128 with accumulation 2; flat joint RQ used 64 with accumulation 4. Thus this run is about 1.97 times slower per training update than DC. The previous 52 percent improvement measures generation throughput against flat joint RQ, not training speed against DC.

This comparison changes the objective as well as architecture. DC reads cached hard pairs and predicts 16,384 atom classes plus 2,048 conditional coefficient classes. The joint pipeline predicts 65,537 classes and regenerates full FP32 stochastic residual soft targets each visit. Its dense teacher target alone occupies 8.0001 GiB per GPU at microbatch 128; teacher and forward/backward soft CE each process 256 chunks per microbatch. These are concrete additional costs; no component-level timing attribution has yet been measured. Streaming/fusing the teacher and soft CE while preserving the full joint distribution is a sensible optimization target. The detailed measured comparison is retained in `diagnostics/training-speed-comparison.json`.
