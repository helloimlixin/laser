# Archived FFHQ compound code with Church settings

The user identified that the preceding `church-original-scratch-*` runs restored the later Church hard-target recipe, not the requested successful FFHQ compound recipe. Those jobs are saved and paused. Their pause receipt is `outputs/church-ffhq-archived-20260911/previous-wrong-recipe-pause.json`. They are not controls for the new experiment.

The actual source uploaded by the successful FFHQ run `ffhqcmp0804205803` is copied byte-for-byte to `src/ffhq_v4_archived.py`, SHA-256 `9ba1b49b4e5e339f0076bebee6fbac5629f6c391601de467019a3723c9d3e33f`. Its recorded FFHQ FID-50k was 8.1743927 at epoch 200. That is a different dataset and is not a Church target score.

The user supplied [the official Church stage-2 configuration](https://github.com/kakaobrain/rq-vae-transformer/blob/main/configs/lsun-church/stage2/lsun-church256-sqgan-8x8x4-350M-simp.yaml). The downloaded bytes match the vendored upstream file, SHA-256 `56d164940f33a2a59005caf9db0a71ddf4aa19cca837aba5d8d57788e453627d`. `configs/church_ffhq_archived_upstream.yaml` preserves that file and supplies model dimensions directly.

| Setting | Mistaken Church scratch run | Archived FFHQ + official Church adaptation |
|---|---|---|
| Stage-2 implementation | Later maintained compound class | Actual archived FFHQ-v4 compound class |
| Effective batch | 2,048 | 256, from Church config |
| Microbatch | 64 | 32, accumulating eight microbatches per GPU |
| Maximum epochs | 100 | 300, from Church config |
| Coefficient targets | Deterministic hard labels | Archived normalized soft targets at T=0.5; stochastic contexts |
| Normalized coefficient range | ±20 | ±3, as in FFHQ |
| Atom loss weight | 1 | 1.5, as in FFHQ |
| Geometry loss | None | Archived distribution geometry, 0.05; delay 2/ramp 3 epochs |
| Coefficient sampling | Untruncated | Archived nucleus p=0.85 |
| Atom sampling | Top-k 250/p=1 | Top-k 250/p=1, matching both sources |
| Training support mask | Later distinct-support mask | Archived behavior: no training mask; distinct supports enforced during generation |
| Schedule | New warmup and rapid decay | Official Church cosine from 5e-4 to zero; no warmup |

The unmodified archived `CompoundLaserRQTransformer` and `compound_objective` execute directly. There is no dependency on the later compound class's flags or objective defaults. The frozen Church tokenizer loader supports four depths; its coefficient-target method is the archived FFHQ method. The maintained common RQ base differs from the pre-FFHQ July 31 commit only by a depth-embedding hook, whose default forwards to the same pair embedding for this archived class. The attention and configuration implementations match that commit. Exact source hashes are preserved with each run; a bitwise reconstruction of the historical hardware/environment is not claimed.

Both models have width 1,024, 24 spatial blocks, four depth blocks, 16 heads, two coefficient micro-transformer blocks, and one coefficient classifier per sparse depth. Church uses 16,384 dictionary atoms and four depths; FFHQ used 2,048 atoms and two depths. The control has 404,738,048 parameters. The looped arm adds two zero-initialized scalar gates and two shared passes through the four depth blocks, with a separate KV cache per application. Full completed support/coefficient pairs condition both future spatial sites and subsequent within-site events.

Both stage-2 models start randomly with the same common initial tensors and empty AdamW state. No pretrained stage-2 checkpoint is loaded. Only their own matching scratch checkpoint can be resumed. The tokenizer is the already completed one-epoch Church fine-tune of the ImageNet rFID-4.2109 checkpoint, frozen throughout.

Use all 126,227 training images from the continuous center-crop cache, rejoining its former 1,024-image holdout. The former holdout is explicitly a train probe. Per-depth scales are max absolute training coefficient / 3. The 300 official validation images are separate, but have been reused in previous experiments. Drop incomplete effective batches as the archived trainer did: 493 updates per epoch, 147,900 maximum updates. AdamW betas (0.9,0.95), weight decay 1e-4, dropout 0.1, and gradient norm clipping at 1 match the sources.

The original LR schedule is common to both arms so this comparison isolates looping. The user's earlier request for faster decay is deferred until the source baseline is established; the discarded aggressive schedule is not silently treated as original. Full training is not stopped by the earlier plateau rule.

The archived training call uses an outer BF16 scope and `model(..., amp=False)`, so the nested model computation is FP32/TF32. The archived sampler is called with `amp=True` and no added outer scope; decoding is FP32. Runtime precision is tested rather than inferred from the upstream `amp` label.

FID uses the established original RQ-VAE Inception implementation and official Church reference statistics. This replaces the FFHQ run's dataset-specific metric reference. FID-4096 screens occur at epoch one and every five epochs. FID-50k runs at epoch 10 and every 50 epochs. The best screening checkpoint from each new run receives an independent-seed FID-50k at completion. Sample counts and epochs are logged explicitly.

Recipe fidelity is not proof of generation quality. The previous oracle audit found that FFHQ's normalized coefficient temperature produces substantial physical perturbations with Church's coefficient scales. That known transfer limitation still applies; this experiment restores the requested baseline without silently substituting a different target distribution. The upstream RQ-VAE soft-code temperature acts on its code distances, not on LASER scalar coefficients, so its numerical temperature alone does not establish equivalent noise. See `outputs/church-ffhq-recipe-20260911/target-noise-probe/results.json`.

Validation covers byte-identical archived source, forbidden pretrained loading, official model dimensions, stochastic normalized targets, full-pair causality, cached/teacher predictions, zero-gate equivalence and gradients, actual full batches, active geometry gradients, actual generation/decoding/FID, and checkpoint/resume through evaluation and geometry activation. Test-only small populations cannot use W&B production IDs. Production launch waits for these checks and verifies source hashes before starting either job.

Entry points: `scripts/train_church_ffhq_archived.py` and `scripts/launch_church_ffhq_archived.py`. Run records and the verification manifest live in `outputs/church-ffhq-archived-20260911`.
