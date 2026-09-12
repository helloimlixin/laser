# Return to the successful Church checkpoint before loop refinement

The user explicitly requested stepping back to the successful version and applying fixes there. The earlier scratch loop comparison is paused, with optimizer and data-stream checkpoints preserved at steps 2,263 (looped) and 2,201 (unrolled). Receipt: `outputs/church-epoch50-loop-20260911/previous-scratch-pause.json`.

The quality reference is the original Church epoch-50 compound pair model: logged FID-50k 13.620649, independently evaluated at 13.724581 and 13.704673. Atom-only guidance produced 13.111186; this continuation uses unguided sampling. The later FFHQ-recipe Church reruns are weaker baselines and are not the reference for success.

## What is restored

Every original pretrained tensor is loaded strictly from `outputs/lsun-church-bar-20260910/assets/epoch50-source.pt` (SHA-256 `cbb7d0a8c22c31e042ed04004201b360b9bacd1238f6fa677d28ba3e0b21ce4a`). The source records epoch 50, global step 3,050, and its original complete configuration. It is a model-only checkpoint, so these are new optimizer continuations, not exact optimizer resumes.

- Original 404,738,048-parameter architecture: width 1,024, 24 spatial blocks, four depth blocks, 16 heads, two coefficient micro-transformer blocks, depth-specific coefficient classifiers, full atom–coefficient pair history in both streams.
- Original frozen Church tokenizer (one-epoch adaptation from ImageNet rFID 4.2109), dictionary, signed 2,048-bin quantizer, normalized range ±20, and depth scales `[1.109375, 0.6234375238418579, 0.3685546815395355, 0.24746093153953552]`.
- Existing center-crop token cache in the original coordinates. Its train and former held-out partitions are rejoined to restore the full 126,227-image training population. The original checkpoint already saw both partitions. The former holdout is labeled `train_probe`; 300 official validation images are also reported, with their prior experimental reuse disclosed.
- Hard coefficient targets, equal atom/coefficient loss weights, no geometry loss; original dropout .1, AdamW weight decay 1e-4, betas (.9, .95), effective batch 2,048, gradient norm clipping 1. There is no added pixel augmentation or token vocabulary change.
- Original unguided sampler: atom top-k 250, untruncated coefficient sampling, temperatures 1, generation batches 128, decoder batches 32, original RQ-VAE Inception and Church reference statistics.

## The one architectural experiment

The control continues the unchanged original model. The looped arm runs the original four depth blocks once and adds two passes through those same four blocks:

`h <- h + tanh(gate_i) * (F(h) - h)`

The two learned scalar gates start at zero. Therefore every pretrained weight and initial prediction is retained. There are 404,738,050 parameters in the looped arm; no pretrained blocks are overwritten, tied to different original blocks, or reinitialized. Each pass has separate causal KV state. The current support–coefficient pair is completed before it conditions subsequent predictions. Additional passes refine a current decision's hidden state; they do not revise sampled previous pairs.

This comparison adds computation, not parameters beyond the two gates. It is not a matched-compute architecture comparison: the control uses four depth-block applications and the active looped model uses twelve. Runtime is measured.

## Continuation and selection

Both arms use a conservative fresh-optimizer learning rate: eight-update warmup to 1e-5, then cosine decay to 1e-6 by update 128. Gate parameters use 100 times that rate without weight decay. The inherited original 5e-4/300-epoch schedule is recorded but is not restarted for this adaptation. This rate change is common to the control and experiment.

Evaluate step zero before any training, then every 32 updates. FID-4096 uses seed 12701. Keep the original step-zero checkpoint eligible for selection and stop after two screens without at least .25 improvement, or at 128 updates. Teacher-forcing losses are diagnostics; they do not replace generation-based selection. The selected checkpoint receives FID-50k with seed 17701, matching the independent original baseline's seed and batching. A lower screening FID alone does not establish improvement over the ~13.7 FID-50k reference.

Precision follows the established source evaluator, including its nested autocast scopes: training/teacher forcing and the cached backbone use FP32 with TF32 enabled; sampled atom/coefficient readouts use the enclosing BF16 scope; the decoder uses FP32. Older records abbreviated this as BF16 AR. CUDA cumsum remains unchanged to preserve the source computation. Strict deterministic-algorithm certification is not claimed; actual checkpoint/resume equality is tested separately.

## Verification

The source-restoration check compares all pretrained tensors and initial logits, checks all 64 sites on two images with nonzero loop gates, and compares generation with the established source sampler. The first 128 generated images' supports and coefficient IDs match the archived FID-50k source codes bitwise at seed 17701. This verifies restoration without claiming a newly computed FID-50k.

Focused tests cover zero-gate preservation, trainable gate gradients, preserved original gradients when gates are closed, causal coefficient use, separate loop caches, sampling equivalence, and the continuation schedule. Actual-model checks cover batch 2,048/microbatch 64, uninterrupted and resumed execution on different GPUs, and stopping/selection of the original checkpoint when continuation fails to improve. Test-only populations and small-sample FIDs cannot launch as production W&B runs.

Exact outcomes and source hashes are in `outputs/church-epoch50-loop-20260911/verification.json`. Runtime status, model selection, source provenance, and samples are logged under `outputs/church-epoch50-loop-20260911/{looped,control}` and the matching W&B group. The original model and all paused pilots remain preserved.
