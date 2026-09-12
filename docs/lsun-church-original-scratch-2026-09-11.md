# Original Church architecture, trained from scratch

The user clarified that “step back to the successful version” means reuse the successful architecture and recipe, then train stage 2 from random initialization. It does not mean initialize from the epoch-50 model. The mistaken checkpoint continuations are saved and paused; their receipt is `outputs/church-original-scratch-20260911/previous-continuation-pause.json`.

## Initialization

Both arms construct fresh stage-2 tensors with seed 0 and empty AdamW state, starting at optimizer step and epoch zero. The model factory has no checkpoint argument and is tested with both checkpoint reading and state loading forbidden. The trainer has no pretrained-source argument. Its optional resume path accepts only a checkpoint with the same scratch-run configuration and initialization audit.

`initialization.json` records `kind=random`, `stage2_checkpoint_loaded=null`, initial step zero, and a SHA-256 fingerprint of the original model's tensors. The two arms have identical initial common tensors. No historical stage-2 model is a selectable fallback. Historical ~13.7 FID-50k is only an evaluation reference.

The pretrained stage-1 tokenizer remains frozen, as required for this stage-2 experiment. It is the one-epoch Church fine-tune of the ImageNet rFID 4.2109 checkpoint, SHA-256 `93f19adc8ca2cfae0cf629d48d2d6f5f7039440b1a48ad1c7ba43cf17932a388`.

## Model and recipe

- Control: original 404,738,048-parameter compound RQ-Transformer, width 1,024, 24 spatial blocks, four depth blocks, 16 heads, two coefficient micro-transformer blocks, and depth-specific coefficient classifiers.
- Looped: the same randomly initialized model, with two additional shared passes through the four depth blocks. Updates are `h <- h + tanh(gate)*(F(h)-h)`, with two gates initialized to zero and learned from scratch. The 404,738,050 parameters include every original block and two extra scalars. Each effective layer has separate causal KV state.
- Both streams condition future tokens on completed support–coefficient pairs. Token construction, the dictionary, and the original 2,048 signed scalar coefficient bins are retained; normalized coefficient range is ±20 with the original four depth scales. There is no complete-site integer vocabulary.
- The full 126,227-image center-crop token cache is used. Full effective batches of 2,048 drop the epoch remainder, matching the original 61 optimizer updates per epoch. Microbatch is 64. The 1,024-image train probe is part of the training population; the 300 official validation images have been used in earlier experiments.
- Hard coefficient targets, equal atom/coefficient NLL weights, no geometry loss, dropout .1, AdamW betas (.9,.95), weight decay 1e-4, and gradient clipping 1. Gate learning rate equals the base learning rate; the continuation's 100x multiplier is removed.

## Schedule and evaluation

The maximum is 100 epochs / 6,100 updates. The common learning-rate change addresses the user's earlier request for faster decay: warm up over one epoch to the original peak 5e-4, decay exponentially to 5e-5 at epoch 10, then cosine decay to 1e-6 at epoch 100. This is a scratch-training schedule, not the mistaken 128-update/1e-5 continuation schedule.

FID-4096 is measured at epoch one and every five epochs, using atom top-k 250, untruncated coefficient sampling, temperatures 1, seed 12701, and the original Church statistics. There is no initial random-weight FID delay before training. Keep the best checkpoint from the new run. Six unimproved screens can stop a run only from epoch 50 onwards. The selected checkpoint gets FID-50k with seed 17701. FID counts and training age must be matched before claims against the historical 13.7 baseline.

The source evaluator's precision scopes are preserved: teacher forcing and cached backbone use FP32/TF32; sampled readouts use BF16; decoding uses FP32. The existing recurrent cache and causal tests are reused. New checks cover forbidden pretrained loading, initial fingerprints, actual full-size scratch updates, full-batch epoch behavior, and exact checkpoint/resume equality. Small-population runs are test-only and cannot use production W&B IDs.

Entry points: `src/church_original_scratch.py`, `scripts/train_church_original_scratch.py`, `scripts/launch_church_original_scratch.py`. Verification and launch status are recorded under `outputs/church-original-scratch-20260911`. Earlier checkpoints remain preserved for reference, with no weight transfer to this experiment.
