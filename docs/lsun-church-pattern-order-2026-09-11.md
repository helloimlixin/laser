# Church: matched coefficient-first ordering pilot

**Status: paused and superseded at the user’s request.** The user dropped complete-site integers and selected a [compound RQ-Transformer loop comparison](lsun-church-looped-rq-2026-09-11.md). Both ordering pilots saved resumable checkpoints; see `outputs/church-interleaved-pattern-20260911/previous-ordering-pause.json`. The launch account below is historical, not current running status.

This experiment tests whether giving atom predictions the coefficient pattern first improves generation. Both arms use the existing 2,048-pattern codebook, the frozen Church tokenizer, and the same 67-bit complete-site integer. The decoder and reconstruction quality of the representation are unchanged.

The preceding support-first run generated distorted buildings. Its epoch-10 fixed-support diagnostic showed coefficient sign accuracy falling from 77.9% with real history to 68.1% with generated history, and its later atom predictions added little information over training frequencies. These observations motivate a new factorization; they do not establish that the new order will work.

## Matched models

Each arm has **198,978,304 parameters**, width 768, 20 spatial blocks, six depth blocks, and 12 attention heads. Both start from identical random parameter tensors with seed 9901. Both use five depth events, one per atom or joint coefficient pattern.

| Arm | Within-site decisions | Information available to atom prediction |
| --- | --- | --- |
| `pattern-first` | pattern, atom 1, atom 2, atom 3, atom 4 | All four chosen signed coefficients, plus the weighted sum of atoms already chosen |
| `support-first` | atom 1, atom 2, atom 3, atom 4, pattern | The unweighted sum of atoms already chosen |

The pattern-first predictor obtains its pattern from preceding spatial sites. It cannot see any current-site atom identity. The pattern is embedded from its four physical coefficients divided by the existing depth scales. Later atom context includes that embedding and the signed weighted partial sparse reconstruction. Both arms mask already selected atoms and use the decoded completed sparse latent as spatial input.

The same small pattern embedding MLP is present in both state dictionaries, but its 67,072 parameters are used only in the pattern-first depth path. Thus nominal parameter counts and initial tensors match; effective use of those parameters differs. The intended comparison changes ordering and the causal coefficient information available to prefix embeddings. It is not a demonstration that arbitrary token permutations alone improve generation.

Five-field transport tensors are internal to the transformer. The existing integer codec still serializes exactly four atoms and one pattern, and integer decoding alone recovers all four signed coefficient-bin IDs. No encoder, dictionary, coefficient table, or image decoder is trained.

## Training and evaluation

Both arms use **joint maximum likelihood**, `(sum of four atom NLLs + pattern NLL) / 5`. Logs include the unnormalized joint NLL and bits per site. Moving the pattern earlier transfers uncertainty between heads, so lower atom NLL alone is insufficient evidence of improvement. Held-out stopping monitors total joint NLL.

This differs from the older run's 60% mean-atom / 40% pattern objective and its four-position depth head with a separate pattern fusion head. The newly trained support-first arm controls those shared changes. Historical scores are context, not a controlled comparison to the candidate.

- Random initialization in both arms; no previous stage-2 weights transferred.
- All 125,203 training image keys; identical shuffled order, fresh pixel crops/flips, and frozen BF16 encoding with FP32 OMP. The separate 1,024-image stage-2 holdout and 300 official validation images remain excluded from stage-2 training.
- Same calibrated hard joint-pattern targets under the selected support Gram metric. No coefficient-target noise or geometry auxiliary loss.
- Effective batch 128 with two microbatches of 64; dropout 0.15; AdamW betas (0.9, 0.95); matrix weight decay 0.05; gradient clipping at norm 1.
- LR warms for 0.5 image epoch to 8e-5, falls exponentially to 1e-5 by epoch five, then uses the existing 60-epoch cosine horizon. The **pilot caps each arm at 12 epochs** without compressing the LR schedule into that shorter interval.
- Evaluate epoch one and every two epochs. Stop after three evaluations without cumulative improvement of 0.01 held-out joint NLL **or** 0.25 screening FID, starting at epoch four. Retain separate best-generation and best-heldout checkpoints.
- FID-4096 uses atom top-k 2048, joint-pattern nucleus p=0.5, temperature 1, seed 18701, batch 128, and the original RQ-VAE Inception and Church reference statistics in both arms.
- At termination, evaluate each selected generation checkpoint on 4,096 samples with independent seed 38701. A later 50,000-image comparison would be needed before a strong quality claim. No automatic extension beyond the bounded pilot is configured.
- Save model, optimizer, exact data-stream state, both stopping monitors, and pending evaluation state. Graceful pause and resume preserve the experiment. Existing shared trainer/model sources remain intact.

Image quality and the independently seeded generation comparison determine whether this direction merits a larger experiment. Teacher-forced sign accuracy is conditional on different information in the two arms and must not be used as a like-for-like quality score.

## Verification

The focused codec/data/model tests cover causal information flow, signed weighted prefixes, exact complete-code recovery, joint-loss gradients, distinct-atom sampling, all-site cached/full-forward equivalence, and identical initial tensors. Full GPU checks use the actual 199M models, including the production batch size and both orderings. Near-untrained smoke FIDs exercise the evaluation lifecycle and are not quality results.

Artifacts and exact verification outcomes are recorded in `outputs/church-pattern-order-20260911/verification.json`. The launcher checks the verification, source hashes, and unchanged codebook before starting. Reproduction uses `src/church_pattern_order.py`, `scripts/train_church_pattern_order.py`, `scripts/launch_church_pattern_order.py`, and `scripts/verify_church_pattern_order_cache.py`.

All **24 focused tests pass**. Four uninterrupted GPU updates match two updates followed by pause/resume and two more updates **bitwise**, including model, optimizer, data stream, both monitors, best-checkpoint metadata, and pending evaluation. The check changes GPU and DataLoader worker count across resume. Both 199M arms train successfully at the actual 128/64 batch sizes. FP32 cached and full teacher-forced logits agree across every site on two images for each ordering. The negligible-LR lifecycle run stops at epoch two, selects epoch one, and completes an independent-seed generation check. **16,384 generated integers** independently recover their supports and all signed coefficient-bin IDs, with distinct atoms throughout.

## Launch status

Both arms launched on September 11 at approximately **16:02 UTC**, and by 16:03:20 both had reached update 70 with finite loss/gradients, zero coefficient clipping, and a saved optimizer/data-stream checkpoint. Configuration equality was checked: only ordering, output directory, and W&B ID differ. The first production generation-quality comparison is scheduled after epoch one and was not yet available at this launch check.

- [Coefficient-first run](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-order-pattern-first-199m-20260911): GPU 0, PID 46839.
- [Support-first control](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-order-support-first-199m-20260911): GPU 1, PID 46840.
- [Launch health](../outputs/church-pattern-order-20260911/launch-health.json) and [verification](../outputs/church-pattern-order-20260911/verification.json).
- Run directories: `outputs/church-pattern-order-20260911/{pattern-first,support-first}`; durable logs are alongside them.

Before launching, the calibrated soft prior was gracefully paused at step **58,272**, epoch **59.5224**, and the previous 201M support-pattern prior at step **14,866**, epoch **15.1850**. Both saved model/optimizer/data-stream checkpoints were loaded and verified against their acknowledged paused steps. Their best and latest checkpoints remain preserved. [Pause receipt](../outputs/church-pattern-order-20260911/previous-runs-pause.json).

## Relation to a looped RQ-Transformer

Looped transformers reuse a recurrent block to perform additional computation in hidden-state space. Published recurrent-depth work demonstrates benefits on language reasoning tasks; it does not establish that looping improves LSUN Church sparse-code generation. [Geiping et al., Scaling up Test-Time Compute with Latent Reasoning](https://arxiv.org/abs/2502.05171), [Saunshi et al., Reasoning with Latent Thoughts](https://arxiv.org/abs/2502.17416).

Our proposed follow-up is to test repeated depth-head computation after establishing whether coefficient-first conditioning helps. Standard hidden-state looping can refine a decision before sampling it; it does not automatically revisit previous sampled atom or pattern IDs. Explicitly revising those IDs would require an additional inference/training design. A fair loop comparison should measure generation quality at matched compute, train the recurrent model for its intended loop counts, and maintain a separate causal KV cache for each recurrent pass. The present pilot has no looped layers, so any observed difference can be attributed to its documented ordering/context change rather than adding recurrence simultaneously.
