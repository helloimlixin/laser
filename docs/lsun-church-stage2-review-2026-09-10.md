LSUN Church stage 2 review — September 10, 2026

The recent evidence points to difficulty generalizing the joint distribution of atom supports and signed coefficients. The masked model has a particularly clear sign problem. The autoregressive pair model can learn coefficients and already produces substantially better samples; its later training increasingly overfits. The available audits do not identify a basic causality or likelihood implementation failure in that model.

This review retrieved metadata for the 150 most recent Church-matching runs in `helloimlixin-rutgers/laser`, inspected selected unsampled histories and diagnostic files, and viewed the masked coefficient oracle grids and selected guided samples. Values below are logged measurements, not fresh GPU evaluations. The newest event-model source files and September checkpoints are absent from this checkout, so their implementations could not be reproduced locally.

The two newest experiments are [raster event order](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-event-raster-20260910-0602) and [depth-first event order](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-event-depth-first-20260910-0602). Both are marked `crashed`; their last logged training steps are 3,680 and 3,080. Each has one FID evaluation, at step 2,048: **47.4433** and **56.3363**, respectively, using 2,048 generated images. Their configured warmup is 2,048 steps and total budget is 32,768 steps. This is an early comparison, not a converged verdict on event order. The retrieved raster console log stops at step 90 without an exception; it does not establish why the process stopped. W&B state alone does not prove numerical failure.

**The strongest coefficient diagnostic concerns signs.** With real supports supplied to the masked model, replacing predicted signs with true signs improves PSNR against the true-code reconstruction from **14.741 to 21.226 dB**. Replacing magnitudes with true magnitudes only reaches **14.806 dB**. The aligned grids visibly recover building structure when signs are supplied. This is an oracle diagnosis, not unconditional generation, and the sign-versus-magnitude result is specific to that masked checkpoint. [Masked crossover/oracle run](https://wandb.ai/helloimlixin-rutgers/laser/runs/churchcross-masked-20260909-170315)

The crossover also shows that improving coefficients alone will not repair masked support generation:

| Generated support source | Coefficient source | FID, 2,048 images |
| --- | --- | ---: |
| Autoregressive | Autoregressive | 19.3452 |
| Autoregressive | Masked | 32.8872 |
| Masked | Masked | 77.2574 |
| Masked | Autoregressive | 112.0222 |

Both runs use seed 2701, the same frozen Church tokenizer and reference statistics, and the same source checkpoints. Crossover scores include compatibility effects between each model and the other model's generated supports; the table does not decompose error additively. [AR-support crossover](https://wandb.ai/helloimlixin-rutgers/laser/runs/churchcross-ar-20260909-170315), [masked-support crossover](https://wandb.ai/helloimlixin-rutgers/laser/runs/churchcross-masked-20260909-170315)

**The autoregressive prior learns training codes but develops a large generalization gap.** In the scratch control run, matched evaluation at step 14,336 gives:

| Metric | Training evaluation | Official validation |
| --- | ---: | ---: |
| Atom NLL | 2.9996 | 11.3720 |
| Coefficient NLL | 4.2284 | 6.3117 |
| Coefficient sign accuracy | 99.33% | 88.66% |

Its held-out coefficient NLL is 5.6006 at step 4,096 and 6.3117 at step 14,336, while training coefficient NLL falls from 5.3059 to 4.2284. Yet screening FID continues improving beyond the best validation likelihood: 25.4514 at step 4,096, 18.6018 at 10,240, and 18.8853 at 14,336. Checkpoint selection should therefore track both held-out likelihood and generation metrics. This run excludes 1,024 training-population images for a separate probe and evaluates 300 official validation images. [Scratch control history](https://wandb.ai/helloimlixin-rutgers/laser/runs/churchscratch-control-20260909-2253)

R-drop improves several held-out metrics but its matched 50,000-image result is **20.2684**, versus **16.0909** for continued control. That particular regularization setting has not improved generation. [R-drop evaluation](https://wandb.ai/helloimlixin-rutgers/laser/runs/churchgeneralization-rdrop-fid50k-20260909-1930), [continued control evaluation](https://wandb.ai/helloimlixin-rutgers/laser/runs/churchgeneralization-control-fid50k-20260909-1930)

**Keep the epoch-50 pair model as the established September 9 baseline.** The `churchftdc` run reaches 50,000-image FID **13.6206** at epoch 50 / step 3,050; subsequent logged evaluations through epoch 90 range from 14.7168 to 15.5363. Its final summary contains the epoch-90 FID alongside a later training epoch, so the history is necessary to identify the evaluated checkpoint. [Training run](https://wandb.ai/helloimlixin-rutgers/laser/runs/churchftdc-20260909024359)

| Evaluation of the selected pair checkpoint | FID, 50,000 images |
| --- | ---: |
| Baseline reevaluation, seed 2701 | 13.7246 |
| Both atom and coefficient temperatures 0.8 | 13.5249 |
| Atom-only guidance strength 0.25 against epoch-10 weak model | 13.1112 |

These are logged single-seed evaluations, not uncertainty estimates. The guidance result changes atom logits and leaves coefficient guidance at zero. Its improvement reinforces the need to consider support modeling as well as coefficients. The guided sample grid contains recognizable churches, with remaining distortions and watermark-like text. [Baseline](https://wandb.ai/helloimlixin-rutgers/laser/runs/churchviews-fid50k-baseline-20260909-1804), [temperature 0.8](https://wandb.ai/helloimlixin-rutgers/laser/runs/churchguide-fid50k-jointtemp08-20260909-1847), [atom guidance](https://wandb.ai/helloimlixin-rutgers/laser/runs/churchguide-fid50k-025-20260909-1847), [guided samples](https://wandb.ai/helloimlixin-rutgers/laser/runs/churchguide-fresh-guided025-20260909-1847)

The selected configuration uses 16,384 atoms, four pairs per 8×8 site, 2,048 coefficient bins, hard coefficient targets, full pair autoregression, depth-specific coefficient heads, a two-layer coefficient micro-transformer, and no geometry loss. Inference uses atom top-k 250 and untruncated coefficient sampling, with both temperatures initially 1. Physical decoding depends on the matching tokenizer, coefficient range and scales; the September configuration differs from the older August launch scripts.

**Several existing checks narrow the possible explanations.** The saved construction audit reports zero current/future-target influence, an objective exactly equal to normalized joint NLL, FP32 cached-versus-parallel coefficient logit error about 0.000050, and coefficient NLL **0.000844** after deliberately overfitting 16 images. These checks demonstrate basic learnability and consistency on the audited inputs; they do not establish generalization or exclude every bug. Its geometry audit finds no final-coefficient versus orthogonal-innovation sign disagreements across 2,048 images. Switching to orthogonal coordinates is therefore not supported as a fix for this checkpoint's sign problem by that audit. [Construction audit, report.json and geometry.json](https://wandb.ai/helloimlixin-rutgers/laser/runs/churchconstruction-20260909-1912)

The older source comment that final OMP coefficients depend on later-selected supports should not be interpreted as invalidating full pair autoregression. A joint model may validly factorize as `p(a_d | previous pairs) p(c_d | previous pairs, a_d)`. The relevant leakage test is whether the current or future targets enter those predictors. The saved audit passes that test on its inspected inputs.

Coefficient resolution is also not the leading explanation: the September 10 screen finds that four coefficients quantized on a shared physical 128-bin grid incur only **0.00963 dB** image PSNR loss versus its baseline and zero clipping on 300 images. This is a reconstruction screen, not evidence that a 128-bin prior generates better images. At the same 4,096-step screen, canonical-order physical-128 gives FID 25.8737 versus the existing control's 25.4514; the simpler representation has not yet demonstrated a generation win. [Rate audit](https://wandb.ai/helloimlixin-rutgers/laser/runs/churchcoderate-audit-20260910-0048), [canonical-order trial](https://wandb.ai/helloimlixin-rutgers/laser/runs/churchcoderate-canonical-20260910-0048)

My next experiment would preserve the selected AR support path and change only coefficient modeling: explicitly predict sign, then magnitude conditioned on sign and the same causal history. This is a hypothesis to test, not an established fix. Compare it against the unchanged categorical head with the same initialization, training budget and sampler; report sign NLL/accuracy and physical coefficient error by depth, including true-support and generated-support conditions. Retain the existing small-sample screens, then confirm a promising candidate with matched 50,000-image evaluations. For the masked branch, repeat the oracle sign test and repair support generation before treating coefficient improvements as a complete solution.

No training jobs, checkpoints, or W&B records were changed during this review.
