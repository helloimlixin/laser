During the audit, production improved to **FID 10.015038 at epoch 60**, comprising mean 2.846459 and covariance 7.168579. Thus the plateau is not a proven permanent floor. The detailed frozen covariance/PCA comparison below remains explicitly tied to epoch55. The pending decision about stopping training should take this later improvement into account.

The current best at the frozen epoch-55 comparison is FID50k **10.6021**. The covariance term has plateaued since roughly epoch 30. It accounts for **96.6% of the gap** to the earlier verified 9.6012 result. This locates the output mismatch; it is not an identified training mechanism.

| Checkpoint | Mean term | Covariance term | FID50k |
|---|---:|---:|---:|
| Earlier LASER best114 | 3.0377 | 6.5635 | 9.6012 |
| Predecessor best55 | 3.5572 | 7.3672 | 10.9243 |
| Repair best55 | 3.0717 | 7.5304 | 10.6021 |

These rows use the same official reference and 50,000 generated samples, but different training recipes and sampling policies. The repaired-versus-predecessor improvement is entirely in the mean term; covariance is 0.1633 worse. The 0.0049 FID improvement from repaired epoch 40 to 55 has no demonstrated practical significance without independent draws.

The covariance discrepancy is mostly its structure, not a scalar shortage of variance. In the real-reference principal-component basis, the per-axis variance contributions are nearly identical: 0.680946 for the earlier9.601 checkpoint and 0.682160 for the repair. The remaining cross-axis contributions are 5.882534 and 6.848287. These are statistical directions, not labeled scene attributes. Even the algebraically optimal uniform scaling of mean-preserved repaired Inception features would improve FID by only 0.0467. That operation is not a realizable image-generation fix. Increasing coefficient temperature or adding covariance regularization is not justified by this decomposition alone.

The optimized loss is token cross-entropy, not FID. Therefore a plateau in FID covariance cannot itself stop gradient descent. Training loss continues improving with finite gradients and a correctly advancing schedule. No new demonstrated causality, cache, loss-normalization or optimizer implementation bug was found by this audit.

The actual-target likelihood audit showed worsening held-out atom and coefficient prediction. A new inference-only test checks whether this merely penalizes alternative sparse representations: sample all four atom/coefficient pairs at each site under the same real spatial history, sum their physical vectors, and evaluate a conditional energy score. It includes a predictive-diversity term and gives identical scores to identical vectors, regardless of atom encoding. This uses the [energy-score construction](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf), with lower being better here.

| Physical conditional score, actual noisy teacher target | Epoch10 | Epoch40 | Epoch55 |
|---|---:|---:|---:|
| Training images | 5.1670 | 3.2928 | 2.9342 |
| Held-out images | 5.4036 | 5.6894 | 5.7299 |

Epoch40-to55 held-out deterioration is +0.04051, with paired-image bootstrap95% interval [0.01654,0.06399]; training improves -0.35858. The clean nearest-bin reference gives the same direction. The test used64images per split, eight draws per site, strict checkpoint loading and dense/cached FP32 parity; it took52.5seconds and2.26GiB allocated GPU memory. Fixed-sample bootstrap uncertainty does not cover independent training runs or all sampling randomness. This measures conditional physical prediction with real spatial contexts, not full spatial rollout or FID. FID improves greatly from10to40 despite worsening held-out conditional scores, so neither this score nor atom NLL is a sufficient FID-selection rule.

A correction to the earlier teacher-only emphasis is necessary. The locally trained native RQ control also underperforms the released RQ checkpoint. Its two existing50k draws now score **9.8031 and9.7974**, mean9.80025, against the same official reference; released RQ scores7.6793. The local control used a native residual-dependent stochastic RQ teacher and the released tokenizer. Thus missing compound stochastic atom targets cannot be the complete explanation of the shared gap. The actual historical training helper was not authenticated in uploaded source, and the authors did not release their stage-two training loop; matching the paper's listed settings is not proof of matching every training detail. The [RQ paper](https://arxiv.org/html/2203.01941#S3.SS2.SSS3) supplies the residual teacher formulation, but does not establish the remedy for these local runs.

The current fixed-support/noisy-coefficient teacher is a valid augmented joint distribution. Retaining its cached suffix atoms is not automatically a causality violation. The earlier stochastic bank result is encouraging evidence of attainable quality, but it changed several settings and also eventually overfit. A residual-teacher replacement must satisfy reconstruction and prefix-consistency checks first; the existing tiny no-refit prototype increased residual error substantially and is not ready for production.

The narrow no-refit candidate proposed here changes only completed-site spatial inputs. They currently include the physical contribution plus a learned adapter of atom identity, coefficient-bin identity and physical contribution. The proposed spatial input uses the physical contribution alone. Full-pair depth history and current-atom coefficient conditioning remain intact. This tests whether the spatial prior memorizes a particular sparse decomposition instead of learning from the decoded latent. It preserves the tokenizer, cache, targets, reconstruction and no-refitting constraint. It is a hypothesis: identity information may also be useful, and the native-RQ gap shows this cannot be presumed to solve every failure.

The reviewable plan is `proposed-experiment.json`. It proposes a matched-initialization single-change pilot, reusing the existing control trajectory only after exact replay checks, capped at8GPU-hours or3000updates, whichever comes first. The original cosine horizon stays unchanged. This is a proposed ceiling, not a dollar quote or an authorized new launch. Assess physical conditional scores and image FID components together at matched updates; any final claimed FID gain needs two independent50k confirmation draws. A failed short pilot is a reason to stop spending on that candidate, not mathematical proof of asymptotic failure.

No new training or refitting was performed for this audit. Production was not changed by these scripts. The training-stop question is separate from this analysis; process status should be read from the final execution receipt.

The matched native-RQ control is now complete. It compares our local epoch80 weights with released weights under the exact same frozen released tokenizer, native stochastic residual histories, and soft targets. Full normalized architecture configs agree exactly, weights load strictly, the downloaded local checkpoint matches its authenticated MD5, and the target replay equals the public quantizer bit for bit.

| Native RQ target KL, lower is better | Local epoch80 | Released |
|---|---:|---:|
| Training64 | 2.804 | 4.917 |
| Held-out64 | 8.326 | 5.904 |

The local model's held-out KL is worse by2.421nats, image-bootstrap95% interval [2.314,2.527], while it fits training targets better. Its held-out predictive entropy is also lower:5.029 versus6.186. This is a direct counterexample to an explanation confined to compound token targets.

The first unconditional event has almost equal held-out KL:6.469 local versus6.412 released. At the first depth of sites1–63, with real spatial history available, local KL is **0.760 train /9.168 held-out**, versus released **3.525 train /4.998 held-out**. This localizes the observed failure toward overconfident conditioning on training-image prefixes. It does not uniquely identify the optimizer, regularizer or data/teacher setting that caused it.

The native comparison used64images per split and one fixed stochastic history, took13.93seconds and2.00GiB allocated GPU memory, and performed no optimization. Its source, identities, architecture check, inputs, arrays and bootstrap are in `native_rq/`. Together with the physical conditional test, this is substantially stronger evidence for a spatial generalization problem than a scalar FID plateau or canonical atom NLL alone.

An additional primary-source check found no missing published LSUN augmentation: the [official LSUN transform branch](https://raw.githubusercontent.com/kakaobrain/rq-vae-transformer/341395e562ac347f5eb62db9f5f08b9f2cc42a60/rqvae/img_datasets/transforms.py) uses deterministic resize/center crop even for training. Adding random image views would be a new regularization experiment, not restoration of an established omitted recipe step.

This does not establish the physical-spatial-input candidate as the sole or certain solution. Native RQ already uses physical spatial inputs and still shows a training gap. The candidate is a narrow, no-refit test of the additional encoding-identity channel in our compound model; it should remain budget-limited and compete against the existing control, not become another open-ended run.
