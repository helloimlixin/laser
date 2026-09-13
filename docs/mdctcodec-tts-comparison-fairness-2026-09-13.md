# Scope of the VCTK pretrained-system reference comparison

The reported scores remain valid measurements of the tested checkpoints on the shared evaluation inputs. The framing as a controlled SOTA comparison was too strong.

The prior uses VCTK training, a frozen nominal-6-kbps MDCTCodec-LASER representation, and learned IDs for speakers seen during prior training. F5-TTS and Chatterbox Turbo use their released pretrained weights and an approximately eight-second training-split reference recording for voice conditioning. Training corpora, compute, capacity, architecture, representation constraints, and inference implementations differ. Parameter totals in the raw provenance cover different components, so they should not be presented as a matched capacity comparison.

The common test set contains 100 recordings across 100 known speakers and 70 distinct normalized transcript keys. It has informed earlier experiments. The prior train/validation/test texts are disjoint, but the codec may have trained on target recordings, and overlap in external baseline pretraining is unknown.

The benchmark supports an end-to-end checkpoint comparison under the recorded settings. It does not establish LASER-versus-RVQ superiority, matched-budget training efficiency, zero-shot LASER performance, or a comprehensive SOTA ranking.

The missing controlled experiment is a stage-2 RVQ prior paired with LASER using the same VCTK transcript splits, equivalent codec training budgets, nominal 6-kbps rate, temporal/depth-prior architecture, speaker conditioning, updates, and validation-selection protocol. Any unavoidable vocabulary/output-head parameter differences must be reported. A new confirmation set should be frozen before additional tuning, with multiple seeds when the budget permits. This control has not been launched or evaluated; the existing stage-1 paired training continues under its original ceiling.
