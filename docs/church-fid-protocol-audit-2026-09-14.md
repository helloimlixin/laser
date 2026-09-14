The Church FID audit found no calculation error explaining the LASER plateau. A fresh replay of the LASER epoch-100 checkpoint reproduced the logged FID4096 exactly, and the released evaluator independently recovered the score from the saved RGB images. The material comparison issue is mixing sample counts and reconstruction versus generation metrics.

| Check | Recorded / streaming result | Independent result |
|---|---:|---:|
| LASER epoch 50, generation FID50k | 13.884514101157 | 13.884514101159 |
| LASER epoch 100, generation FID50k | 14.255952812107 | 14.255952812106 |
| LASER epoch 100, fresh generation FID4096 | 16.744624149472 | 16.744621706728 |
| Matched reconstruction FID4096, active stage-1 accumulator | 8.392328128769 | 8.392328128772 |

The first two checks use an independent symmetric covariance calculation: the square root of the reference covariance and the eigenvalues of its sandwich with the generated covariance. This does not use the training evaluator's `sqrtm` function. It validates the saved 50k-image statistics and final arithmetic; those 50k images were not regenerated in this audit.

The third check regenerated 4,096 images from the retained epoch-100 LASER checkpoint, using exactly the recorded tokenizer, compact codebook, two ranks, seeds, batch sizes, and sampling settings. It recovered the historical score with zero difference. All 4,096 code sequences were distinct, and the two ranks had different streams. The RGB arrays were saved as float32 pickles and read back using the released `compute_statistics_from_files` implementation, with a different Inception batch size. Its dense NumPy statistics and released FID equation differed from the streaming score by only **0.00000244**. The small discrepancy is consistent with float32 versus float64 mean accumulation; covariance differences were below 1e-15. Model and tokenizer states remained unchanged.

The fourth check ran the exact historical accumulator used by the active fine-tune in two CPU processes, feeding recorded matched original/reconstruction features. Both counts were 4,096 after reduction. Its result matched dense NumPy aggregation to 2.5e-12. An identical-distribution FID check returned approximately zero (-3.5e-12). Separately, the historical stage-1 Inception and the stage-2 Inception had identical weights and produced exactly identical features on fresh LASER outputs on both GPUs. Serial versus batched LASER decoding had pixel MSE below 3.4e-13.

Sample count changes the measured score substantially:

| Released Church model, same saved pool of generated images | FID |
|---|---:|
| All 50,000 images | 7.670966 |
| Random subset of 4,096, seed 0 | 10.415683 |
| Random subset of 4,096, seed 1 | 10.447818 |
| Random subset of 4,096, seed 2 | 10.401769 |
| Random subset of 4,096, seed 3 | 10.325777 |

The four subset scores average 10.397762. These subsets demonstrate the sample-count effect; their difference from FID50k is not a universal correction to subtract from other models. Generation comparisons should use FID50k with the same reference and sampling protocol. FID4096 remains useful for training trends and comparisons with other FID4096 evaluations. Four subsets are not a confidence interval for independent 50k evaluations.

The previous published-model reproduction measured FID50k 7.67 against the [released repository's reported 7.45](https://github.com/kakaobrain/rq-vae-transformer/blob/main/README.md). That residual 0.22 gap is not fully explained by this audit; the reproduction is approximate. The current checks do not establish that the 0.37 change between LASER epochs 50 and 100 is statistically significant.

The reference file currently used by both LASER stage-2 drivers is the official Church file, SHA256 `809489d8316b9e6eb9dc3bc021b6d602f4b6d816cc80621c6b9c189a9253a7f6`. Its hash was checked again and matches the previous byte-for-byte verification against the official statistics archive. Both evaluators use the released TensorFlow-FID-compatible 2,048-dimensional Inception features, rather than an ImageNet classification-weight substitute. Pixel handling agrees with the [released sampling implementation](https://github.com/kakaobrain/rq-vae-transformer/blob/main/main_sampling_fid.py) and [FID implementation](https://github.com/kakaobrain/rq-vae-transformer/blob/main/rqvae/metrics/fid.py): convert decoder output to [0,1], clamp, and evaluate continuous float32 RGB. A PNG or uint8 round-trip would change this protocol. Inception performs its own resize to 299×299.

The metric meanings are recorded in [fid-protocol.json](/workspace/Projects/laser/outputs/church-laser-three-epoch-20260914/fid-protocol.json):

- Stage-2 FID4096 and FID50k compare generated images against the official Church reference. Both use temperature 1, top-k 1400, top-p 1, FP16 AR sampling, FP32 decoding/Inception, FP64 moments, and covariance denominator N−1. Generated sample counts have no padding.
- Stage-1 `valid/rfid` compares originals with their reconstructions on the **training population**, following the released LSUN dataset factory. It is not a held-out generation score. Two-rank `DistributedSampler` evaluates 126,228 entries for 126,227 unique images, repeating the first image once. This minor padding discrepancy is documented; its exact score effect has not been measured. The current fine-tune's source was kept intact.
- The subsequent compact-tokenizer screen computes matched rFID4096 and, separately, reconstruction FID4096 against the official reference. Its quantized bottleneck and sample count differ from the continuous LASER stage-1 rFID. Comparing those numbers directly would confound both changes.

No FID formula or active training source was changed. The audit added a reproducible tool, saved RGB samples/codes/features/statistics, and explicit protocol documentation. All 356 protected pipeline source files still match their launch manifest. Three-epoch fine-tuning continues, with fresh stage-2 training still queued behind tokenizer preparation.

Reproduction commands (from `/workspace/Projects/laser`, using `/tmp/laser-sign-venv/bin/python` and fresh output directories):

```bash
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
/tmp/laser-sign-venv/bin/python scripts/tools/audit_church_fid_protocol.py statistics --output /tmp/church-fid-statistics-audit
/tmp/laser-sign-venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 scripts/tools/audit_church_fid_protocol.py reconstruction --output /tmp/church-rfid-aggregation-audit
/tmp/laser-sign-venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 scripts/tools/audit_church_fid_protocol.py images --output /tmp/church-fid-image-audit
```

Evidence: [50k arithmetic and sample-count checks](/workspace/Projects/laser/outputs/church-fid-protocol-audit-20260914/statistics/result.json), [fresh LASER / released evaluator comparison](/workspace/Projects/laser/outputs/church-fid-protocol-audit-20260914/laser-epoch100/result.json), [distributed reconstruction check](/workspace/Projects/laser/outputs/church-fid-protocol-audit-20260914/reconstruction/result.json), [audit tool](/workspace/Projects/laser/scripts/tools/audit_church_fid_protocol.py).
