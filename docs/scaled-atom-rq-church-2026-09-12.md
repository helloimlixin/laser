# Frozen scaled-atom RQ experiment on LSUN Church

Completed: eight signed coefficient levels retain reconstruction quality closely
enough to be a candidate for a separate prior experiment. The 4,096-image matched
reconstruction FID is **8.5843**, compared with **8.5166** for the frozen OMP4
control. No new autoregressive training was launched by this reconstruction study.

The experiment replaces OMP's coefficient refitting with residual nearest-neighbor
selection from a shared, structured codebook. It does not train the encoder,
dictionary, decoder, or a new prior. The original RQ-VAE/RQTransformer baseline
continues separately.

For normalized dictionary atom `D[:, a]` and signed coefficient level `q[b]`,
entry `1 + a * B + b` is `q[b] * D[:, a]`. Entry zero is the unique zero vector.
Each 8×8 latent location has four tokens. At each depth the selected vector is
subtracted from the current residual. Earlier contributions remain fixed, and an
atom may be selected again. Support and coefficient are selected jointly.

The search is mathematically identical to an explicit expanded RQ codebook:
for each atom, quantize its scalar projection onto the nearest level, then choose
the atom/level with greatest residual squared-error reduction. Compare this
reduction with zero to account for the zero entry. Atom norms are included, so
the implementation also handles unnormalized dictionaries.

## Source and protocol

- Frozen LASER Church checkpoint:
  `outputs/lsun-church-sign-pattern-20260910/assets/best_rfid_slot1_model.pt`.
- SHA256: `93f19adc8ca2cfae0cf629d48d2d6f5f7039440b1a48ad1c7ba43cf17932a388`.
- Dictionary: 256 dimensions × 16,384 atoms; four residual steps.
- This is the sparse checkpoint previously identified as the 4.90 rFID tokenizer.
  Historical rFID is not used as the contemporaneous comparison.
- Coefficient calibration: Church training indices 60,000–62,047, deterministic
  resize/center crop, using continuous matching-pursuit coefficients. Symmetric
  scalar Lloyd fitting includes a fixed zero center and shares nonzero levels
  across all depths. Encoder and decoder weights remain frozen.
- Reconstruction screen: training indices 0–4,095, disjoint from calibration.
  These are not unseen examples for the original trained tokenizer.
- Additional imagewise validation: all 300 official Church validation images.
- Controls: the existing continuous OMP4 tokenizer and continuous MP4 without
  coefficient refitting. MP4 separates a change in residual selection from
  coefficient discretization; it is not a formal lower bound for final RQ error.
- Signed nonzero level counts: 2, 4, 8, 16, 32, giving vocabularies of 32,769;
  65,537; 131,073; 262,145; 524,289, including zero.
- All compared encodings, quantization, decodings, and Inception features use
  FP32 with CUDA matmul/cuDNN TF32 disabled, the same inputs and decoder, and
  output pixels clamped to [0, 1]. Historical BF16 latent caches are not reused.
- Screen rFID is reported both against the same original images and against the
  published Church reference. A 4,096-image screen is not a 50,000-image rFID.
- No physical noise is added in this reconstruction experiment.

## Verification and artifacts

`tests/test_scaled_atom_rq.py` checks every greedy step against an explicit
expanded book, hard codes and stochastic soft targets against the released
RQBottleneck, zero and repeated-atom behavior, and the OMP control against an
independent least-squares solve. A fifth test feeds these tokens to the released
RQTransformer and verifies that changing either support or coefficient affects
future depth and spatial predictions without affecting earlier predictions. All
five tests passed. An additional two-image
integration check found exactly identical supports and coefficients to
`LaserAux.encode_sparse_components`; its receipt is
`outputs/church-scaled-atom-rq-20260912/omp-control-parity.json`.

The runner is `scripts/tools/evaluate_scaled_atom_rq.py`, with the quantizer in
`src/scaled_atom_rq.py`. Reproduction:

```bash
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  /tmp/laser-sign-venv/bin/python scripts/tools/evaluate_scaled_atom_rq.py \
  --output outputs/church-scaled-atom-rq-20260912/sweep \
  --calibration-images 2048 --screen-images 4096 \
  --levels 2 4 8 16 32 --batch-size 16
```

The output directory contains fitted levels and exact image indices, code and
checkpoint hashes, frozen-state fingerprints, token usage by depth, Inception
features, reconstruction contact sheets, metrics, and an atomic progress file.
The soft-target implementation is currently a dense verification reference;
large-vocabulary training would need an appropriately sized batch or chunked
target computation. Passing reconstruction tests does not establish that an
autoregressive model can generate these codes well.

## Results

All rows use the same 4,096 screen images. PSNR is calculated from mean pixel MSE.
Both FID columns are 4,096-reconstruction estimates, including the column using
published reference statistics. Neither should be described as 50,000-image rFID.

| Quantizer | Discrete vocabulary | Matched rFID | rFID vs published reference | PSNR (dB) |
| --- | ---: | ---: | ---: | ---: |
| Frozen OMP4 control | continuous | 8.5166 | 7.4490 | 18.6078 |
| Continuous MP4, no refitting | continuous | 8.5349 | 7.4340 | 18.5540 |
| RQ, 2 signed levels | 32,769 | 11.6166 | 9.7589 | 17.5058 |
| RQ, 4 signed levels | 65,537 | 8.7827 | 7.5630 | 18.3136 |
| RQ, 8 signed levels | 131,073 | 8.5843 | 7.4546 | 18.4900 |
| RQ, 16 signed levels | 262,145 | 8.5092 | 7.4124 | 18.5354 |
| RQ, 32 signed levels | 524,289 | 8.5725 | 7.4615 | 18.5487 |

The continuous MP4 control is close to OMP4 in reconstruction FID. This indicates
that removing coefficient refitting is viable for this frozen tokenizer. The
two-level book is too coarse; four levels are a lower-cost alternative with a
larger reconstruction penalty. Eight levels have a matched rFID gap of +0.0676
and PSNR loss of 0.1177 dB versus OMP4. Increasing to 16 or 32 levels provides
little additional reconstruction FID benefit in this screen. Differences this
small should not be interpreted as proof that any variant beats OMP4.

The eight nonzero levels, in physical latent units, are approximately
`±[1.97670, 3.92997, 6.52144, 9.65730]`, plus the unique zero vector.
`sweep/scaled-atom-codebooks.pt` stores the normalized dictionary and all five
calibrated level sets. `sweep/candidate.json` selects the eight-level variant and
records its source checkpoint and limitations. Loading the quantizer alone:

```python
spec = torch.load('outputs/church-scaled-atom-rq-20260912/sweep/scaled-atom-codebooks.pt',
                  map_location='cpu', weights_only=True)
quantizer = ScaledAtomRQ(spec['dictionary'], spec['levels']['8'], depth=spec['depth'])
```

With the released Church architecture unchanged except for its classifier output
width, the eight-level prior would have 487,644,161 parameters, compared with
370,087,936 for the original 16,384-entry RQ prior. The released architecture
projects fixed codebook vectors into the transformer; it does not need another
trainable 131,073-entry input embedding table. The capacity calculation was
verified by constructing the released model on the PyTorch meta device; this was
not a training run.

On all 300 official validation images, OMP4/eight-level RQ PSNR was
18.3769/18.2629 dB, respectively. On the reconstruction screen, the eight-level
variant increased mean paired squared Inception-feature error by 2.51%. The
additional-analysis artifact includes per-image paired confidence intervals for
this error difference; these are not FID confidence intervals.

The sweep completed in 703 seconds. Source hashes were verified after execution.
The full frozen state had identical SHA256 fingerprints before and after:
`421c01f6039c625bf00360e2ea67f372af63dba3bdb6e633e66973aace61245b`.
The calibrated codebook artifact SHA256 is
`275bc44c7eed0b1b7f14308e49b1031ec8b4b0258a2dcb91ac2bc69ec6b8a1f1`.
