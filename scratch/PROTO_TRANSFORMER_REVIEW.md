# `proto.py` Transformer Review Notes

Date: 2026-03-12

This note captures the two expanded issues from the review of [`scratch/proto.py`](./proto.py), focused on the stage-2 transformer paths for quantized and real-valued sparse coefficients.

## 1. `--quantize_sparse_coeffs` is not a usable boolean CLI flag

### What the code does

The flag is defined as:

- [`scratch/proto.py#L3492`](./proto.py#L3492)

```python
parser.add_argument("--quantize_sparse_coeffs", type=bool, default=True)
```

With `argparse`, `type=bool` does **not** parse strings like a normal CLI boolean. It calls `bool(...)` on the raw string. In Python:

- `bool("False") == True`
- `bool("0") == True`
- `bool("no") == True`
- only `bool("") == False`

So a command like:

```bash
python3 scratch/proto.py --quantize_sparse_coeffs False
```

still sets `args.quantize_sparse_coeffs` to `True`.

### Why this matters in this codebase

That flag decides which bottleneck and transformer path the run uses:

- Run naming and output directory suffixing use it in [`scratch/proto.py#L3617`](./proto.py#L3617) and [`scratch/proto.py#L3619`](./proto.py#L3619).
- The LASER bottleneck is constructed from it in [`scratch/proto.py#L3681`](./proto.py#L3681).
- Token precompute branches on it in [`scratch/proto.py#L2877`](./proto.py#L2877) and [`scratch/proto.py#L2881`](./proto.py#L2881).
- Stage 2 infers whether it is in the real-valued path from whether `coeffs_flat` exists in [`scratch/proto.py#L4014`](./proto.py#L4014).

In practice, that means a user can believe they launched the real-valued coefficient model, while the run actually stays in quantized mode end-to-end.

### Concrete failure mode

If you pass `--quantize_sparse_coeffs False` expecting:

- atom-only tokens
- cached real-valued coefficients
- the real-valued transformer with coefficient regression

you instead still get:

- interleaved atom/bin tokens
- no cached `coeffs_flat`
- `real_valued = False` in stage 2
- the quantized-only transformer path

So the real-valued transformer branch is effectively unreachable from normal CLI usage.

### Why I consider this a high-severity issue

This is not a cosmetic CLI bug. It changes the entire experiment type while leaving the command line looking plausible. That is especially risky here because the quantized and real-valued paths train different models, cache different data, and decode samples differently.

## 2. The real-valued stage-2 path does not use the same coefficient distribution as stage 1

### What stage 1 trains the decoder to see

Stage 1 starts from raw OMP coefficients, but in the non-quantized path it **clamps** them before reconstructing the latent that the decoder is trained against:

- dense bottleneck: [`scratch/proto.py#L1000-L1017`](./proto.py#L1000-L1017)
- patch bottleneck: [`scratch/proto.py#L1423-L1440`](./proto.py#L1423-L1440)

The relevant lines are:

- [`scratch/proto.py#L1014`](./proto.py#L1014)
- [`scratch/proto.py#L1438`](./proto.py#L1438)

```python
coeffs_for_recon = coeffs_flat.clamp(-self.coef_max, self.coef_max)
```

So even in "real-valued" mode, the decoder is trained on a bounded coefficient manifold: every latent reconstruction uses coefficients clipped into `[-coef_max, coef_max]`.

### What stage 2 caches and models

The stage-2 dataset is built from `encode_to_atoms_and_coeffs()`:

- [`scratch/proto.py#L1599-L1607`](./proto.py#L1599-L1607)

That path returns the raw OMP coefficients from `_encode_sparse_codes()` without clamping them first. `precompute_tokens()` then stores those raw values:

- [`scratch/proto.py#L2881`](./proto.py#L2881)
- [`scratch/proto.py#L2898-L2899`](./proto.py#L2898-L2899)

So the transformer is trained on coefficient targets that may exceed `coef_max`, even though stage 1 never trained the decoder on reconstructions using those values.

### Where the mismatch shows up

There are three distinct places where this matters.

#### A. Reference statistics for sample filtering

Stage-2 preview filtering decodes cached training-set atoms and coefficients to estimate "good" sample statistics:

- [`scratch/proto.py#L2203-L2209`](./proto.py#L2203-L2209)

In the real-valued branch it uses:

```python
imgs = ae.decode_from_atoms_and_coeffs(tok, coeff)
```

`decode_from_atoms_and_coeffs()` reconstructs from the supplied coefficients directly:

- [`scratch/proto.py#L1626-L1642`](./proto.py#L1626-L1642)

There is no clamp on the coefficients in that decode path. So your sample-quality reference distribution is computed from raw cached coefficients, not from the clamped coefficient distribution stage 1 used to train the decoder.

#### B. Real-valued transformer training targets

In stage 2, the real-valued branch normalizes and predicts the cached coefficient values directly:

- [`scratch/proto.py#L3090-L3092`](./proto.py#L3090-L3092)
- [`scratch/proto.py#L3119-L3124`](./proto.py#L3119-L3124)

For `mse` and `huber`, the coefficient regression target is the normalized raw cached coefficient. If OMP produced a coefficient outside the allowed stage-1 reconstruction range, the transformer is still asked to model it.

That means the transformer is learning a target distribution that is not identical to the one that defined the stage-1 decoder's training manifold.

#### C. Sample generation

At sampling time, the real-valued transformer produces denormalized coefficients in:

- [`scratch/proto.py#L1976-L2024`](./proto.py#L1976-L2024)

Those generated coefficients are then decoded directly in:

- [`scratch/proto.py#L3240-L3256`](./proto.py#L3240-L3256)

Again, no clamp is applied before `decode_from_atoms_and_coeffs()`.

So the stage-2 sampler can generate latents using coefficient magnitudes the stage-1 decoder was never optimized around.

### Nuance: some stage-2 losses partly paper over this, but only inside the loss

For `recon_mse` and `gt_atom_recon_mse`, the code **does** clamp `pred_coeff` and `target_coeff` before building the latent-space reconstruction loss:

- [`scratch/proto.py#L3132-L3144`](./proto.py#L3132-L3144)
- [`scratch/proto.py#L3148-L3160`](./proto.py#L3148-L3160)

That helps align the loss with the stage-1 bottleneck behavior, but only for those two loss variants and only inside the training loss computation.

It does **not** fix:

- the cached stage-2 coefficient targets
- the reference-stat decode path
- the sample-generation decode path
- the `mse` and `huber` coefficient-target mismatch

### Why I consider this high severity

The model family is trying to learn a prior over stage-1 sparse codes. If stage 1 defines the usable code manifold with clamped coefficients, but stage 2 trains and samples from unclamped coefficients, then the prior is not actually learning the same latent space the decoder was trained to invert.

The likely consequences are:

- noisier or unstable real-valued coefficient targets
- sample filtering based on the wrong decoded distribution
- generated samples decoding from out-of-manifold coefficients
- misleading comparisons between quantized and real-valued runs

### Short version

Stage 1 says: "real-valued coefficients are continuous, but only within a clipped range for reconstruction."

Stage 2 says: "learn and generate the raw OMP coefficients anyway."

Those are not the same data distribution.
