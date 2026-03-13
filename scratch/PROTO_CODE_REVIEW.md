# proto.py Code Review

## Critical / Correctness Bugs

### 1. `_encode_sparse_codes` return mismatch in `PatchDictionaryLearningTokenized` (line 1338-1342)

The method returns **4 values** `(support, coeffs, H, W)`, but the type hint on line 1309 says it returns `Tuple[torch.Tensor, torch.Tensor]`. The parent class `DictionaryLearningTokenized._encode_sparse_codes` returns 2 values. This inconsistency means you can't polymorphically call `_encode_sparse_codes` — callers must always branch on the bottleneck type, which you do, but it's fragile.

### 2. `analyze_patch_spectrum` references non-existent `ae_module.pre` (line 2124)

```python
z = ae_module.encoder(x)
z = ae_module.pre(z)    # <-- LASER has no `.pre` attribute
```

This will crash at runtime. The `LASER` class has `encoder`, `bottleneck`, and `decoder` — no `pre`.

### 3. `generate` atom/coeff masking uses loop variable `_` for parity check (line 1999)

```python
for _ in steps:
    ...
    if (_ % 2) == 0:
```

This relies on `_` being the step index from `range(T)`, which works, but `_` conventionally signals "unused variable." More importantly, **this masking assumes alternating atom/coeff structure**, but `generate()` is called even in `quantize_sparse_coeffs=False` mode where every token is an atom. The content_vocab_size guard at line 1997 is only entered when `self.content_vocab_size is not None`, which is only set when `atom_vocab_size` and `coeff_vocab_size` are both set — so this is technically guarded. Still, it's easy to misconfigure.

### 4. `generate_with_coeffs` doesn't apply atom/coeff vocabulary masking (lines 2044-2056)

Unlike `generate()` which masks special tokens and enforces alternating atom/coeff vocabulary, `generate_with_coeffs` only masks `pad_token_id` and `bos_token_id`. It never restricts sampling to only atom tokens. Since `generate_with_coeffs` is the non-quantized path where every token should be an atom, it should mask out `coeff_token_offset:` range if `content_vocab_size` is set.

---

## Numerical / Training Concerns

### 5. Bottleneck loss gradient disconnect (lines 1048-1054)

```python
dl_latent_loss = F.mse_loss(z_q, z_e.detach())
e_latent_loss = F.mse_loss(z_q.detach(), z_e)
loss = dl_latent_loss + self.commitment_cost * e_latent_loss
```

`dl_latent_loss` pushes gradients into the **dictionary only** (z_e is detached). But OMP runs under `torch.no_grad()`, so `z_q` is built from `dictionary` via `_reconstruct_sparse` with atom indices selected under no_grad. This means the dictionary gradient from `dl_latent_loss` only flows through `_reconstruct_sparse` — the sparse code selection is frozen. This is intentional (VQ-VAE style), but it means **dictionary updates are biased** since they optimize reconstruction of codes that were selected under the old dictionary. Worth noting in comments.

### 6. Batched OMP: `G[I[batch_idx, :], index[expanded_batch_idx]]` indexing (line 451)

The fancy indexing at line 451 builds the Gram sub-columns for the Cholesky update. The `expanded_batch_idx` construction at line 450 transposes the batch index expansion. This works but is hard to verify — an off-by-one in the `.t()` would silently produce wrong Cholesky factors. Consider adding a shape assertion.

### 7. `_batch_ssim` doesn't detach gradient (line 2348)

`_batch_ssim` is called during validation within `torch.no_grad()`, so this is fine in practice, but the function signature doesn't enforce it. If called outside the no_grad context during training, it would backprop through the SSIM computation unnecessarily.

---

## Architecture / Design Issues

### 8. Massive code duplication between `DictionaryLearningTokenized` and `PatchDictionaryLearningTokenized`

The quantization methods (`_quantize_coeff`, `_dequantize_coeff`, `_pack_quantized_tokens`, `_unpack_quantized_tokens`) and all the vocab/token bookkeeping are fully duplicated between the two classes (~150 lines). Extract a base class or mixin.

### 9. Global mutable state for FID and W&B (lines 492-496)

```python
_RFID_MODEL = None
_RFID_MODEL_DEVICE = None
_RFID_METRIC = None
_RFID_METRIC_DEVICE = None
_WANDB_LOG_STEP = 0
```

Global mutable state makes the code non-reentrant and problematic for multi-process scenarios. The W&B step counter in particular means you can't have two runs in the same process.

### 10. `tokens_to_latent` in non-quantized mode requires external coefficients but no shape validation

Line 1077-1081: When `quantize_sparse_coeffs=False`, the caller must pass `coeffs` with shape `[B, H, W, D]`, but there's no check that the spatial dims of `coeffs` match `tokens`. A shape mismatch would produce a silent wrong reconstruction.

### 11. Stage-2 training doesn't save best model (line 3359)

Stage-1 saves both `ae_last.pt` and `ae_best.pt` (tracking best val reconstruction). Stage-2 only saves `transformer_last.pt` — there's no validation loss tracking or best-model checkpointing. For a 100-epoch training run, this means you always get the last checkpoint regardless of overfitting.

### 12. No learning rate schedule for stage-2 (line 3085)

Stage-1 has full cosine annealing with warmup. Stage-2 uses a bare `Adam` with constant LR for potentially 100 epochs. This is a significant gap.

---

## Minor Issues

### 13. `_stage1_lr_scale` accepts `schedule` but only implements `"cosine"` (line 2579)

```python
if str(schedule) != "cosine":
    return 1.0
```

The argparse `choices=["constant", "cosine"]` allows "constant", but there's no actual constant schedule — it just returns 1.0 as a fallback. This works but is confusingly implicit.

### 14. `Encoder.forward` attention indexing bug potential (line 693-694)

```python
if len(self.down[i_level].attn) > 0:
    h = self.down[i_level].attn[i_block](h)
```

If `attn_resolutions` triggers attention for some but not all res blocks at a level, `len(attn)` might be less than `num_res_blocks`, causing an index error at `attn[i_block]`. The RQ-VAE reference code has the same pattern, so this is inherited, but it assumes attention is either applied to all blocks at a resolution or none — worth a guard.

### 15. `val_psnr` / `val_ssim` weighted averaging is slightly off (lines 2795-2800)

You weight by `x.size(0)` (batch size), but `_batch_psnr` and `_batch_ssim` already compute a mean over the batch. So your aggregation `psnr.detach() * x.size(0)` correctly re-weights for the batch-weighted mean. This is actually fine — just noting it's easy to get wrong.

### 16. Dead argument `num_res_hiddens` (line 3530)

`args.num_res_hiddens` is parsed but never used anywhere in the code.

---

## Summary of actionable items (priority order)

| Priority | Issue | Line |
|----------|-------|------|
| **Fix now** | `ae_module.pre` doesn't exist in `analyze_patch_spectrum` | 2124 |
| **Fix now** | Dead code / unused arg `num_res_hiddens` | 3530 |
| **Fix soon** | `generate_with_coeffs` missing atom vocabulary masking | 2044-2056 |
| **Fix soon** | No stage-2 best-model checkpoint or LR schedule | 3359, 3085 |
| **Improve** | Duplicated quantization code between bottleneck classes | ~150 lines |
| **Improve** | `_encode_sparse_codes` return type inconsistency | 1309-1342 |
| **Improve** | Global mutable state for metrics/logging | 492-496 |

The most urgent fix is the `ae_module.pre` reference in `analyze_patch_spectrum` — that will crash any `--analyze_spectrum` run.
