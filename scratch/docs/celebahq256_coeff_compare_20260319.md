# CelebA-HQ 256 Coefficient Compare

Date: 2026-03-19

This note summarizes the completed 8-run coefficient-mode comparison under:

```text
/scratch/xl598/runs/celebahq256_patch_coeff_compare
```

All runs used the same patch recipe with:

- `image_size=256`
- `patch_size=4`
- `patch_stride=2`
- `stage1_epochs=10`
- `stage2_epochs=10`
- `batch_size=6`
- `stage2_batch_size=6`

The only intended sweep axes were:

- coefficient mode: real-valued (`rv`) vs quantized (`qb512_c4p0`)
- `num_atoms`: `4096`, `6144`
- `sparsity_level`: `16`, `24`

The earlier `celebahq256_patch_tuned_sweep` jobs were cancelled around `2026-03-19 05:49` and are not part of this comparison.

## Regenerated Summary

Generate the same table with:

```bash
./scripts/summarize_coeff_compare.py /scratch/xl598/runs/celebahq256_patch_coeff_compare
```

The repo launcher that now reflects the balanced default from this sweep is:

```bash
./scripts/patch_celebahq256_best.sh
```

## Results

| run | mode | atoms | k | stage1 val_loss | stage1 psnr | stage1 ssim | stage2 epoch_loss | coeff term | coeff raw |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| celebahq256_cmp10_qb512_c4p0_p4s2_a4096_k16 | qb512_c4p0 | 4096 | 16 | 0.0205 | 25.493 | 0.7567 | 4.8576 | 2.7814 | 2.7814 |
| celebahq256_cmp10_qb512_c4p0_p4s2_a4096_k24 | qb512_c4p0 | 4096 | 24 | 0.0171 | 26.262 | 0.7715 | 4.5779 | 2.5510 | 2.5510 |
| celebahq256_cmp10_qb512_c4p0_p4s2_a6144_k16 | qb512_c4p0 | 6144 | 16 | 0.0193 | 25.725 | 0.7624 | 4.8954 | 2.7505 | 2.7505 |
| celebahq256_cmp10_qb512_c4p0_p4s2_a6144_k24 | qb512_c4p0 | 6144 | 24 | 0.0150 | 26.606 | 0.7809 | 4.8179 | 2.5355 | 2.5355 |
| celebahq256_cmp10_rv_p4s2_a4096_k16 | rv | 4096 | 16 | 0.0368 | 23.908 | 0.7135 | 5.6761 | 0.1271 | 1.2707 |
| celebahq256_cmp10_rv_p4s2_a4096_k24 | rv | 4096 | 24 | 0.0239 | 25.322 | 0.7487 | 5.6255 | 0.6690 | 6.6900 |
| celebahq256_cmp10_rv_p4s2_a6144_k16 | rv | 6144 | 16 | 0.0310 | 24.332 | 0.7240 | 5.7835 | 0.2884 | 2.8843 |
| celebahq256_cmp10_rv_p4s2_a6144_k24 | rv | 6144 | 24 | 0.0202 | 25.784 | 0.7576 | 7.9556 | 5.1074 | 51.0744 |

`coeff term` means:

- quantized: `stage2/weighted_coeff_ce_loss`
- real-valued: `stage2/weighted_coeff_loss`

`coeff raw` means:

- quantized: `stage2/coeff_ce_loss`
- real-valued: `stage2/coeff_mse_loss`

## Takeaways

- Quantized beats real-valued on stage-1 `val_loss` in all 4 matched atoms/sparsity pairs.
- Increasing sparsity from `16` to `24` helped in both modes.
- The best stage-1 reconstruction run was `celebahq256_cmp10_qb512_c4p0_p4s2_a6144_k24` with `val_loss=0.0150`, `psnr=26.606`, `ssim=0.7809`.
- The lowest stage-2 `epoch_loss` in this sweep was `celebahq256_cmp10_qb512_c4p0_p4s2_a4096_k24` at `4.5779`.
- `rv a6144 k24` looks unstable for stage 2: `weighted_coeff_loss=5.1074` and `coeff_mse_loss=51.0744`.
- The promoted default launcher uses `a4096 k24` as the balanced choice; use `NUM_ATOMS=6144` with that launcher when stage-1 reconstruction quality matters more than stage-2 loss or compute.

## Caveat

Stage-2 losses are only approximately comparable across modes. Quantized runs use coefficient cross-entropy, while real-valued runs use a weighted coefficient regression term, so the cleanest cross-mode signal here is the consistent stage-1 gap plus the real-valued coefficient blow-up at `a6144 k24`.
