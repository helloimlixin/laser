# Church integer coefficient-resolution screen

The current integer prior predicts one of two fitted coefficient values for
each of 16,384 atoms, plus a zero token. Both values are in raw latent units.
This screen measures what additional coefficient choices can recover with the
same dictionary, encoder, frozen decoder, and four residual depths.

All candidates reconstruct the same 128 evenly spaced Church training images
used by the previous distortion audit. Their original pixels use the same
resize/center-crop transform as the training latent cache and FID reference.
The screen ran in FP32 on CPU while the matched prior experiment occupied the
GPUs. The two-level control reproduced the previous GPU pixel MSE exactly.

| Levels per atom | Vocabulary | Latent MSE | Pixel MSE in [0,1] | Pixel error change |
|---:|---:|---:|---:|---:|
| 2, current | 32,769 | 0.01779817 | 0.01398435 | baseline |
| 4 | 65,537 | 0.01360313 | 0.01359424 | −2.79% |
| 8 | 131,073 | 0.01268248 | 0.01346631 | −3.70% |
| 16 | 262,145 | 0.01182058 | 0.01341270 | −4.09% |

The additional levels are unfitted nested scalar grids around each atom's
existing signed values. Four levels retain the old values and add half their
magnitudes; the coefficient range is unchanged. Eight and sixteen also extend
the available range. There is no coefficient clipping or normalization.
The codebooks are candidate artifacts, not replacements for the running model.

Four levels give the largest initial gain per added vocabulary entry. A larger
vocabulary also increases classifier size, target memory, and class sparsity;
it is not automatically easier for the prior to learn. The 65,537-entry book
already exceeds uint16 capacity; diagnostic codes are saved as uint32.

These are paired reconstruction errors, not FID or a forecast of generated
quality. No claim about reaching the released RQ Church FID follows from this
screen. A useful next representation experiment is a fitted four-level book,
followed by matched reconstruction FID and a measured training-throughput
check before a full stage-2 run. The current temperature A/B trial remains
unchanged and separately addresses generalization.

Files: `outputs/church-coefficient-resolution-probe-20260922/`, including
`probe.py`, `results.json`, candidate codebooks, and uint32 diagnostic codes.
