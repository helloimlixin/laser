FID50k for the three-epoch LASER tokenizer / stage-2 epoch-50 checkpoint is **11.849154** against a freshly rebuilt reference using the exact stage-1 fine-tuning transform. The same generated samples score **11.535297** against the downloaded official reference.

| Fixed generated sample set | Exact fine-tuning-transform reference | Downloaded official reference |
|---|---:|---:|
| LASER, three-epoch tokenizer, stage-2 epoch 50 | 11.849154 | 11.535297 |
| LASER, one-epoch tokenizer, stage-2 epoch 50 | 14.268734 | 13.884514 |
| Released Church RQ-VAE / RQ-Transformer pair | 8.049792 | 7.670966 |

Every row uses 50,000 generated images. The generated feature statistics were held fixed so the comparison changes only the real reference. These are existing epoch-50 sample sets, not samples from the current later training step.

The new reference was built from all **126,227 unique Church training LMDB entries**, once each. The saved fine-tuning configuration and its historical transform implementation were loaded directly. The transformation is PIL RGB → bilinear resize with shorter side 256 → center crop 256×256 → tensor → normalize to [-1,1]. The released evaluator converts back to [0,1] before Inception, exactly as it does for original images in reconstruction evaluation. No distributed-sampler padding or sample dropping was used. Dataset-key count and uniqueness were verified separately.

Both real and generated images use the same released FID Inception with its bilinear 256×256 → 299×299 resize (`align_corners=False`). This experiment matches the real training-image preprocessing; it does not disable the common Inception resize. Feature extraction, mean/covariance computation, and Frechet distance use the released repository functions. A forward hook saves the real features and progress without changing network outputs.

The rebuilt and official real distributions have FID **0.0624493** between their statistics. Their maximum mean-coordinate difference is 0.0301979. This is a measurable reference mismatch: it raises the current LASER score by **0.313857**, the previous LASER score by 0.384220, and the released-model score by 0.378826. This check does not identify which historical dataset, decoding, or preprocessing detail accounts for the mismatch, and it does not establish that bilinear resizing caused it.

The rebuilt reference is saved separately. Existing W&B metric history and the active training evaluator's official-reference configuration were not rewritten by this recomputation.

Artifacts: [results](/workspace/Projects/laser/outputs/church-stage1-transform-fid-20260914/result.json), [exact configuration and source provenance](/workspace/Projects/laser/outputs/church-stage1-transform-fid-20260914/specification.json), [rebuilt reference](/workspace/Projects/laser/outputs/church-stage1-transform-fid-20260914/lsun-church-stage1-transform-full.npz), [real feature matrix](/workspace/Projects/laser/outputs/church-stage1-transform-fid-20260914/real-features.npy), [recomputation script](/workspace/Projects/laser/scripts/tools/recompute_church_fid_stage1_reference.py).
