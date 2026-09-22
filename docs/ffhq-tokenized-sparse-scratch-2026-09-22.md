# FFHQ scratch pipeline with shared sparse tokens

Run: `ffhq256-var341-tokenized-scratch-20260922`.

The previous VAR tokenizer already decoded discretized coefficients. Its stage-1 training used deterministic OMP and coefficient rounding, whereas the compound prior trained on stochastic atom/coefficient trajectories. This experiment trains the stage-1 decoder on those stochastic hard-token trajectories too. It retains OMP support extraction; replacing OMP with a learned token predictor would be a separate experiment.

`src/models/sparse_token_codec.py` now implements the common trajectory codec. Stage 1 selects integer atom/coefficient IDs, reconstructs the latent with the shared dictionary and coefficient grid, and feeds that hard-token latent to the image decoder. Each later scale is computed from the actually sampled preceding scales. A straight-through gradient trains the image encoder, and the bottleneck loss trains the dictionary/residual convolutions. Unquantized least-squares coefficients serve only token selection and soft supervision; they do not bypass the hard-token decoder input.

The `compound-hard-token-v1` policy uses per-scale physical squared-distance temperatures proportional to the learned coefficient range squared: atom ratio 0.0001 and coefficient ratio 0.000025. Stage-1 ranges track training observations; evaluation freezes them. The selected tokenizer freezes this same policy for stage-2 cache construction. The calibration phase checks reconstruction quality without changing its temperatures. Checkpoint metadata, cache construction and prior loading reject a policy mismatch. Deterministic reconstructions remain reproducible diagnostics.

Both trainable stages start randomly, with no trained checkpoint transfer from earlier experiments. The six-update preflight is also separate from production initialization. Only the prepared FFHQ images, verified FID reference, architecture and experiment settings are reused. The frozen LPIPS loss and Inception evaluator retain their standard pretrained feature weights.

- Stage 1: 50 epochs, 64 images per GPU, three H200 GPUs; validation reconstruction FID every epoch; last and best reconstruction-FID tokenizer checkpoints uploaded online.
- Stage 2: 100 epochs, 128 images per GPU; fresh 16-trajectory token cache for both horizontal views; scale loss weights `[1, 8, 4, 2, 1]`; unconditional sampling with atom temperature 0.6 and coefficient temperature 1.0.
- Stage-2 checkpoint selection: 50,000 generated images against the exact 60,000 cached training images at epoch 1, every 10 epochs, and the final epoch. The best checkpoint receives a final evaluation. Preview grids contain 64 images in 8 columns.
- All benchmark generation uses FP32 image decoding and continuous pixels without PNG rounding, the verified Inception implementation, fixed seeds, and reference checksums. No older score using a different real reference is treated as this run's benchmark.
- Each stage has a separate W&B run. The supervisor advances through preflight, codec verification, tokenizer training, quality audit, token cache, prior preflight, prior training, and final evaluation; it verifies committed last/best artifacts before completing.

Reference SHA-256: `e2eb3396c6a9b7c80d61422580524260cddc802017c195d00f3ac93c29c053ef`.

Source snapshots, launch receipt, configuration, reference verification, logs and phase status live under `outputs/ffhq256-var341-tokenized-scratch-20260922` (backed by `/tmp/laser-var-runs`). Full checkpoints live under `/tmp/laser-var-checkpoints/ffhq256-var341-tokenized-scratch-20260922` and are uploaded to W&B.

Validation before launch: 44 tests passed, covering hard-token decoder equality, encoder/dictionary gradients, stochastic stage-1/stage-2 parity, cache reconstruction, policy rejection, existing compound prediction, resume and evaluation contracts, FFHQ data and supervisor behavior. The GPU preflight additionally checks optimizer/RNG state on all three ranks and exact equality of the actual decoder input and pixels after a token round trip.

Stage 1: https://wandb.ai/helloimlixin-rutgers/laser/runs/ffhq256-var341-tokenized-scratch-20260922-stage1

Stage 2: https://wandb.ai/helloimlixin-rutgers/laser/runs/ffhq256-var341-tokenized-scratch-20260922-stage2
