# STL-10 64px LASER-VAR proof of concept

This run trains the full pipeline from random initialization: five tokenizer
epochs followed by fifty VAR epochs on the labeled STL-10 training split.

- Resolution: 64x64.
- Tokenizer downsampling: 16, producing a 4x4 final latent grid.
- VAR scales: `1,2,3,4`, or 30 spatial transformer positions.
- Sparse representation: two atom/coefficient pairs at each position, handled
  by the local causal sparse-depth head rather than extra attention positions.
- Classes: 10.
- Train/test sizes: 5,000/8,000.
- Tokenizer effective batch: 128.
- Prior effective batch: 256.
- W&B run: `stl10-var-laser-poc-20260916`.

The original one-epoch launch completed tokenizer epoch one and was stopped
during reconstruction evaluation when the budget was extended. Its epoch-one
checkpoint is reused; the run is not restarted from random weights.

## V2 quality-gated run

The five-epoch tokenizer collapsed to blurry, nearly identical reconstructions
(512-image rFID 283.4), so its prior was stopped after one update. V2 restarts
from random initialization with 50 tokenizer epochs, a 50-update warmup,
adversarial training beginning at update 200, and reconstruction evaluation
every five epochs. The 50-epoch VAR prior starts only if the tokenizer's final
512-image reconstruction FID is at most 100.
