# Debugging the Stage-1 codec mismatch: maintained `src` vs. third-party RQ-VAE

Date: 2026-08-11

## Executive conclusion

The aborted launch used `train.py stage1`, which instantiated the maintained
`src.models.laser.LASER` wrapper. The intended reference run used
`third_party/rq-vae-transformer/main_stage1.py` and the upstream RQ-VAE wrapper.
Those are not interchangeable training paths, even when their Hydra values look
similar.

The most important finding is slightly counter-intuitive:

- `src/models/rqvae/modules.py` and
  `third_party/rq-vae-transformer/rqvae/models/rqvae/modules.py` are currently
  byte-for-byte identical. The core `Encoder` and `Decoder` selected by
  `backbone=rqvae` in `src.models.laser` therefore have the same block math as
  the third-party copy.
- The aborted model was still architecturally different because the maintained
  wrapper skipped the learned `quant_conv` and `post_quant_conv` layers that the
  third-party RQ-VAE always creates.
- If the comparison was instead made against `src/models/encoder.py` and
  `src/models/decoder.py`, those are not the DDPM classes used by
  `src.models.laser` for `backbone=rqvae`. They are a second implementation and
  comparing or editing them does not change that path.

So the mistake was not only "the wrong Encoder class." It was selecting the
wrong top-level codec/trainer contract and assuming matching configuration
labels made it equivalent to the third-party RQ-VAE.

## What the aborted job actually instantiated

The archived Hydra config for the stopped run is:

`outputs/imagenet-a16384-k4-compound-official-20260811-172655/stage1/2026-08-11_17-27-58/.hydra/config.yaml`

It selected `backbone: rqvae`, `backbone_latent_channels: 256`, and
`embedding_dim: 256`.

The import boundary is visible at `src/models/laser.py:28-31`:

```python
from .rqvae.modules import Decoder as DDPMDecoder
from .decoder import SimpleDecoder
from .rqvae.modules import Encoder as DDPMEncoder
from .encoder import SimpleEncoder
```

For `backbone == "rqvae"`, construction proceeds through
`src/models/laser.py:580-620`. Therefore:

- the active encoder is `src.models.rqvae.modules.Encoder`;
- the active decoder is `src.models.rqvae.modules.Decoder`;
- `src.models.encoder.Encoder` and `src.models.decoder.Decoder` are not used;
- only `SimpleEncoder` and `SimpleDecoder` are imported from those latter files,
  and they belong to the non-RQ-VAE fallback branch.

This is the first debugging trap: looking at `src/models/encoder.py` or
`src/models/decoder.py` suggests changes that are absent from the actual model.

## Exact implementation comparison

### 1. The selected `src` DDPM modules are currently identical to third party

The following check returns success:

```bash
cmp -s \
  src/models/rqvae/modules.py \
  third_party/rq-vae-transformer/rqvae/models/rqvae/modules.py
```

Both implementations have:

- the same encoder input convolution and width schedule;
- `num_res_blocks` encoder blocks per resolution;
- attention at every configured resolution;
- an unconditional middle attention block;
- `num_res_blocks + 1` decoder blocks per resolution;
- nearest-neighbor upsampling followed by a 3x3 convolution;
- the same normalization and output convolution.

Relevant third-party lines are
`third_party/rq-vae-transformer/rqvae/models/rqvae/modules.py:10-98` for the
encoder and `:101-202` for the decoder. The corresponding `src` lines are the
same.

Implication: copying these classes into `src` again will not fix the mismatch.
Source provenance still matters for preventing later drift, which is why the
corrected launch resolves both classes directly from the third-party tree.

### 2. The maintained wrapper silently removed two learned projections

This is the concrete architecture mismatch in the aborted run.

At `src/models/laser.py:596-617`, the maintained wrapper does this:

```python
if self.backbone_latent_channels == int(embedding_dim) and not self.force_quant_conv:
    self.pre_bottleneck = nn.Identity()
    self.post_bottleneck = nn.Identity()
else:
    self.pre_bottleneck = nn.Conv2d(..., kernel_size=1)
    self.post_bottleneck = nn.Conv2d(..., kernel_size=1)
```

Because the aborted job set both channel counts to 256 and did not set
`force_quant_conv=true`, it took the `Identity` branch.

The third-party RQ-VAE unconditionally creates both learned projections at
`third_party/rq-vae-transformer/rqvae/models/rqvae/rqvae.py:125-126`:

```python
self.quant_conv = nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
```

For 256 channels, this removes 131,584 trainable parameters from the maintained
model:

```text
2 * (256 * 256 weights + 256 biases) = 131,584
```

More importantly, these are not cosmetic parameters. `quant_conv` learns the
coordinate system presented to OMP, and `post_quant_conv` learns how sparse
reconstructions are mapped back into the decoder's latent space. The reference
checkpoint contains all four tensors:

```text
quant_conv.weight       (256, 256, 1, 1)
quant_conv.bias         (256,)
post_quant_conv.weight  (256, 256, 1, 1)
post_quant_conv.bias    (256,)
```

The maintained identity shortcut therefore cannot produce a checkpoint with
the same state-dict contract as the reference.

### 3. The generic `src` encoder/decoder are a divergent second implementation

If code elsewhere imports `src.models.encoder.Encoder` or
`src.models.decoder.Decoder`, there are real differences from third party:

- `src/models/encoder.py:8-74` reimplements the upstream layers locally. Its
  `ResnetBlock` has no activation-checkpointing path. Third party defines
  `self.checkpointing` and calls `torch.utils.checkpoint.checkpoint` at
  `third_party/rq-vae-transformer/rqvae/models/rqvae/layers.py:68` and
  `:122-126`.
- `src/models/encoder.py:203-304` makes middle attention optional through
  `use_mid_attention`. Third party always installs and executes the middle
  attention block at `modules.py:53-63` and `:89-92`.
- `src/models/decoder.py:101-136` changes decoder depth to
  `num_res_blocks + extra_res_blocks`. Third party fixes it at
  `num_res_blocks + 1` at `modules.py:147` and `:188`.

For the intended recipe, `num_res_blocks=2`, so third party uses three decoder
blocks per resolution. A generic `src.models.decoder.Decoder` instantiated with
`extra_res_blocks=0` would use only two and would be checkpoint-incompatible.

The aborted `src.models.laser` path did not actually use these generic DDPM
classes, but their presence makes imports and debugging ambiguous. Keeping two
near-duplicate implementations is itself a maintenance hazard.

### 4. Some knobs exposed by the maintained wrapper do nothing on its active path

`src/models/laser.py:581-594` passes `use_mid_attention` into the DDPM kwargs.
The active `src.models.rqvae.modules.Encoder` and `Decoder` accept arbitrary
extra kwargs and ignore this value. Middle attention is always enabled.

Likewise, the maintained config exposes `decoder_extra_residual_layers`, but it
is not passed under a name consumed by the active third-party-style module and
that module fixes decoder depth at `num_res_blocks + 1` anyway.

These knobs make the configuration appear more precise than the instantiated
model. Tests should assert the number and types of modules, not just serialized
config values.

### 5. The ImageNet preprocessing paths are different

The reference third-party transform is at
`third_party/rq-vae-transformer/rqvae/img_datasets/transforms.py:18-33`:

```text
train: Resize(256) -> RandomCrop(256) -> RandomHorizontalFlip
val:   Resize(256) -> CenterCrop(256) -> Resize((256, 256))
```

The maintained `ImageFolderDataModule` builds `Resize(self._resize_to())` at
`src/data/image_folder.py:113` and does not add a crop when
`train_crop_size == image_size`. With `_resize_to()` represented as `(256,
256)`, this directly warps each image to a square. Its validation transform at
`:122-129` also lacks the reference center crop.

This changes the training distribution and validation rFID inputs even if all
model weights and hyperparameters match.

## What was not wrong

Several settings from the aborted launch did match the reference intent:

- widths `[1, 1, 2, 2, 4, 4]`, base width 128, and 8x8 latent resolution;
- 256-dimensional sparse latents;
- a 16,384-atom dictionary and requested `k=4` extension;
- MSE reconstruction, LPIPS weight 1.0, GAN weight 0.75;
- Adam betas `(0.5, 0.9)`, float32, and effective batch 128;
- the dictionary and commitment objective split. The reference's logged
  `loss_latent` is `loss_dictionary + loss_commitment`, then weighted by 0.25;
  the maintained launch's two 0.25 weights are algebraically equivalent.

The LASER dictionary state also matches the reference checkpoint contract:
`quantizer.dictionary` is `(256, 16384)`, and the revival/data-initialization
buffers have identical names and shapes. This is why the corrected integration
only replaces the RQ quantizer while preserving the third-party codec.

## Required fixes

### Preferred fix: use the third-party codec as the source of truth

This is the path now used by the corrected pipeline:

1. Launch `third_party/rq-vae-transformer/main_stage1.py`.
2. Instantiate `rqvae.models.rqvae.rqvae.RQVAE`.
3. Instantiate `Encoder` and `Decoder` from
   `third_party/rq-vae-transformer/rqvae/models/rqvae/modules.py`.
4. Keep `quant_conv` and `post_quant_conv` unconditionally.
5. Exchange only the RQ quantizer for the LASER dictionary learner.
6. Use the third-party ImageNet transforms and trainer loss contract.

### If the maintained `src` path must be repaired later

At minimum:

1. Remove the identity shortcut at `src/models/laser.py:596-617`, or make
   `force_quant_conv=true` mandatory for all RQ-VAE-compatible runs.
2. Stop maintaining a second DDPM implementation in `src/models/encoder.py` and
   `src/models/decoder.py`; import the third-party classes explicitly or add
   parity tests that fail on any state-dict/forward drift.
3. Make ImageNet preprocessing reproduce the third-party single-side resize
   plus crop semantics.
4. Remove or reject ignored config knobs such as `use_mid_attention` and
   `decoder_extra_residual_layers` on the upstream-compatible path.
5. Add a strict reference-checkpoint test. It should load every encoder,
   decoder, projection, and dictionary key with `strict=True`.
6. Add a runtime source assertion:

   ```python
   assert "third_party/rq-vae-transformer" in inspect.getfile(type(model.encoder))
   assert "third_party/rq-vae-transformer" in inspect.getfile(type(model.decoder))
   ```

## Verification performed for the corrected path

The corrected adapter was instantiated with the reference `k=2` dimensions and
strict-loaded from
`outputs/imagenet_x3h5cl0h_stage2/stage1_checkpoint/best_rfid_slot3_model.pt`.
The result was:

```text
encoder_source .../third_party/rq-vae-transformer/rqvae/models/rqvae/modules.py
decoder_source .../third_party/rq-vae-transformer/rqvae/models/rqvae/modules.py
dictionary_shape (256, 16384)
reference_checkpoint_strict_load PASS
```

This checks more than class names: strict loading proves the full encoder,
decoder, quant/post-quant projections, and LASER dictionary state are compatible
with the Stage-1 reference checkpoint.

The corrected entry point now also enforces provenance at runtime at
`third_party/rq-vae-transformer/main_stage1.py:67-81`. It resolves the source
file for the instantiated encoder and decoder with `inspect.getfile`, compares
both paths against the vendored `rqvae/models/rqvae/modules.py`, and aborts
before moving the model to CUDA if either class came from another tree.

A four-GPU smoke run then completed two training batches, two validation
batches, and an atomic checkpoint save. The saved state contains the
discriminator and both learned projection layers, and
`quantizer.dictionary.shape == (256, 16384)`. A separate out-of-order metric
test verified that the rolling rFID policy retains exactly the three lowest
values while replacing fixed slot names.

## Checkpoint retention and online replacement

Stage 1 performs rFID validation and checkpoint selection every epoch. The
implementation at
`third_party/rq-vae-transformer/rqvae/trainers/trainer.py:144-226` atomically
replaces `last_model.pt`, ranks candidates in ascending rFID order, and keeps
only:

```text
best_rfid_slot1_model.pt
best_rfid_slot2_model.pt
best_rfid_slot3_model.pt
```

The writer uploads those same fixed paths with W&B's immediate run-file policy
at `third_party/rq-vae-transformer/rqvae/utils/writer.py:80-92`. Consequently,
each save replaces the online `last` and current best-three files instead of
creating an unlimited series of immutable checkpoint artifacts.

Stage 2 uses the analogous fixed names `last.pt`, `best-fid-01.pt`,
`best-fid-02.pt`, and `best-fid-03.pt`; its ranking metric is generative FID,
not reconstruction FID.

## Production verification

The corrected production run reports both source checks in its live log:

```text
verified third-party encoder source: .../third_party/rq-vae-transformer/rqvae/models/rqvae/modules.py
verified third-party decoder source: .../third_party/rq-vae-transformer/rqvae/models/rqvae/modules.py
```

Its online W&B config identifies the program as
`third_party/rq-vae-transformer/main_stage1.py`, the bottleneck as `laser`, the
dictionary size as 16,384, and the requested code shape as `[8, 8, 4]`.
