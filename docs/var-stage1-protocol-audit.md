# Original VAR stage-1 protocol audit, 2026-09-14

**Subsequent decision:** the user authorized resuming the earlier matched
ImageNet-only scratch experiments despite the unavailable original recipe.
The existing VQ and LASER checkpoints are resumed with unchanged settings.
The stop described below is historical; it is no longer an instruction to keep
these runs paused. See the [current experiment notes](imagenet-var-scratch.md).

The ImageNet-only scratch pair has been checkpointed and stopped. Its 100-epoch
tokenizer schedule, LR 1e-4, global batch 128, and PatchGAN training choices
were locally selected. They are not established settings of the published VAR
tokenizer. Preserve the checkpoints as a custom ablation; do not represent
them as a reproduction or silently resume with changed settings.

## Confirmed original protocol

| Item | Evidence |
| --- | --- |
| Stage-1 data | Full OpenImages v4 collection; about 8 million successfully downloaded images after inaccessible URLs, per an author |
| Training structure | Train encoder, multi-scale quantizer and decoder jointly with image reconstruction, quantization, perceptual and adversarial losses; subsequently freeze the tokenizer for VAR prior training |
| Spatial factor | 16 |
| Codebook | 4096 entries shared across scales |
| Released implementation | Width 160 in the released checkpoint, latent dimension 32, ten scales 1,2,3,4,5,6,8,10,13,16, commitment weight 0.25, four shared residual convolutions |

Sources: [author dataset clarification](https://github.com/FoundationVision/VAR/issues/145#issuecomment-2719768814),
[paper, sections 3–4](https://arxiv.org/html/2404.02905v2),
[released VAE construction](https://github.com/FoundationVision/VAR/blob/78b95394fc5896192e3a003e4b295f8ea743c48f/models/__init__.py),
[quantizer](https://github.com/FoundationVision/VAR/blob/78b95394fc5896192e3a003e4b295f8ea743c48f/models/quant.py).
The final NeurIPS conference paper was also checked; it does not supply a
tokenizer optimizer/schedule table. The reported 200–350 epochs describe the
transformer, not the stage-1 tokenizer.

## What the linked toolkit does and does not establish

An [author points to vaex](https://github.com/FoundationVision/VAR/issues/63#issuecomment-2428386973)
for tokenizer training. At revision `c08db77e484ceac354e6de1883c703a9bab25230`,
its [argument defaults](https://github.com/FoundationVision/vaex/blob/c08db77e484ceac354e6de1883c703a9bab25230/utils/arg_util.py)
include 250 epochs, global batch 768, AdamW, LR 3e-4 for both optimizers, cosine
decay to 0.3 of the initial LR, warmup for 1% of training, and generator EMA
0.9999. Its losses include L1, L2, LPIPS and a DINO-based discriminator.

These are toolkit defaults, not a recovered launch configuration for
`vae_ch160v4096z32.pth`. Its default dataset string is `o_cc` (OpenImages plus
CC12M), and its [quantizer](https://github.com/FoundationVision/vaex/blob/c08db77e484ceac354e6de1883c703a9bab25230/models/quant.py)
performs single-scale lookup. The source also references an undefined
`UnlabeledImageFolders` dataset class. Copying these defaults and replacing the
quantizer would constitute a reconstruction of the recipe, not a verified
replication of the published checkpoint.

A later [author reply](https://github.com/FoundationVision/VAR/issues/170#issuecomment-3058579231)
links [BitVAE](https://github.com/FoundationVision/BitVAE), whose README explicitly
identifies it as Infinity's bitwise tokenizer with single-scale pretraining and
multi-scale fine-tuning. That is another model, not evidence that original VAR
used those commands.

## Remaining requirements

An exact stage-1 reproduction still needs the original multi-scale tokenizer's
resolved training configuration or author-confirmed optimizer settings, update
budget, loss weights, discriminator, initialization and any training stages.
The downloaded OpenImages image manifest also matters: the authors' surviving
URLs are not supplied in the sources examined. No OpenImages dataset was found
in the local data directory; the existing manifest is ImageNet.

Once the recipe is established, the intended experiment is two fresh tokenizer
runs using the same data and training protocol, changing VQ to LASER, followed
by the same VAR prior training and evaluation. Any unavoidable departure must
be documented as such. The spatially matched LASER representation still uses
more nominal code bits, which remains a separate comparison limitation.

Public-source receipts and the examined vaex source are preserved under
`outputs/imagenet-var-scratch-20260914/stage1-protocol-audit/`.
