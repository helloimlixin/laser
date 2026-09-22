The historical successful FFHQ and ImageNet K=2 stage-2 runs both froze the
stage-1 codec and introduced scalar coefficient discretization in the stage-2
auxiliary code. They provide no evidence that their success depended on
fine-tuning the decoder through those scalar bins.

The audit downloaded the actual stage-2 source uploaded by each run and queried
the original W&B configuration and summary. Evidence is retained under
`outputs/church-tokenized-bottleneck-audit-20260922`.

| Run | Reported generation result | Stage-2 representation | Codec handling |
| --- | --- | --- | --- |
| [FFHQ ffhqcmp0804205803](https://wandb.ai/helloimlixin-rutgers/laser/runs/ffhqcmp0804205803) | FID50k 8.1744 at epoch 200, against 70,000 training images | 2,048 atoms, K=2, compound pairs, 2,048 coefficient bins | Encoder, dictionary, post-quantization convolution, and decoder frozen |
| [ImageNet swgbasnb](https://wandb.ai/helloimlixin-rutgers/laser/runs/swgbasnb) | Reported FID50k 25.1518 at epoch 35 | 16,384 atoms, K=2, alternating atom/coefficient tokens, 2,048 coefficient bins | Frozen codec; scalar bins constructed in stage 2 |
| [ImageNet v8yrnory](https://wandb.ai/helloimlixin-rutgers/laser/runs/v8yrnory) | Reported FID50k 19.5370 at epoch 45 | Cached continuation of the same scalar-token lineage | Same pretrained tokenizer |
| [ImageNet v8dup0731113220](https://wandb.ai/helloimlixin-rutgers/laser/runs/v8dup0731113220) | Reported FID50k 16.3682 at epoch 100 | Continuation of that ImageNet K=2 lineage | Same pretrained tokenizer |

These are historical logged generation scores, not independently recomputed
results. The archived ImageNet evaluators use TorchMetrics against a validation
loader; they are not the current Church run's official RQ evaluator against
full training statistics. Different datasets, conditioning, sample references,
and evaluator paths prevent treating the scores as a matched experiment.

All four stage-2 sources construct the coefficient bins after loading the
tokenizer checkpoint, call `eval().requires_grad_(False)` on the auxiliary
codec, and optimize only transformer parameters. Decoding explicitly rebuilds
the latent as `sum(dictionary[atom] * coefficient_center)` before the frozen
post-quantization convolution and image decoder. The temporary RQVAE object
created with `bottleneck_type='rq'` only supplies the codec architecture: its
RQ quantizer weights are excluded and replaced by the checkpoint's LASER
dictionary. It is not evidence that the checkpoint was trained with RQ codes.

The FFHQ stage-1 run is
`ffhq-a2048-k2-rqvae-strict-20260720-145706`; the ImageNet source is
`x3h5cl0h-a16384-k2-20260719-014434`. Both archived stage-1 configurations use
`bottleneck_type: laser`, K=2 and a learned sparse dictionary, with no scalar-bin
configuration. Their fully executed stage-1 dictionary implementation was not
available among the uploaded run files, so the stage-1 interpretation rests on
the saved configurations and stage-2 loading/tokenization code, not a fresh
execution of those historical checkpoints.

FFHQ uses uniform bins over normalized [-3,3], with depth scales
36.2083333 and 8.5833333. The ImageNet scalar-token lineage uses uniform bins
over [-20,20] and scale 6.4 at both depths. Both use stochastic coefficient
targets; copying their numeric temperature into raw Church coefficients would
not reproduce the same physical perturbation.

For the current Church tokenizer the complete frozen training source is
available: `DictionaryLearning.forward` selects OMP supports and real-valued
least-squares coefficients, reconstructs their sum, and uses a straight-through
latent for the decoder. Its training path does not apply the 2,048 scalar
centers subsequently fitted for the compound prior. This is a real difference
between the stage-1 and stage-2 bottlenecks. Original RQ-VAE instead decodes its
quantized bottleneck in the stage-1 forward pass, as shown in the
[released implementation](https://github.com/kakaobrain/rq-vae-transformer/blob/341395e562ac347f5eb62db9f5f08b9f2cc42a60/rqvae/models/rqvae/rqvae.py).

Existing Church measurements nevertheless show small deterministic rounding
error: the current physical bins add latent MSE 1.8551e-7 over 300 validation
images and 16 stochastic support variants. An eight-image decoder check found
pixel MSE 2.8836e-7 relative to continuous-coefficient decoding. These values
measure nearest-bin quantization, not draws from the soft coefficient targets
and not generated FID.

The next useful comparison is continuous reconstruction, nearest-bin
reconstruction, and reconstruction using the exact stochastic support and
coefficient-token distributions used by stage 2. Evaluate all three on the
same images with matched reconstruction FID and perceptual error. If the
stochastic token stream materially degrades reconstruction, a decoder-only
fine-tune through those exact tokens isolates decoder adaptation while leaving
token meanings and the current prior compatible. Keep raw coefficient units
and no explicit coefficient clipping.

If that cannot recover fidelity, fine-tune the encoder, dictionary, and decoder
with the actual discrete forward bottleneck and a suitable straight-through
gradient. Changes to the encoder, dictionary, or coefficient centers require
regenerating the token cache and training a fresh stage-2 prior for a clean
comparison. A scalar-bin-aware stage-1 update does not by itself address prior
atom overfitting or guarantee lower generated FID. No new training was launched
and the current Church run was left running.
