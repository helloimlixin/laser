The Church compound prior with direct causal coefficient history is being
trained **from scratch**. Every stage-2 component starts with fresh parameters,
including the spatial body, atom head, depth stack, local coefficient conditioner,
and new history decoder. No epoch-57 weights, optimizer moments, scheduler state,
RNG state, or best-FID record are used for initialization.

The selected frozen stage-1 tokenizer and existing full-training token cache
are reused. Both the production and benchmark checkpoint directories were
verified empty before launch. An independent eight-update, five-GPU preflight
verified zero optimizer state and scheduler step zero on every rank. Its
checkpoint is stored separately and is not used by production.

Run: [church-laser-compound-history-scratch300-b2048-h200x5-20260922](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-laser-compound-history-scratch300-b2048-h200x5-20260922).
Local experiment: `outputs/church-compound-history-scratch-20260922`.

The model has 412,617,216 trainable parameters. The added 7,879,168-parameter
coefficient decoder has two width-512 layers and eight attention heads. It sees
the current atom, the backbone state, shifted completed pair embeddings, and
the current site's strictly earlier physical reconstruction. Causal attention
spans all 256 compound events. Its residual output starts at zero, while its
internal weights are random; it learns jointly with the freshly initialized
backbone. This retains the tested architecture while changing initialization
to the user's requested fresh training.

The model remains fully autoregressive over compound pairs, with atom choice
followed by its conditional signed coefficient. Coefficients retain the same
2,048 nonuniform centers in raw units with no explicit clipping or coefficient
normalization. This experiment changes context modeling rather than coefficient
quantization. The implementation's causal, cache, and parameter-update checks
are recorded in the [decoder design](church-compound-history-trial-2026-09-22.md).

Training uses all 126,227 Church training images, five H200 GPUs, exact global
batch 2,048, and 300 planned epochs. AdamW uses LR 5e-4, betas (0.9, 0.95),
weight decay 1e-4, and cosine decay from step zero over 18,600 updates, with no
warmup. Residual dropout is 0.1; attention and embedding dropout are zero.
The gradient norm limit is 1. No data are removed for validation.

Official RQ FID is computed on 50,000 generated samples every epoch against
all 126,227 training images. Reference images and stage-2 token-cache input
share RGB conversion, bilinear resize of the short side to 256, center crop
256, and normalization by 0.5. Sampling uses atom temperature 1/top-k 250 and
coefficient temperature 1/all bins; both top-p settings are 1. FID generation
uses batches of 2,048 per GPU. The first 64 rank-zero samples are logged as an
unselected preview, alongside fixed-probe training/validation diagnostics.

Full latest and best-FID checkpoints are uploaded online with remote digest
and size verification. They contain optimizer, scheduler, and all five rank
RNG states. The supervisor only resumes checkpoints belonging to this fresh
run. An atomic `train/stop-request.json` can request a stop after a full epoch
has been evaluated and saved; uploads then drain after GPU state is released.

The cancelled continuation's completed epoch-59 state and inherited epoch-57
best were preserved separately. Its two FIDs are continuation results and must
not be reported as results of this scratch run.
