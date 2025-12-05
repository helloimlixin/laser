"""
Autoregressive Transformer for LASER Pattern Generation.

This module implements a GPT-style transformer that learns to generate
pattern indices autoregressively. Each image is represented as a sequence
of 16 pattern tokens (for 128x128 images with patch_size=8).

Architecture:
- Token embedding: pattern_vocab_size -> d_model
- Positional embedding: learnable, max_seq_len positions
- Transformer decoder: N layers of causal self-attention + FFN
- Output head: d_model -> pattern_vocab_size

Training:
- Input: pattern indices [B, seq_len]
- Target: shifted pattern indices (next token prediction)
- Loss: cross-entropy

Generation:
- Start with [BOS] token or unconditional
- Sample/argmax next token
- Repeat until [EOS] or max_len
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
import torchvision
from typing import Optional, Tuple
import wandb


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with optional flash attention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        # Q, K, V projections
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask (lower triangular)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D]
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale  # [B, H, T, T]

        # Apply causal mask
        attn = attn.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # Apply attention to values
        out = attn @ v  # [B, H, T, D]
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # [B, T, C]

        return self.resid_dropout(self.out_proj(out))


class TransformerBlock(nn.Module):
    """Transformer decoder block with pre-norm."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, max_seq_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class ARTransformer(pl.LightningModule):
    """
    Autoregressive Transformer for pattern sequence generation.

    Takes pattern indices from LASER model and learns to generate them
    autoregressively for unconditional image generation.
    """

    def __init__(
        self,
        vocab_size: int = 2048,
        seq_len: int = 16,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_steps: int = 100000,
        use_bos: bool = True,
        use_eos: bool = False,
    ):
        """
        Args:
            vocab_size: Number of pattern tokens (should match num_patterns in LASER)
            seq_len: Sequence length (number of patches per image, e.g., 16 for 4x4 grid)
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ff: Feed-forward hidden dimension
            dropout: Dropout rate
            learning_rate: Peak learning rate
            weight_decay: Weight decay for AdamW
            warmup_steps: Linear warmup steps
            max_steps: Total training steps for cosine decay
            use_bos: Add BOS token at start of sequence
            use_eos: Add EOS token at end of sequence
        """
        super().__init__()
        self.save_hyperparameters()

        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.use_bos = use_bos
        self.use_eos = use_eos

        # Special tokens
        self.bos_token = vocab_size if use_bos else None
        self.eos_token = vocab_size + 1 if use_eos else (vocab_size if use_bos else None)
        self.total_vocab = vocab_size + int(use_bos) + int(use_eos)
        self.total_seq_len = seq_len + int(use_bos) + int(use_eos)

        # Embeddings
        self.token_embed = nn.Embedding(self.total_vocab, d_model)
        self.pos_embed = nn.Embedding(self.total_seq_len, d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, self.total_seq_len)
            for _ in range(n_layers)
        ])

        # Output
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, self.total_vocab, bias=False)

        # Weight tying (optional but common)
        self.token_embed.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Learning rate schedule params
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

        # LASER model reference for visualization (set externally)
        self._laser_model = None
        self._log_images_every_n_epochs = 1
        self._num_samples_to_generate = 8

    def set_laser_model(self, laser_model, log_images_every_n_epochs: int = 1, num_samples: int = 8):
        """
        Set LASER model reference for generation visualization.

        Args:
            laser_model: Pretrained LASER model for decoding patterns to images
            log_images_every_n_epochs: Log generated images every N epochs
            num_samples: Number of samples to generate for visualization
        """
        self._laser_model = laser_model
        self._laser_model.eval()
        for p in self._laser_model.parameters():
            p.requires_grad = False
        self._log_images_every_n_epochs = log_images_every_n_epochs
        self._num_samples_to_generate = num_samples

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            idx: Token indices [B, T]

        Returns:
            logits: [B, T, vocab_size]
        """
        B, T = idx.shape
        device = idx.device

        # Token + position embeddings
        tok_emb = self.token_embed(idx)  # [B, T, d_model]
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)  # [1, T]
        pos_emb = self.pos_embed(pos)  # [1, T, d_model]
        x = tok_emb + pos_emb

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Output projection
        x = self.ln_f(x)
        logits = self.lm_head(x)  # [B, T, vocab_size]

        return logits

    def prepare_batch(self, pattern_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare input and target from pattern indices.

        Args:
            pattern_indices: [B, seq_len] pattern indices from LASER

        Returns:
            input_ids: [B, seq_len] or [B, seq_len+1] if use_bos
            target_ids: [B, seq_len] or [B, seq_len+1] if use_eos
        """
        B = pattern_indices.shape[0]
        device = pattern_indices.device

        if self.use_bos:
            # Prepend BOS token
            bos = torch.full((B, 1), self.bos_token, dtype=torch.long, device=device)
            input_ids = torch.cat([bos, pattern_indices], dim=1)  # [B, seq_len+1]
        else:
            input_ids = pattern_indices

        if self.use_eos:
            # Append EOS token to targets
            eos = torch.full((B, 1), self.eos_token, dtype=torch.long, device=device)
            target_ids = torch.cat([pattern_indices, eos], dim=1)  # [B, seq_len+1]
        else:
            target_ids = pattern_indices

        # For next-token prediction: input[:-1] predicts target[1:]
        # But we handle this in the loss computation
        return input_ids, target_ids

    def compute_loss(self, pattern_indices: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute cross-entropy loss for next-token prediction.

        Args:
            pattern_indices: [B, seq_len] pattern indices

        Returns:
            loss: scalar loss
            metrics: dict with additional metrics
        """
        input_ids, target_ids = self.prepare_batch(pattern_indices)

        # Forward pass
        logits = self(input_ids)  # [B, T, vocab_size]

        # Shift for next-token prediction
        # Input: [BOS, t1, t2, ..., t_n]
        # Target: [t1, t2, ..., t_n, (EOS)]
        shift_logits = logits[:, :-1, :].contiguous()  # [B, T-1, V]
        shift_targets = target_ids[:, :shift_logits.shape[1]].contiguous()  # [B, T-1]

        # Cross-entropy loss
        loss = F.cross_entropy(
            shift_logits.view(-1, self.total_vocab),
            shift_targets.view(-1),
            ignore_index=-100,
        )

        # Compute accuracy
        with torch.no_grad():
            preds = shift_logits.argmax(dim=-1)
            acc = (preds == shift_targets).float().mean()

            # Perplexity
            ppl = torch.exp(loss)

        metrics = {
            'loss': loss,
            'accuracy': acc,
            'perplexity': ppl,
        }

        return loss, metrics

    def training_step(self, batch, batch_idx):
        pattern_indices = batch
        loss, metrics = self.compute_loss(pattern_indices)

        # Log metrics
        self.log('train/loss', metrics['loss'], prog_bar=True, sync_dist=True)
        self.log('train/accuracy', metrics['accuracy'], prog_bar=True, sync_dist=True)
        self.log('train/perplexity', metrics['perplexity'], sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        pattern_indices = batch
        loss, metrics = self.compute_loss(pattern_indices)

        # Log metrics
        self.log('val/loss', metrics['loss'], prog_bar=True, sync_dist=True)
        self.log('val/accuracy', metrics['accuracy'], prog_bar=True, sync_dist=True)
        self.log('val/perplexity', metrics['perplexity'], sync_dist=True)

        return loss

    @torch.no_grad()
    def generate(
        self,
        batch_size: int = 1,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Generate pattern sequences autoregressively.

        Args:
            batch_size: Number of sequences to generate
            temperature: Sampling temperature (1.0 = normal, <1 = sharper, >1 = flatter)
            top_k: Top-k sampling (None = disabled)
            top_p: Nucleus sampling threshold (None = disabled)
            device: Device for generation

        Returns:
            pattern_indices: [batch_size, seq_len] generated pattern indices
        """
        self.eval()
        device = device or next(self.parameters()).device

        # Start with BOS token if used
        if self.use_bos:
            idx = torch.full((batch_size, 1), self.bos_token, dtype=torch.long, device=device)
        else:
            # Start with random first token
            idx = torch.randint(0, self.vocab_size, (batch_size, 1), device=device)

        # Generate tokens autoregressively
        for _ in range(self.seq_len - (0 if self.use_bos else 1)):
            # Get logits for last position
            logits = self(idx)[:, -1, :]  # [B, vocab_size]

            # Apply temperature
            logits = logits / temperature

            # Mask special tokens during generation (only predict pattern tokens)
            if self.use_bos:
                logits[:, self.bos_token] = float('-inf')
            if self.use_eos:
                logits[:, self.eos_token] = float('-inf')

            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Top-p (nucleus) sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]

            # Append to sequence
            idx = torch.cat([idx, next_token], dim=1)

        # Remove BOS token if present
        if self.use_bos:
            idx = idx[:, 1:]

        return idx[:, :self.seq_len]

    def configure_optimizers(self):
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        for name, param in self.named_parameters():
            if 'ln' in name or 'bias' in name or 'embed' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': self.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ], lr=self.learning_rate, betas=(0.9, 0.95))

        # Cosine schedule with warmup
        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / max(1, self.warmup_steps)
            progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
            return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            }
        }

    @torch.no_grad()
    def decode_patterns_to_images(self, pattern_indices: torch.Tensor) -> torch.Tensor:
        """
        Decode pattern indices to images using LASER model.

        Args:
            pattern_indices: [B, seq_len] pattern indices

        Returns:
            images: [B, C, H, W] reconstructed images in [0, 1] range
        """
        if self._laser_model is None:
            raise ValueError("LASER model not set. Call set_laser_model() first.")

        device = pattern_indices.device
        batch_size = pattern_indices.shape[0]
        laser = self._laser_model.to(device)
        laser.eval()

        # Get LASER model parameters
        bottleneck = laser.bottleneck
        patch_size = bottleneck.patch_size  # tuple (patch_h, patch_w)
        patch_h, patch_w = patch_size
        num_patches = pattern_indices.shape[1]
        channels = bottleneck.embedding_dim
        patch_flatten_order = getattr(bottleneck, 'patch_flatten_order', 'channel_first')

        # Infer spatial dimensions from num_patches
        # For 128x128 with 4x downsampling -> 32x32 latent, with patch_size=8 -> 4x4 patches = 16
        patches_per_side = int(math.sqrt(num_patches))
        latent_h = patches_per_side * patch_h
        latent_w = patches_per_side * patch_w

        # Decode through bottleneck helper (handles normalization/reshape)
        z_dl_nchw = bottleneck.decode_pattern_indices_to_latent(pattern_indices)
        z_dl_nchw = laser.post_bottleneck(z_dl_nchw)
        images = laser.decoder(z_dl_nchw)

        # Denormalize from [-1, 1] to [0, 1]
        images = (images + 1.0) / 2.0
        images = images.clamp(0.0, 1.0)

        return images

    def on_validation_epoch_end(self):
        """Generate and log sample images at end of validation epoch."""
        if self._laser_model is None:
            return

        # Only log every N epochs
        current_epoch = self.current_epoch
        if current_epoch % self._log_images_every_n_epochs != 0:
            return

        # Only log from rank 0
        if not self.trainer.is_global_zero:
            return

        # Check if logger is available
        if not hasattr(self, 'logger') or self.logger is None:
            return

        try:
            # Generate samples with different temperatures
            device = next(self.parameters()).device
            num_samples = self._num_samples_to_generate

            generated_images = []
            temperatures = [0.7, 1.0]

            for temp in temperatures:
                # Generate pattern indices
                pattern_indices = self.generate(
                    batch_size=num_samples,
                    temperature=temp,
                    top_k=100,
                    device=device,
                )

                # Decode to images
                images = self.decode_patterns_to_images(pattern_indices)
                generated_images.append((temp, images))

            # Create image grids and log to wandb
            log_dict = {"global_step": self.global_step, "epoch": current_epoch}

            for temp, images in generated_images:
                # Create grid
                grid = torchvision.utils.make_grid(images, nrow=4, normalize=False, padding=2)
                grid_np = grid.cpu().numpy().transpose(1, 2, 0)

                # Log to wandb
                log_dict[f"generated/temp_{temp:.1f}"] = wandb.Image(
                    grid_np,
                    caption=f"Generated samples (temp={temp:.1f}, epoch={current_epoch})"
                )

            # Add reconstruction sanity check: decode real pattern indices from validation data
            if hasattr(self, '_last_val_patterns') and self._last_val_patterns is not None:
                real_patterns = self._last_val_patterns[:num_samples].to(device)
                recon_images = self.decode_patterns_to_images(real_patterns)
                recon_grid = torchvision.utils.make_grid(recon_images, nrow=4, normalize=False, padding=2)
                recon_grid_np = recon_grid.cpu().numpy().transpose(1, 2, 0)
                log_dict["reconstructed/from_real_patterns"] = wandb.Image(
                    recon_grid_np,
                    caption=f"Reconstructed from real patterns (epoch={current_epoch})"
                )

            # Log all images
            if hasattr(self.logger, 'experiment') and hasattr(self.logger.experiment, 'log'):
                self.logger.experiment.log(log_dict)

        except Exception as e:
            print(f"Warning: Failed to generate visualization: {e}")

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        """Store a batch of patterns for reconstruction visualization."""
        if batch_idx == 0:
            self._last_val_patterns = batch.detach().cpu()
