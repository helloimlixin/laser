import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
import torchvision
import wandb

from .vqvae import VQVAE


class GPTConfig:
	embd_pdrop = 0.1
	resid_pdrop = 0.1
	attn_pdrop = 0.1

	def __init__(self, vocab_size, block_size, **kwargs):
		self.vocab_size = vocab_size
		self.block_size = block_size
		for k, v in kwargs.items():
			setattr(self, k, v)


class CausalSelfAttention(nn.Module):
	def __init__(self, config: GPTConfig):
		super().__init__()
		assert config.n_embd % config.n_head == 0
		self.key = nn.Linear(config.n_embd, config.n_embd)
		self.query = nn.Linear(config.n_embd, config.n_embd)
		self.value = nn.Linear(config.n_embd, config.n_embd)
		self.attn_drop = nn.Dropout(config.attn_pdrop)
		self.resid_drop = nn.Dropout(config.resid_pdrop)
		self.proj = nn.Linear(config.n_embd, config.n_embd)
		mask = torch.tril(torch.ones(config.block_size, config.block_size))
		self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))
		self.n_head = config.n_head

	def forward(self, x):
		B, T, C = x.size()
		k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
		q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
		v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

		att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
		att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
		att = F.softmax(att, dim=-1)
		att = self.attn_drop(att)
		y = att @ v
		y = y.transpose(1, 2).contiguous().view(B, T, C)
		y = self.resid_drop(self.proj(y))
		return y


class Block(nn.Module):
	def __init__(self, config: GPTConfig):
		super().__init__()
		self.ln1 = nn.LayerNorm(config.n_embd)
		self.ln2 = nn.LayerNorm(config.n_embd)
		self.attn = CausalSelfAttention(config)
		self.mlp = nn.Sequential(
			nn.Linear(config.n_embd, 4 * config.n_embd),
			nn.GELU(),
			nn.Linear(4 * config.n_embd, config.n_embd),
			nn.Dropout(config.resid_pdrop),
		)

	def forward(self, x):
		x = x + self.attn(self.ln1(x))
		x = x + self.mlp(self.ln2(x))
		return x


class GPTModule(nn.Module):
	def __init__(self, vocab_size, block_size, n_layer=8, n_head=8, n_embd=512,
	             embd_pdrop=0.0, resid_pdrop=0.0, attn_pdrop=0.0):
		super().__init__()
		config = GPTConfig(vocab_size=vocab_size, block_size=block_size,
		                   embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop,
		                   n_layer=n_layer, n_head=n_head, n_embd=n_embd)
		self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
		self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
		self.drop = nn.Dropout(config.embd_pdrop)
		self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
		self.ln_f = nn.LayerNorm(config.n_embd)
		self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
		self.block_size = config.block_size
		self.apply(self._init_weights)
		self.config = config

	def _init_weights(self, module):
		if isinstance(module, (nn.Linear, nn.Embedding)):
			module.weight.data.normal_(mean=0.0, std=0.02)
			if isinstance(module, nn.Linear) and module.bias is not None:
				module.bias.data.zero_()
		elif isinstance(module, nn.LayerNorm):
			module.bias.data.zero_()
			module.weight.data.fill_(1.0)

	def get_block_size(self):
		return self.block_size

	def forward(self, idx):
		token_embeddings = self.tok_emb(idx)  # [B, T, C]
		T = token_embeddings.size(1)
		assert T <= self.block_size, "Sequence length exceeds block size."
		position_embeddings = self.pos_emb[:, :T, :]
		x = self.drop(token_embeddings + position_embeddings)
		x = self.blocks(x)
		x = self.ln_f(x)
		logits = self.head(x)
		return logits

	@torch.no_grad()
	def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, forbid_token=None):
		for _ in range(max_new_tokens):
			idx_cond = idx[:, -self.block_size:]
			logits = self(idx_cond)
			logits = logits[:, -1, :] / max(temperature, 1e-8)
			# Optionally forbid a specific token (e.g., BOS after the first step)
			if forbid_token is not None and idx.size(1) >= 1:
				logits[:, forbid_token] = -float('Inf')
			# Top-p (nucleus) sampling takes precedence if provided
			if top_p is not None:
				probs = F.softmax(logits, dim=-1)
				sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
				cumprobs = torch.cumsum(sorted_probs, dim=-1)
				# mask tokens where cumulative prob exceeds threshold (keep minimal set > top_p)
				mask = cumprobs > float(top_p)
				# ensure at least one token kept
				mask[..., 1:] = mask[..., :-1].clone()
				mask[..., 0] = False
				sorted_probs[mask] = 0.0
				sorted_probs = sorted_probs / (sorted_probs.sum(dim=-1, keepdim=True) + 1e-12)
				next_sorted = torch.multinomial(sorted_probs, num_samples=1)
				next_idx = sorted_indices.gather(-1, next_sorted)
			else:
				if top_k is not None:
					v, _ = torch.topk(logits, top_k)
					logits[logits < v[:, [-1]]] = -float('Inf')
				probs = F.softmax(logits, dim=-1)
				next_idx = torch.multinomial(probs, num_samples=1)
			idx = torch.cat((idx, next_idx), dim=1)
		return idx


class MinGPT(pl.LightningModule):
	def __init__(self,
	             vqvae: VQVAE,
	             block_size: int,
	             n_layer: int = 8,
	             n_head: int = 8,
	             n_embd: int = 512,
	             learning_rate: float = 3e-4,
	             beta: float = 0.9,
	             freeze_vqvae: bool = True):
		super().__init__()
		self.vqvae = vqvae
		if freeze_vqvae:
			if hasattr(self.vqvae, "eval"):
				self.vqvae.eval()
			# Some test stubs may not expose .parameters()
			try:
				for p in self.vqvae.parameters():
					p.requires_grad = False
			except Exception:
				pass
		codebook_size = self.vqvae.vector_quantizer.num_embeddings
		self.bos_id = codebook_size  # reserve last id for BOS
		vocab_size = codebook_size + 1
		self.gpt = GPTModule(vocab_size=vocab_size, block_size=block_size,
		                     n_layer=n_layer, n_head=n_head, n_embd=n_embd)
		self.learning_rate = learning_rate
		self.beta = beta
		self.log_images_every_n_steps = 200
		self.save_hyperparameters(ignore=['vqvae', 'gpt'])

	def training_step(self, batch, batch_idx):
		x = batch[0] if isinstance(batch, (list, tuple)) else batch
		with torch.no_grad():
			indices, H_z, W_z = self.vqvae.encode_to_indices(x)
		T = H_z * W_z
		if T > self.gpt.get_block_size():
			indices = indices[:, :self.gpt.get_block_size()]
			T = self.gpt.get_block_size()
		# Autoregressive with BOS priming: predict the true tokens given BOS+shifted inputs
		bos = torch.full((indices.size(0), 1), self.bos_id, dtype=torch.long, device=indices.device)
		x_in = torch.cat([bos, indices[:, :-1]], dim=1)
		x_out = indices
		logits = self.gpt(x_in)
		loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), x_out.reshape(-1))
		self._safe_log('train/transformer_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
		if (batch_idx % self.log_images_every_n_steps == 0) and self._is_global_zero():
			with torch.no_grad():
				progress = self._generate_progress(batch_size=4, H_z=H_z, W_z=W_z, temperature=1.0, top_k=64)
			self._log_progress_images(progress, split='train')
		return loss

	def validation_step(self, batch, batch_idx):
		x = batch[0] if isinstance(batch, (list, tuple)) else batch
		with torch.no_grad():
			indices, H_z, W_z = self.vqvae.encode_to_indices(x)
		T = H_z * W_z
		if T > self.gpt.get_block_size():
			indices = indices[:, :self.gpt.get_block_size()]
			T = self.gpt.get_block_size()
		bos = torch.full((indices.size(0), 1), self.bos_id, dtype=torch.long, device=indices.device)
		x_in = torch.cat([bos, indices[:, :-1]], dim=1)
		x_out = indices
		logits = self.gpt(x_in)
		loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), x_out.reshape(-1))
		self._safe_log('val/transformer_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
		if (batch_idx % self.log_images_every_n_steps == 0) and self._is_global_zero():
			with torch.no_grad():
				progress = self._generate_progress(batch_size=4, H_z=H_z, W_z=W_z, temperature=1.0, top_k=64)
			self._log_progress_images(progress, split='val')
		return loss

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(self.beta, 0.999))
		return optimizer

	@torch.no_grad()
	def sample(self, batch_size: int, H_z: int, W_z: int, temperature: float = 1.0, top_k: int = None):
		device = next(self.parameters()).device
		T = H_z * W_z
		start = torch.full((batch_size, 1), self.bos_id, dtype=torch.long, device=device)
		tokens_bos = self.gpt.generate(start, max_new_tokens=T, temperature=temperature, top_k=top_k, top_p=None, forbid_token=self.bos_id)
		tokens = tokens_bos[:, 1:1+T]
		recon = self.vqvae.decode_from_indices(tokens, H_z, W_z)
		return recon

	@torch.no_grad()
	def _generate_progress(self, batch_size: int, H_z: int, W_z: int, temperature: float = 1.0, top_k: int = None):
		device = next(self.parameters()).device
		T = H_z * W_z
		milestones = [max(1, int(T * r)) for r in (0.25, 0.5, 0.75, 1.0)]
		milestones = sorted(set(milestones))
		idx = torch.full((batch_size, 1), self.bos_id, dtype=torch.long, device=device)
		progress_recons = []
		for step in range(1, T):
			idx_cond = idx[:, -self.gpt.get_block_size():]
			logits = self.gpt(idx_cond)
			logits = logits[:, -1, :] / max(temperature, 1e-8)
			logits[:, self.bos_id] = -float('Inf')
			if top_k is not None:
				v, _ = torch.topk(logits, top_k)
				logits[logits < v[:, [-1]]] = -float('Inf')
			probs = F.softmax(logits, dim=-1)
			next_idx = torch.multinomial(probs, num_samples=1)
			idx = torch.cat((idx, next_idx), dim=1)
			if idx.size(1) in milestones:
				tokens_full = idx
				if tokens_full.size(1) < T:
					pad = torch.zeros(batch_size, T - tokens_full.size(1), dtype=torch.long, device=device)
					tokens_full = torch.cat([tokens_full, pad], dim=1)
				# drop BOS for decoding
				recon = self.vqvae.decode_from_indices(tokens_full[:, 1:1+T], H_z, W_z)
				progress_recons.append(recon)
			if idx.size(1) >= T:
				break
		if len(progress_recons) == 0 or milestones[-1] != T:
			tokens_full = idx
			if tokens_full.size(1) < T:
				pad = torch.zeros(batch_size, T - tokens_full.size(1), dtype=torch.long, device=device)
				tokens_full = torch.cat([tokens_full, pad], dim=1)
			recon = self.vqvae.decode_from_indices(tokens_full[:, 1:1+T], H_z, W_z)
			progress_recons.append(recon)
		return progress_recons

	def _is_global_zero(self) -> bool:
		trainer = getattr(self, "trainer", None)
		return bool(getattr(trainer, "is_global_zero", False))

	def _log_progress_images(self, progress_recons, split: str = 'train'):
		if not getattr(self, "logger", None) or not hasattr(self.logger, "experiment"):
			return
		if not self._is_global_zero():
			return
		if not progress_recons:
			return
		bmax = min(4, progress_recons[0].size(0))
		tiles = []
		for bi in range(bmax):
			for recon in progress_recons:
				tiles.append(recon[bi:bi+1])
		stack = torch.cat(tiles, dim=0)
		dm = getattr(getattr(self, "trainer", None), "datamodule", None)
		if dm is not None and hasattr(dm, "config") and hasattr(dm.config, "mean") and hasattr(dm.config, "std"):
			mean = torch.tensor(dm.config.mean, device=stack.device, dtype=stack.dtype).view(1, -1, 1, 1)
			std = torch.tensor(dm.config.std, device=stack.device, dtype=stack.dtype).view(1, -1, 1, 1)
			stack_disp = (stack * std + mean).clamp(0.0, 1.0)
		else:
			stack_disp = ((stack + 1.0) / 2.0).clamp(0.0, 1.0)
		grid = torchvision.utils.make_grid(stack_disp, nrow=len(progress_recons), normalize=False)
		grid = torch.nan_to_num(grid, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
		grid_np = grid.cpu().numpy().transpose(1, 2, 0)
		self.logger.experiment.log({
			f"{split}/gpt_progress": [wandb.Image(grid_np, caption="left→right: 25%, 50%, 75%, 100%")],
			"global_step": self.global_step
		})

	def _safe_log(self, *args, **kwargs):
		# Avoid accessing the public `trainer` property (raises when None)
		trainer = getattr(self, '_trainer', None)
		if trainer is not None:
			try:
				self.log(*args, **kwargs)
			except Exception:
				pass

