import os
import sys
import torch
import pytest

# Add the src directory to the path (match other tests)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.vq_transformer import GPTModule, MinGPT


class DummyVectorQuantizer:
	def __init__(self, num_embeddings, embedding_dim):
		self.num_embeddings = num_embeddings
		# provide a codebook for potential decoding usage
		self.embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
		torch.nn.init.uniform_(self.embedding.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)


class DummyVQVAE:
	"""
	Minimal stub of VQVAE exposing:
	  - vector_quantizer.num_embeddings
	  - encode_to_indices(x) -> (indices [B, T], H_z, W_z)
	  - decode_from_indices(indices, H_z, W_z) -> images [B, 3, 64, 64]
	"""
	def __init__(self, num_embeddings=32, embedding_dim=64, H_z=4, W_z=4, out_hw=(64, 64)):
		self.vector_quantizer = DummyVectorQuantizer(num_embeddings, embedding_dim)
		self.H_z = H_z
		self.W_z = W_z
		self.out_hw = out_hw

	@torch.no_grad()
	def encode_to_indices(self, x):
		B = x.size(0)
		T = self.H_z * self.W_z
		indices = torch.randint(low=0, high=self.vector_quantizer.num_embeddings, size=(B, T), device=x.device)
		return indices, self.H_z, self.W_z

	@torch.no_grad()
	def decode_from_indices(self, indices, H_z, W_z):
		B = indices.size(0)
		H, W = self.out_hw
		# produce a simple image tensor; content is not validated here
		return torch.rand(B, 3, H, W, device=indices.device)


def test_gptmodule_shapes():
	vocab_size = 16
	block_size = 32
	B, T = 2, 20
	model = GPTModule(vocab_size=vocab_size, block_size=block_size, n_layer=2, n_head=2, n_embd=32)
	idx = torch.randint(0, vocab_size, (B, T))
	logits = model(idx)
	assert logits.shape == (B, T, vocab_size)
	assert torch.isfinite(logits).all()


def test_mingpt_training_step_loss_finite():
	device = torch.device('cpu')
	# dummy vqvae with small codebook and latent size
	vqvae = DummyVQVAE(num_embeddings=32, embedding_dim=32, H_z=4, W_z=4, out_hw=(32, 32))
	block_size = vqvae.H_z * vqvae.W_z
	model = MinGPT(vqvae=vqvae, block_size=block_size, n_layer=2, n_head=2, n_embd=64, learning_rate=1e-3, freeze_vqvae=True).to(device)
	# dummy batch
	x = torch.randn(2, 3, 32, 32, device=device)
	loss = model.training_step(x, batch_idx=0)
	assert torch.isfinite(loss)


def test_mingpt_sampling_shape():
	device = torch.device('cpu')
	vqvae = DummyVQVAE(num_embeddings=32, embedding_dim=32, H_z=4, W_z=4, out_hw=(32, 32))
	block_size = vqvae.H_z * vqvae.W_z
	model = MinGPT(vqvae=vqvae, block_size=block_size, n_layer=2, n_head=2, n_embd=64, learning_rate=1e-3, freeze_vqvae=True).to(device)
	with torch.no_grad():
		recon = model.sample(batch_size=2, H_z=vqvae.H_z, W_z=vqvae.W_z, temperature=1.0, top_k=5)
	assert recon.shape == (2, 3, 32, 32)
	assert torch.isfinite(recon).all()


