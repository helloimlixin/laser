import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """
    Vector Quantization module as described in "Neural Discrete Representation Learning"
    by van den Oord et al. (https://arxiv.org/abs/1711.00937)
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        # Create embedding table
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
    
    def forward(self, z):
        # z shape: [B, D, H, W]
        
        # Flatten z to [B*H*W, D]
        z_flattened = z.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)
        
        # Calculate distances to embeddings
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - \
            2 * torch.matmul(z_flattened, self.embedding.weight.t())
        
        # Find nearest embedding
        min_encoding_indices = torch.argmin(d, dim=1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.num_embeddings, device=z.device)
        min_encodings.scatter_(1, min_encoding_indices.unsqueeze(1), 1)
        
        # Get the quantized vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape[0], z.shape[2], z.shape[3], self.embedding_dim)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        
        # Compute loss
        # vq_loss: make the embedding vectors close to the encoder outputs
        # commitment_loss: make the encoder outputs close to the chosen embedding vectors
        vq_loss = F.mse_loss(z_q.detach(), z)
        commitment_loss = F.mse_loss(z_q, z.detach())
        loss = vq_loss + self.commitment_cost * commitment_loss
        
        # Straight-through estimator
        # Pass gradients from decoder to encoder
        z_q = z + (z_q - z).detach()
        
        return z_q, loss, min_encoding_indices
    
    def get_codebook_entry(self, indices):
        """
        Get codebook entries for given indices
        
        Args:
            indices: Tensor of indices [B, H, W]
        
        Returns:
            Tensor of codebook entries [B, D, H, W]
        """
        # Convert indices to one-hot
        indices = indices.view(-1)
        min_encodings = torch.zeros(indices.shape[0], self.num_embeddings, device=indices.device)
        min_encodings.scatter_(1, indices.unsqueeze(1), 1)
        
        # Get quantized vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight)
        
        # Reshape to match expected output
        z_q = z_q.view(indices.shape[0], self.embedding_dim)
        return z_q
