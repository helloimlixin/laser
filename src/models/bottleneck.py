import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

class VectorQuantizer(nn.Module):
    """
    Vector Quantizer implementation for VQ-VAE.
    
    The Vector Quantizer maps continuous encodings to discrete codes from a learned codebook.
    This is the key component that enables VQ-VAE to learn discrete representations.
    
    The quantization process involves:
    1. Finding the nearest embedding vector in the codebook for each spatial position in the input
    2. Replacing the input vectors with their corresponding codebook vectors
    3. Computing loss terms to train both the encoder and the codebook
    4. Using a straight-through estimator to allow gradient flow through the discrete bottleneck
    
    The codebook can be updated using either:
    - EMA updates (when decay > 0): More stable, not directly influenced by optimizer
    - Gradient descent (when decay = 0): Standard backpropagation approach
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25,
                 decay=0.99, epsilon=1e-5):
        """
        Initialize the Vector Quantizer.
        
        Args:
            num_embeddings (int): Size of the embedding dictionary (codebook size, typically 512 or 1024)
            embedding_dim (int): Dimension of each embedding vector in the codebook
            commitment_cost (float): Weight for the commitment loss, balancing encoder vs codebook training
            decay (float): Decay factor for exponential moving average updates of embeddings
                           If 0, standard backpropagation is used to update embeddings
            epsilon (float): Small constant to prevent division by zero in EMA update normalization
        """
        super().__init__()

        self.embedding_dim = embedding_dim  # Dimension of each embedding vector
        self.num_embeddings = num_embeddings  # Number of embedding vectors in the codebook
        self.commitment_cost = commitment_cost  # Weighting for commitment loss
        self.use_ema = bool(decay > 0.0)  # Whether to use EMA updates for the codebook
        self.decay = decay  # EMA decay factor (higher = slower updates)
        self.epsilon = epsilon  # Small constant for numerical stability

        # Create embedding table (codebook)
        # This is the dictionary of codes that continuous vectors will be mapped to
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # Initialize embedding weights with small random values
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

        if self.use_ema:
            print('Using EMA updates for codebook...')
            # Register buffers for EMA updates (not model parameters - not directly optimized)
            # ema_cluster_size: Tracks how often each codebook entry is used
            self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
            # ema_w: EMA of the encoder outputs assigned to each codebook entry
            self.register_buffer('ema_w', self.embedding.weight.data.clone())
            # Flag to control when EMA updates are performed
            self.register_buffer('ema_updating', torch.ones(1))
        else:
            print('Using standard backpropagation for codebook... (no EMA)')

    def forward(self, z):
        """
        Forward pass through the vector quantizer.
        
        Args:
            z (torch.Tensor): Output from encoder with shape [B, D, H, W]
                              B: batch size, D: embedding dimension, H, W: spatial dimensions
        
        Returns:
            z_q (torch.Tensor): Quantized tensor with same shape as input [B, D, H, W]
            loss (torch.Tensor): VQ loss (codebook loss + commitment loss)
            min_encoding_indices (torch.Tensor): Indices of the nearest embedding vectors [B*H*W]
        """
        # z shape: [B, D, H, W]

        # Reshape z to [B*H*W, D] for easier processing
        # The permute operation reorders dimensions to [B, H, W, D] before reshaping
        z_flattened = z.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)
        
        # Calculate squared distances between z_flattened and embedding vectors
        # This uses the || x - e ||^2 = ||x||^2 + ||e||^2 - 2*x^T*e formula for efficiency
        # Rather than computing distances directly, which would be more expensive
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - \
            2 * torch.matmul(z_flattened, self.embedding.weight.t())

        # Find nearest embedding for each z_flattened vector
        # min_encoding_indices contains indices of the closest codebook entry for each position
        min_encoding_indices = torch.argmin(d, dim=1)
        
        # Convert indices to one-hot encodings for gathering embeddings
        # min_encodings shape: [B*H*W, num_embeddings]
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.num_embeddings, device=z.device)
        min_encodings.scatter_(1, min_encoding_indices.unsqueeze(1), 1)

        # Update codebook using EMA if enabled and in training mode
        if self.use_ema and self.training:
            # EMA update for the codebook
            
            # 1. Update cluster size (how many vectors are assigned to each codebook entry)
            # Count occurrences of each embedding being used in this batch
            encodings_sum = min_encodings.sum(0)
            # Update the exponential moving average of cluster sizes
            self.ema_cluster_size.data.mul_(self.decay).add_(
                encodings_sum, alpha=(1 - self.decay)
            )

            # 2. Update embedding vectors based on assigned encoder outputs
            # Compute sum of all z vectors assigned to each embedding
            dw = torch.matmul(min_encodings.t(), z_flattened)
            # Update the exponential moving average of assigned vectors
            self.ema_w.data.mul_(self.decay).add_(dw, alpha=(1 - self.decay))

            # 3. Normalize the updated embeddings
            # Get total cluster size (with smoothing to prevent division by zero)
            n = self.ema_cluster_size.sum()
            # Normalize cluster sizes with Laplace smoothing
            cluster_size = ((self.ema_cluster_size + self.epsilon) /
                            (n + self.num_embeddings * self.epsilon) * n)
            # Normalize embeddings by their corresponding cluster sizes
            embed_normalized = self.ema_w / cluster_size.unsqueeze(1)
            # Update the actual embedding weights with the EMA-updated version
            self.embedding.weight.data.copy_(embed_normalized)

        # Get the quantized latent vectors using embedding lookup
        # Multiply one-hot encodings by embedding weights to get quantized vectors
        # Then reshape back to the original tensor shape [B, H, W, D]
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(
            z.shape[0], z.shape[2], z.shape[3], self.embedding_dim)
        # Permute back to match input tensor ordering [B, D, H, W]
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        # Compute VQ losses:
        
        # 1. Codebook loss (makes codebook vectors move towards encoder outputs)
        # The detach() on z prevents gradients from flowing back to the encoder
        vq_loss = F.mse_loss(z_q.detach(), z)
        
        # 2. Commitment loss (makes encoder outputs move towards codebook vectors)
        # The detach() on z_q prevents gradients from flowing back to the codebook
        commitment_loss = F.mse_loss(z_q, z.detach())
        
        # Combine losses with commitment_cost weighting
        loss = vq_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator
        # This allows gradients to flow back to encoder even though quantization is discrete
        # In the forward pass: z_q = selected embeddings
        # In the backward pass: gradients flow directly from z_q to z, bypassing quantization
        z_q = z + (z_q - z).detach()

        return z_q, loss, min_encoding_indices

    def get_codebook_entry(self, indices):
        """
        Retrieve specific vectors from the codebook using their indices.
        
        This is used during inference to map from discrete codes back to latent vectors.
        
        Args:
            indices (torch.Tensor): Indices of the codebook entries to retrieve
        
        Returns:
            z_q (torch.Tensor): The corresponding vectors from the codebook
        """
        # Flatten indices to 1D
        indices = indices.view(-1)
        
        # Convert indices to one-hot encodings
        min_encodings = torch.zeros(indices.shape[0], self.num_embeddings, device=indices.device)
        min_encodings.scatter_(1, indices.unsqueeze(1), 1)
        
        # Get corresponding embedding vectors using matmul with one-hot encodings
        z_q = torch.matmul(min_encodings, self.embedding.weight)
        
        # Reshape to match expected dimensions
        z_q = z_q.view(indices.shape[0], self.embedding_dim)
        
        return z_q


def setup_test_environment(seed=42):
    """
    Set up test data and model instances.
    
    Args:
        seed: Random seed for reproducibility
        
    Returns:
        tuple: (test input tensor, VQ model with EMA, VQ model without EMA, test parameters)
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    # Test parameters
    params = {
        'batch_size': 4,
        'embedding_dim': 64,
        'num_embeddings': 512,
        'height': 16,
        'width': 16
    }
    
    # Create a vector quantizer instance (with EMA updates)
    vq_with_ema = VectorQuantizer(
        num_embeddings=params['num_embeddings'],
        embedding_dim=params['embedding_dim'],
        commitment_cost=0.25,
        decay=0.99  # Using EMA updates
    )
    
    # Create another vector quantizer instance (without EMA updates)
    vq_without_ema = VectorQuantizer(
        num_embeddings=params['num_embeddings'],
        embedding_dim=params['embedding_dim'],
        commitment_cost=0.25,
        decay=0.0  # No EMA updates
    )
    
    # Generate random encoder output
    z = torch.randn(
        params['batch_size'], 
        params['embedding_dim'], 
        params['height'], 
        params['width']
    )
    
    return z, vq_with_ema, vq_without_ema, params

def test_forward_pass(model, input_tensor, name=""):
    """
    Test the forward pass of a VectorQuantizer model.
    
    Args:
        model: VectorQuantizer model instance
        input_tensor: Input tensor for the model
        name: Name identifier for the model (for printing)
        
    Returns:
        tuple: (quantized output, loss, indices)
    """
    print(f"\n--- Testing VectorQuantizer {name} ---")
    model.train()  # Set to training mode
    z_q, loss, indices = model(input_tensor)
    
    print(f"Quantized output shape: {z_q.shape}")
    print(f"VQ Loss: {loss.item():.6f}")
    print(f"Indices shape: {indices.shape}")
    
    return z_q, loss, indices

def analyze_codebook_usage(indices, num_embeddings):
    """
    Analyze how the codebook is being used.
    
    Args:
        indices: Indices tensor from VectorQuantizer
        num_embeddings: Total number of embeddings in the codebook
        
    Returns:
        tuple: (perplexity, number of codebook entries used)
    """
    # Convert indices to one-hot encodings
    encodings = torch.zeros(indices.shape[0], num_embeddings)
    encodings.scatter_(1, indices.unsqueeze(1), 1)
    
    # Calculate average probability for each codebook entry
    avg_probs = torch.mean(encodings, dim=0)
    
    # Calculate perplexity (effective codebook usage)
    perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
    
    # Count how many codebook entries are used
    codebook_usage = torch.sum(avg_probs > 0)
    
    print(f"Perplexity: {perplexity.item():.2f} (effective codebook usage)")
    print(f"Codebook entries used: {codebook_usage.item()} out of {num_embeddings}")
    
    return perplexity, codebook_usage

def test_codebook_retrieval(model, indices, z_q, embedding_dim):
    """
    Test the get_codebook_entry method of VectorQuantizer.
    
    Args:
        model: VectorQuantizer model instance
        indices: Indices tensor from forward pass
        z_q: Quantized output from forward pass
        embedding_dim: Dimension of embedding vectors
        
    Returns:
        float: Reconstruction error
    """
    print("\n--- Testing get_codebook_entry ---")
    # Get a subset of indices
    test_indices = indices[:10]
    retrieved_embeddings = model.get_codebook_entry(test_indices)
    print(f"Retrieved embeddings shape: {retrieved_embeddings.shape}")
    
    # Verify reconstruction error is minimal
    original_flat = z_q.permute(0, 2, 3, 1).contiguous().view(-1, embedding_dim)
    original_subset = original_flat[:10]
    reconstruction_error = F.mse_loss(retrieved_embeddings, original_subset)
    print(f"Reconstruction error: {reconstruction_error.item():.10f} (should be close to zero)")
    
    return reconstruction_error.item()

def visualize_codebook_usage(indices, model_name=""):
    """
    Visualize the distribution of codebook indices used.
    
    Args:
        indices: Indices tensor from VectorQuantizer
        model_name: Name identifier for the model (for output filename)
        
    Returns:
        str: Path to saved visualization
    """
    print(f"\n--- Visualizing Codebook Usage {model_name} ---")
    used_indices = indices.cpu().numpy()
    
    plt.figure(figsize=(10, 5))
    plt.hist(used_indices, bins=50, alpha=0.7)
    plt.title(f"Codebook Usage Distribution {model_name}")
    plt.xlabel("Codebook Index")
    plt.ylabel("Frequency")
    
    filename = f"codebook_usage_{model_name.replace(' ', '_')}.png"
    plt.savefig(filename)
    plt.close()
    
    print(f"Saved codebook usage visualization to '{filename}'")
    return filename

def test_inference_mode(model, input_tensor, model_name=""):
    """
    Test the model in evaluation/inference mode.
    
    Args:
        model: VectorQuantizer model instance
        input_tensor: Input tensor for the model
        model_name: Name identifier for the model (for printing)
        
    Returns:
        tuple: (quantized output, loss, indices)
    """
    print(f"\n--- Testing inference mode {model_name} ---")
    model.eval()  # Set to evaluation mode
    z_q, loss, indices = model(input_tensor)
    
    print(f"Inference mode - Quantized output shape: {z_q.shape}")
    print(f"Inference mode - VQ Loss: {loss.item():.6f}")
    
    return z_q, loss, indices

def run_all_tests(
    test_ema=True,                # Test model with EMA updates
    test_no_ema=True,             # Test model without EMA updates
    test_codebook_usage=True,     # Analyze codebook usage statistics
    test_retrieval=True,          # Test codebook entry retrieval
    test_visualization=True,      # Generate visualizations
    test_inference=True,          # Test models in inference mode
    seed=42                       # Random seed for reproducibility
):
    """
    Run tests for the VectorQuantizer with control over which tests to execute.
    
    Args:
        test_ema: Whether to test the model with EMA updates
        test_no_ema: Whether to test the model without EMA updates
        test_codebook_usage: Whether to analyze codebook usage
        test_retrieval: Whether to test codebook entry retrieval
        test_visualization: Whether to generate visualizations
        test_inference: Whether to test inference mode
        seed: Random seed for reproducibility
        
    Returns:
        dict: Results from the tests that were run
    """
    # Setup
    print("Setting up test environment...")
    z, vq_with_ema, vq_without_ema, params = setup_test_environment(seed)
    print(f"Input shape: {z.shape}")
    
    # Dictionary to store test results
    results = {}
    
    # Test model with EMA updates
    if test_ema:
        z_q_ema, loss_ema, indices_ema = test_forward_pass(
            vq_with_ema, z, name="with EMA updates"
        )
        results['ema'] = {'z_q': z_q_ema, 'loss': loss_ema, 'indices': indices_ema}
        
        # Analyze codebook usage for EMA model
        if test_codebook_usage:
            perplexity, usage = analyze_codebook_usage(indices_ema, params['num_embeddings'])
            results['ema']['perplexity'] = perplexity
            results['ema']['codebook_usage'] = usage
        
        # Test codebook retrieval for EMA model
        if test_retrieval:
            error = test_codebook_retrieval(
                vq_with_ema, indices_ema, z_q_ema, params['embedding_dim']
            )
            results['ema']['retrieval_error'] = error
        
        # Test inference mode for EMA model
        if test_inference:
            z_q_eval, loss_eval, indices_eval = test_inference_mode(
                vq_with_ema, z, model_name="with EMA"
            )
            results['ema']['inference'] = {
                'z_q': z_q_eval,
                'loss': loss_eval,
                'indices': indices_eval
            }
    
    # Test model without EMA updates
    if test_no_ema:
        z_q_no_ema, loss_no_ema, indices_no_ema = test_forward_pass(
            vq_without_ema, z, name="without EMA updates"
        )
        results['no_ema'] = {'z_q': z_q_no_ema, 'loss': loss_no_ema, 'indices': indices_no_ema}
        
        # Analyze codebook usage for non-EMA model
        if test_codebook_usage:
            perplexity, usage = analyze_codebook_usage(indices_no_ema, params['num_embeddings'])
            results['no_ema']['perplexity'] = perplexity
            results['no_ema']['codebook_usage'] = usage
        
        # Test codebook retrieval for non-EMA model
        if test_retrieval:
            error = test_codebook_retrieval(
                vq_without_ema, indices_no_ema, z_q_no_ema, params['embedding_dim']
            )
            results['no_ema']['retrieval_error'] = error
        
        # Test inference mode for non-EMA model
        if test_inference:
            z_q_eval, loss_eval, indices_eval = test_inference_mode(
                vq_without_ema, z, model_name="without EMA"
            )
            results['no_ema']['inference'] = {
                'z_q': z_q_eval,
                'loss': loss_eval,
                'indices': indices_eval
            }
    
    # Generate visualizations
    if test_visualization and test_ema:
        vis_path = visualize_codebook_usage(
            results['ema']['indices'], model_name="with EMA"
        )
        results['ema']['visualization_path'] = vis_path
    
    if test_visualization and test_no_ema:
        vis_path = visualize_codebook_usage(
            results['no_ema']['indices'], model_name="without EMA"
        )
        results['no_ema']['visualization_path'] = vis_path
    
    print("\nCompleted requested tests successfully!")
    return results

# Add this line at the bottom of the file to run all tests by default when the script is executed
if __name__ == "__main__":
    run_all_tests()  # Run all tests with default settings
