import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
import gc
import psutil
import os
from sklearn.manifold import TSNE

from models.bottleneck import VectorQuantizer, DictionaryLearningBottleneck

# Simple VAE architecture for both bottleneck types
class SimpleVAE(nn.Module):
    def __init__(self, bottleneck_module):
        super().__init__()
        
        # Encoder: input image -> latent representation
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),  # 28x28 -> 14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 14x14 -> 7x7
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),  # 7x7 -> 7x7
            nn.ReLU()
        )
        
        # Bottleneck: VQ or ODL
        self.bottleneck = bottleneck_module
        
        # Decoder: latent representation -> reconstructed image
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),  # 7x7 -> 7x7
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 7x7 -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),  # 14x14 -> 28x28
            nn.Sigmoid()  # Output is in [0, 1] range
        )
        
    def forward(self, x):
        z = self.encoder(x)
        z_q, loss, indices_or_codes = self.bottleneck(z)
        x_recon = self.decoder(z_q)
        return x_recon, loss, indices_or_codes

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def benchmark_training(model_type='vq', epochs=5, sparsity=5):
    """Train and benchmark a model using either VQ or ODL bottleneck"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Create bottleneck module
    if model_type == 'vq':
        bottleneck = VectorQuantizer(
            num_embeddings=512,
            embedding_dim=64,
            commitment_cost=0.25,
            decay=0.99
        )
        model_name = "VQ-VAE"
    else:
        bottleneck = DictionaryLearningBottleneck(
            dict_size=512,
            embedding_dim=64,
            sparsity=sparsity,
            commitment_cost=0.25,
            decay=0.99
        )
        model_name = f"ODL-{sparsity}"
    
    # Create model and optimizer
    model = SimpleVAE(bottleneck).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Track metrics
    train_losses = []
    recon_losses = []
    bottleneck_losses = []
    test_losses = []
    memory_usage = []
    epoch_times = []
    inference_times = []
    
    # Initial memory usage
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    initial_memory = get_memory_usage()
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_bottleneck_loss = 0
        
        start_time = time.time()
        
        for batch_idx, (data, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            data = data.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            recon_batch, bottleneck_loss, _ = model(data)
            
            # Reconstruction loss
            recon_loss = nn.MSELoss()(recon_batch, data)
            
            # Total loss
            loss = recon_loss + bottleneck_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track losses
            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_bottleneck_loss += bottleneck_loss.item()
            
            # Track memory after every 100 batches
            if batch_idx % 100 == 0:
                memory_usage.append(get_memory_usage() - initial_memory)
        
        # Track time per epoch
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)
        
        # Calculate average losses
        avg_loss = epoch_loss / len(train_loader)
        avg_recon_loss = epoch_recon_loss / len(train_loader)
        avg_bottleneck_loss = epoch_bottleneck_loss / len(train_loader)
        
        train_losses.append(avg_loss)
        recon_losses.append(avg_recon_loss)
        bottleneck_losses.append(avg_bottleneck_loss)
        
        # Test performance
        model.eval()
        test_loss = 0
        inference_start = time.time()
        
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                recon_batch, bottleneck_loss, _ = model(data)
                recon_loss = nn.MSELoss()(recon_batch, data)
                test_loss += (recon_loss + bottleneck_loss).item()
        
        inference_time = time.time() - inference_start
        inference_times.append(inference_time)
        
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_loss:.4f} (Recon: {avg_recon_loss:.4f}, Bottleneck: {avg_bottleneck_loss:.4f})")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Epoch Time: {epoch_time:.2f}s, Inference Time: {inference_time:.2f}s")
        print(f"  Memory Usage: {memory_usage[-1]:.2f} MB")
        
        # Save sample reconstructions at the end of each epoch
        with torch.no_grad():
            sample_data = next(iter(test_loader))[0][:8].to(device)
            reconstructions, _, _ = model(sample_data)
            
            plt.figure(figsize=(12, 6))
            for i in range(8):
                # Original images
                plt.subplot(2, 8, i+1)
                plt.imshow(sample_data[i, 0].cpu().numpy(), cmap='gray')
                plt.axis('off')
                
                # Reconstructions
                plt.subplot(2, 8, i+9)
                plt.imshow(reconstructions[i, 0].cpu().numpy(), cmap='gray')
                plt.axis('off')
            
            plt.suptitle(f"{model_name} - Epoch {epoch+1}")
            plt.tight_layout()
            plt.savefig(f"visualizations/{model_name}_recon_epoch_{epoch+1}.png")
            plt.close()
    
    # Save model and results
    torch.save(model.state_dict(), f"checkpoints/{model_name}_model.pt")
    
    # Return all metrics for plotting
    return {
        'name': model_name,
        'train_losses': train_losses,
        'recon_losses': recon_losses,
        'bottleneck_losses': bottleneck_losses,
        'test_losses': test_losses,
        'memory_usage': memory_usage,
        'epoch_times': epoch_times,
        'inference_times': inference_times,
        'model': model,
    }

def compare_bottlenecks(epochs=5, sparsity_values=[3, 5, 10]):
    """Run benchmarks for both bottleneck types and plot comparison"""
    results = []
    
    # Benchmark VQ-VAE
    print("Benchmarking VQ-VAE...")
    vq_results = benchmark_training(model_type='vq', epochs=epochs)
    results.append(vq_results)
    
    # Benchmark ODL with different sparsity values
    for sparsity in sparsity_values:
        print(f"Benchmarking ODL with sparsity={sparsity}...")
        odl_results = benchmark_training(model_type='odl', epochs=epochs, sparsity=sparsity)
        results.append(odl_results)
    
    # Plot comparison of training losses
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    for res in results:
        plt.plot(range(1, epochs+1), res['train_losses'], marker='o', label=res['name'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    for res in results:
        plt.plot(range(1, epochs+1), res['test_losses'], marker='o', label=res['name'])
    plt.title('Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    for res in results:
        plt.plot(range(1, epochs+1), res['epoch_times'], marker='o', label=res['name'])
    plt.title('Training Time per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    for res in results:
        plt.plot(range(1, epochs+1), res['inference_times'], marker='o', label=res['name'])
    plt.title('Inference Time on Test Set')
    plt.xlabel('Epoch')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("visualizations/bottleneck_comparison.png")
    plt.close()
    
    # Plot memory usage
    plt.figure(figsize=(10, 6))
    for res in results:
        plt.plot(res['memory_usage'], label=res['name'])
    plt.title('Memory Usage During Training')
    plt.xlabel('Training Steps (per 100 batches)')
    plt.ylabel('Memory Usage (MB)')
    plt.legend()
    plt.grid(True)
    plt.savefig("visualizations/bottleneck_memory_usage.png")
    plt.close()
    
    # Visualize latent space structure with test examples
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    test_batch, test_labels = next(iter(test_loader))
    test_batch = test_batch.to(device)
    
    plt.figure(figsize=(20, 10))
    
    for i, res in enumerate(results):
        model = res['model']
        model.eval()
        
        with torch.no_grad():
            # Get latent representations
            z = model.encoder(test_batch)
            z_q, _, codes = model.bottleneck(z)
            
            if isinstance(model.bottleneck, VectorQuantizer):
                # For VQ-VAE, plot histogram of code usage
                plt.subplot(2, len(results), i+1)
                code_hist = torch.bincount(codes, minlength=512)
                plt.bar(range(len(code_hist)), code_hist.cpu().numpy())
                plt.title(f"{res['name']} - Codebook Usage")
                plt.xlabel("Code Index")
                plt.ylabel("Usage Count")
            else:
                # For ODL, plot histogram of number of codes used per example
                plt.subplot(2, len(results), i+1)
                nonzero_counts = (codes.abs() > 1e-6).float().sum(dim=0)
                plt.hist(nonzero_counts.cpu().numpy(), bins=20)
                plt.title(f"{res['name']} - Atoms Used")
                plt.xlabel("Number of Active Atoms")
                plt.ylabel("Count")
            
            # Also plot t-SNE of embeddings
            plt.subplot(2, len(results), i+1+len(results))
            
            # Get a representative embedding for each example
            if isinstance(model.bottleneck, VectorQuantizer):
                # For VQ, use the quantized vectors directly
                embeddings = z_q.view(z_q.size(0), -1).cpu().numpy()
            else:
                # For ODL, use the reconstructed vectors
                embeddings = z_q.view(z_q.size(0), -1).cpu().numpy()
            
            # Only use 500 examples for t-SNE to speed things up
            embeddings_subset = embeddings[:500]
            labels_subset = test_labels.numpy()[:500]
            
            # Apply t-SNE
            tsne = TSNE(n_components=2, random_state=42)
            embedding_tsne = tsne.fit_transform(embeddings_subset)
            
            # Plot with different colors for different digits
            for digit in range(10):
                mask = labels_subset == digit
                plt.scatter(embedding_tsne[mask, 0], embedding_tsne[mask, 1], label=str(digit), alpha=0.6, s=20)
            
            if i == 0:  # Only add legend for the first plot to save space
                plt.legend(loc='best')
            plt.title(f"{res['name']} - t-SNE of Embeddings")
            
    plt.tight_layout()
    plt.savefig("visualizations/bottleneck_latent_space.png")
    plt.close()
    
    return results

if __name__ == "__main__":
    # Run the benchmark
    results = compare_bottlenecks(epochs=3, sparsity_values=[3, 5, 10])
    
    print("\nBenchmark Summary:")
    for res in results:
        final_train_loss = res['train_losses'][-1]
        final_test_loss = res['test_losses'][-1]
        avg_epoch_time = sum(res['epoch_times']) / len(res['epoch_times'])
        avg_inference_time = sum(res['inference_times']) / len(res['inference_times'])
        max_memory = max(res['memory_usage'])
        
        print(f"\n{res['name']}:")
        print(f"  Final Train Loss: {final_train_loss:.4f}")
        print(f"  Final Test Loss: {final_test_loss:.4f}")
        print(f"  Avg Epoch Time: {avg_epoch_time:.2f}s")
        print(f"  Avg Inference Time: {avg_inference_time:.2f}s")
        print(f"  Max Memory Usage: {max_memory:.2f} MB")