import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Re-using our Dictionary Learning Classes ---

def soft_thresholding(x, threshold):
    return torch.sign(x) * torch.relu(torch.abs(x) - threshold)

def batch_omp(X, D, k_max, tol=1e-7, debug=False):
    """
    Batched Orthogonal Matching Pursuit.
    Args:
        X: Signals [B, M]
        D: Dictionary [M, N]
        k_max: Sparsity level
    """
    B, M = X.shape
    _, N = D.shape
    
    # We work with X transposed [M, B] for compatibility with logic
    X_t = X.t()
    
    dictionary_t = D.t()
    diag_eps = 1e-5
    G = dictionary_t @ D
    G = G + diag_eps * torch.eye(N, device=X.device, dtype=X.dtype)
    eps = (X_t ** 2).sum(dim=0)
    h_bar = (dictionary_t @ X_t).t()
    
    h = h_bar.clone()
    x = torch.zeros_like(h_bar)
    L = torch.ones(B, 1, 1, device=X.device, dtype=X.dtype)
    I = torch.ones(B, 0, device=X.device, dtype=torch.long)
    I_logic = torch.zeros_like(h_bar, dtype=torch.bool)
    delta = torch.zeros(B, device=X.device, dtype=X.dtype)
    
    def _update_logical(logical, to_add):
        running_idx = torch.arange(to_add.shape[0], device=to_add.device)
        logical[running_idx, to_add] = True

    k = 0
    batch_idx = torch.arange(B, device=X.device)
    
    while k < k_max and eps.max() > tol:
        k += 1
        index = (h * (~I_logic).float()).abs().argmax(dim=1)
        _update_logical(I_logic, index)
        expanded_batch_idx = batch_idx.unsqueeze(0).expand(k, B).t()
        
        if k > 1:
            G_stack = G[I[batch_idx, :], index[expanded_batch_idx[..., :-1]]].view(B, k - 1, 1)
            try:
                w = torch.linalg.solve_triangular(L, G_stack, upper=False)
            except AttributeError:
                w = torch.triangular_solve(G_stack, L, upper=False).solution
            w = w.view(B, 1, k - 1)
            diag_g = G[index, index].view(B, 1, 1)
            w_corner = torch.sqrt(torch.clamp(diag_g - (w ** 2).sum(dim=2, keepdim=True), min=diag_eps))
            
            k_zeros = torch.zeros(B, k - 1, 1, device=X.device, dtype=X.dtype)
            L = torch.cat((torch.cat((L, k_zeros), dim=2), torch.cat((w, w_corner), dim=2)), dim=1)
            
        I = torch.cat([I, index.unsqueeze(1)], dim=1)
        h_stack = h_bar[expanded_batch_idx, I[batch_idx, :]].view(B, k, 1)
        try:
            x_stack = torch.cholesky_solve(h_stack, L)
        except AttributeError:
            x_stack = torch.linalg.cholesky_solve(h_stack, L)
        
        x[batch_idx.unsqueeze(1), I[batch_idx]] = x_stack.squeeze(-1)
        beta = x[batch_idx.unsqueeze(1), I[batch_idx]].unsqueeze(1).bmm(G[I[batch_idx], :]).squeeze(1)
        h = h_bar - beta
        new_delta = (x * beta).sum(dim=1)
        eps = eps + delta - new_delta
        delta = new_delta
        
        if not torch.isfinite(x).all(): break
            
    return x # [B, N]

def ista_solve(Y, D, lambda_l1, num_iters=50, lr=0.01):
    batch_size = Y.shape[0]
    n_atoms = D.shape[0]
    X = torch.zeros(batch_size, n_atoms, device=Y.device)
    
    # Precompute for efficiency
    Dt = D.t()
    
    # Lipschitz constant estimation (power iteration could be better, but simple bound here)
    # L <= ||D||_F^2 is a loose bound, ||D^T D||_2 is tighter.
    # Since columns are normalized, ||D||_2 <= sqrt(n_atoms) approx?
    # Actually, just let's cap the values to avoid explosion.
    
    for _ in range(num_iters):
        # Y (B, F), X (B, A), D (A, F) -> X @ D (B, F)
        residual = Y - (X @ D)
        # residual (B, F), Dt (F, A) -> grad_X (B, A)
        grad_X = -residual @ Dt
        
        # Check for NaNs
        if torch.isnan(grad_X).any():
            # If NaN, stop updating
            break
            
        X = X - lr * grad_X
        X = soft_thresholding(X, lambda_l1 * lr)
    return X

class DictionaryLearning(nn.Module):
    def __init__(self, n_features, n_atoms, sparsity_level=5):
        super().__init__()
        self.n_features = n_features
        self.n_atoms = n_atoms
        self.sparsity_level = sparsity_level
        
        # Initialize D randomly
        self.D = nn.Parameter(torch.randn(n_atoms, n_features))
        self.normalize_dictionary()

    def normalize_dictionary(self):
        with torch.no_grad():
            self.D.div_(torch.norm(self.D, p=2, dim=1, keepdim=True) + 1e-8)

    def forward(self, Y, n_ista_iters=None, ista_lr=None):
        with torch.no_grad():
            # Use OMP instead of ISTA
            # batch_omp expects [B, Features] signals and [Atoms, Features] dictionary?
            # Wait, batch_omp implementation takes X:[B,M], D:[M,N]?
            # Let's check signature above: X [B, M], D [M, N] 
            # Here Y is [B, Features]. D is [Atoms, Features].
            # This means M=Features. N=Atoms?
            # We need D to be [Features, Atoms] for the OMP math (D^T D).
            # Our self.D is [Atoms, Features].
            # So pass D=self.D.t() which is [Features, Atoms].
            X = batch_omp(Y, self.D.t(), k_max=self.sparsity_level)
            
        Y_pred = X @ self.D
        return Y_pred, X

# --- 2. Setup & Data Loading ---

def load_mnist(batch_size=256):
    transform = transforms.Compose([
        transforms.ToTensor(), 
        # transforms.Normalize((0.1307,), (0.3081,)) # Standard MNIST Normalization
        # Let's keep it in [0, 1] range for easier interpretation and stability
    ])
    
    # Download MNIST
    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    # We'll use a smaller subset to make training faster for this demo
    subset_indices = torch.arange(2000) 
    subset = torch.utils.data.Subset(dataset, subset_indices)
    
    loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True)
    return loader

# --- 3. Visualization Helper ---

def plot_dictionary_atoms(D, n_rows=10, n_cols=10):
    """Reshapes flattened atoms back to 28x28 images and plots them grid-style"""
    # D shape: (n_atoms, 784)
    plt.figure(figsize=(10, 10))
    
    # Move to CPU for plotting
    atoms = D.detach().cpu().numpy()
    
    for i in range(n_rows * n_cols):
        if i >= len(atoms): break
        plt.subplot(n_rows, n_cols, i + 1)
        # Reshape 784 -> 28x28
        plt.imshow(atoms[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
    
    plt.suptitle("Learned Dictionary Atoms (The 'Strokes')", fontsize=16)
    plt.savefig('atoms.png')
    print("Saved dictionary atoms to atoms.png")
    plt.close()

def plot_reconstruction(original, reconstructed):
    """Compare original digits vs reconstructed"""
    original = original.detach().cpu().numpy()
    reconstructed = reconstructed.detach().cpu().numpy()
    
    plt.figure(figsize=(10, 4))
    n_show = 5
    for i in range(n_show):
        # Original
        plt.subplot(2, n_show, i + 1)
        plt.imshow(original[i].reshape(28, 28), cmap='gray')
        plt.title("Original")
        plt.axis('off')
        
        # Reconstructed
        plt.subplot(2, n_show, i + 1 + n_show)
        plt.imshow(reconstructed[i].reshape(28, 28), cmap='gray')
        plt.title("Reconstructed")
        plt.axis('off')
    plt.savefig('reconstruction.png')
    print("Saved reconstructions to reconstruction.png")
    plt.close()

# --- 4. Main Execution ---

def run_mnist_experiment():
    # Detect device (Mac MPS / CUDA / CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Config
    BATCH_SIZE = 100
    N_EPOCHS = 20
    N_ATOMS = 100       # We want 100 basic shapes
    SPARSITY_LEVEL = 5  # Fixed number of atoms per image
    
    train_loader = load_mnist(BATCH_SIZE)
    
    # Initialize dictionary with random patches from data for faster convergence
    init_batch, _ = next(iter(train_loader))
    init_batch = init_batch.view(init_batch.size(0), -1)
    # Randomly select N_ATOMS from the batch (with replacement if batch < atoms)
    indices = torch.randint(0, init_batch.size(0), (N_ATOMS,))
    init_D = init_batch[indices].clone()
    
    # 28x28 images = 784 features
    model = DictionaryLearning(n_features=784, n_atoms=N_ATOMS, sparsity_level=SPARSITY_LEVEL).to(device)
    # Overwrite random D with data-initialized D
    with torch.no_grad():
        model.D.data.copy_(init_D.to(device))
        model.normalize_dictionary()
        
    optimizer = optim.Adam([model.D], lr=0.01) # Switch to Adam for smoother convergence
    
    print("Starting Training...")
    
    loss_history = []
    
    for epoch in range(N_EPOCHS):
        total_loss = 0
        total_sparsity = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            # Flatten image: (Batch, 1, 28, 28) -> (Batch, 784)
            data = data.view(data.size(0), -1).to(device)
            
            optimizer.zero_grad()
            
            # Forward (Infer X, Reconstruct Y)
            Y_pred, X_learned = model(data)
            
            # Loss: 0.5 * ||Y - Y_pred||^2
            # Normalize by batch size to keep gradients stable regardless of batch size
            loss = 0.5 * torch.mean(torch.sum((data - Y_pred) ** 2, dim=1))
            
            # Backward
            loss.backward()
            optimizer.step()
            
            # Normalize D columns
            model.normalize_dictionary()
            
            total_loss += loss.item()
            # Calculate sparsity (% of zero elements in X)
            sparsity = (X_learned.abs() < 1e-3).float().mean()
            total_sparsity += sparsity.item()
            
        avg_loss = total_loss / len(train_loader)
        avg_sparsity = total_sparsity / len(train_loader)
        loss_history.append(avg_loss)
        
        print(f"Epoch {epoch+1}: Loss={avg_loss:.2f}, Sparsity={avg_sparsity*100:.1f}% (Zeros)")

    print("Training Complete.")
    
    # --- Visualization ---
    
    # 1. Show the Atoms
    plot_dictionary_atoms(model.D)
    
    # 2. Show Reconstruction
    # Get a fresh batch
    data, _ = next(iter(train_loader))
    data = data.view(data.size(0), -1).to(device)
    Y_pred, _ = model(data)
    plot_reconstruction(data, Y_pred)

if __name__ == "__main__":
    run_mnist_experiment()
