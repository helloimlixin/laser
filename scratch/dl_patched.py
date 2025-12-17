import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import os

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# ==========================================
# 1. OPTIMIZED SOLVER (With Power Iteration)
# ==========================================

def batch_omp(X, D, k_max, tol=1e-6):
    """
    Batched Orthogonal Matching Pursuit.
    Args:
        X: Signals [B, M]
        D: Dictionary [M, N] (passed as [Atoms, Features] D.t() from forward)
        k_max: Sparsity level
    """
    # X is [B, M], D is [M, N] (after transpose from D[A, F])
    # D passed here should be [M, N] where M=Features, N=Atoms.
    # Our dictionary model stores D as [N, M]. So we pass D.t().
    B, M = X.shape
    _, N = D.shape
    
    # We work with X transposed [M, B] for compatibility with OMP logic
    X_t = X.t()
    
    dictionary_t = D.t() # [N, M]
    diag_eps = 1e-4  # Increased for numerical stability
    # G = D^T D [N, N]
    G = dictionary_t @ D
    G = G + diag_eps * torch.eye(N, device=X.device, dtype=X.dtype)
    eps = (X_t ** 2).sum(dim=0)
    h_bar = (dictionary_t @ X_t).t() # [B, N]
    
    # Check for NaN in inputs
    if not torch.isfinite(h_bar).all() or not torch.isfinite(eps).all():
        return torch.zeros(B, N, device=X.device, dtype=X.dtype)
    
    h = h_bar.clone()
    x = torch.zeros_like(h_bar) # [B, N]
    L = torch.ones(B, 1, 1, device=X.device, dtype=X.dtype)
    I = torch.ones(B, 0, device=X.device, dtype=torch.long)
    I_logic = torch.zeros_like(h_bar, dtype=torch.bool)
    delta = torch.zeros(B, device=X.device, dtype=X.dtype)
    
    def _update_logical(logical, to_add):
        running_idx = torch.arange(to_add.shape[0], device=to_add.device)
        logical[running_idx, to_add] = True

    k = 0
    batch_idx = torch.arange(B, device=X.device)
    
    # Use clamped eps to avoid negative values from floating point errors
    while k < k_max and eps.clamp(min=0).max() > tol:
        k += 1
        # Greedy selection - mask already selected atoms
        masked_h = h.clone()
        masked_h[I_logic] = 0
        index = masked_h.abs().argmax(dim=1)
        _update_logical(I_logic, index)
        expanded_batch_idx = batch_idx.unsqueeze(0).expand(k, B).t()
        
        if k > 1:
            # Update Cholesky
            G_stack = G[I[batch_idx, :], index[expanded_batch_idx[..., :-1]]].view(B, k - 1, 1)
            try:
                w = torch.linalg.solve_triangular(L, G_stack, upper=False)
            except AttributeError:
                w = torch.triangular_solve(G_stack, L, upper=False).solution
            w = w.view(B, 1, k - 1)
            diag_g = G[index, index].view(B, 1, 1)
            # More robust Cholesky corner computation
            corner_val = diag_g - (w ** 2).sum(dim=2, keepdim=True)
            w_corner = torch.sqrt(torch.clamp(corner_val, min=diag_eps))
            
            k_zeros = torch.zeros(B, k - 1, 1, device=X.device, dtype=X.dtype)
            L = torch.cat((torch.cat((L, k_zeros), dim=2), torch.cat((w, w_corner), dim=2)), dim=1)
            
        I = torch.cat([I, index.unsqueeze(1)], dim=1)
        h_stack = h_bar[expanded_batch_idx, I[batch_idx, :]].view(B, k, 1)
        # Solve
        try:
            x_stack = torch.cholesky_solve(h_stack, L)
        except Exception:
            # Fallback: if cholesky_solve fails, break
            break
        
        # Check for NaN before assignment
        if not torch.isfinite(x_stack).all():
            break
        
        x[batch_idx.unsqueeze(1), I[batch_idx]] = x_stack.squeeze(-1)
        
        # Residual update
        beta = x[batch_idx.unsqueeze(1), I[batch_idx]].unsqueeze(1).bmm(G[I[batch_idx], :]).squeeze(1)
        h = h_bar - beta
        new_delta = (x * beta).sum(dim=1)
        eps = eps + delta - new_delta
        delta = new_delta
        
        if not torch.isfinite(x).all():
            # Reset to last good state (zeros for problematic entries)
            x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
            break
            
    return x # [B, N]

def compute_lipschitz_constant(D, num_iters=10):
    """Approximates L = ||D||_2^2 using Power Iteration"""
    n_features = D.shape[1]
    v = torch.randn(n_features, 1, device=D.device)
    v = v / (torch.norm(v) + 1e-8)
    
    for _ in range(num_iters):
        # Iteration: v <- D.t @ (D @ v)
        v = D.t() @ (D @ v)
        v = v / (torch.norm(v) + 1e-8)
        
    # L = ||D v||^2
    L = torch.norm(D @ v) ** 2
    return L.item()

def soft_thresholding(x, threshold):
    return torch.sign(x) * torch.relu(torch.abs(x) - threshold)

def ista_solve(Y, D, lambda_l1, num_iters=50, lr=None):
    batch_size = Y.shape[0]
    n_atoms = D.shape[0]
    X = torch.zeros(batch_size, n_atoms, device=Y.device)
    
    # --- OPTIMIZATION: Power Iteration ---
    if lr is None:
        L = compute_lipschitz_constant(D, num_iters=5) # 5 iters is usually enough
        lr = 1.0 / (L + 1e-6) 
        
    for _ in range(num_iters):
        residual = Y - (X @ D)
        grad_X = -residual @ D.t()
        X = X - lr * grad_X 
        X = soft_thresholding(X, lambda_l1 * lr)
    return X

# ==========================================
# 2. DICTIONARY MODEL & PATCH HANDLER
# ==========================================

class DictionaryLearning(nn.Module):
    def __init__(self, n_features, n_atoms, sparsity_level=10):
        super().__init__()
        # Use Xavier initialization for better numerical stability
        self.D = nn.Parameter(torch.empty(n_atoms, n_features))
        nn.init.xavier_normal_(self.D)
        self.sparsity_level = sparsity_level
        self.normalize_dictionary()

    def normalize_dictionary(self):
        with torch.no_grad():
            norms = torch.norm(self.D, p=2, dim=1, keepdim=True)
            # Avoid division by very small norms
            norms = torch.clamp(norms, min=1e-6)
            self.D.div_(norms)

    def forward(self, Y):
        with torch.no_grad():
            # Y: [B, F]. D: [A, F].
            # batch_omp takes D as [F, A]. So we pass self.D.t()
            X = batch_omp(Y, self.D.t(), k_max=self.sparsity_level)
        Y_pred = X @ self.D 
        return Y_pred, X

class PatchHandler:
    def __init__(self, img_shape, patch_size, stride, channels=3):
        self.H, self.W = img_shape
        self.C = channels
        self.patch_size = patch_size
        self.stride = stride
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=stride)
        self.fold = nn.Fold(output_size=(self.H, self.W), kernel_size=patch_size, stride=stride)
        
        dummy_ones = torch.ones(1, self.C, self.H, self.W)
        self.normalization_mask = self.fold(self.unfold(dummy_ones))
        self.normalization_mask = torch.clamp(self.normalization_mask, min=1.0)

    def image_to_patches(self, image):
        patches = self.unfold(image) 
        patches = patches.permute(0, 2, 1).reshape(-1, self.C * self.patch_size**2)
        # Subtract Mean
        patch_means = patches.mean(dim=1, keepdim=True)
        patches_centered = patches - patch_means
        return patches_centered, patch_means

    def patches_to_image(self, patches_centered, patch_means, batch_size=1):
        # Add Mean Back
        patches = patches_centered + patch_means
        n_patches = patches.shape[0] // batch_size
        patches = patches.view(batch_size, n_patches, -1).permute(0, 2, 1)
        summed = self.fold(patches)
        return summed / self.normalization_mask.to(patches.device)

# ==========================================
# 3. EXPERIMENT LOOP
# ==========================================

def run_fast_experiment():
    print(f"Running Optimized CelebA on {DEVICE}...")
    
    # Config
    IMAGE_SIZE = 64
    PATCH_SIZE = 16
    STRIDE = 8
    N_FEATURES = 3 * PATCH_SIZE**2 
    # Allow env overrides; fewer atoms by default to reduce garbage
    N_ATOMS = int(os.environ.get("N_ATOMS", "512"))
    SPARSITY_LEVEL = int(os.environ.get("SPARSITY_LEVEL", "20"))
    # Regularization (can override via env)
    LAMBDA_COH = float(os.environ.get("LAMBDA_COH", "1e-3"))   # mutual coherence
    LAMBDA_L2 = float(os.environ.get("LAMBDA_L2", "1e-4"))    # weight decay-like
    # Train longer by default; override via env for quick smoke tests
    N_EPOCHS = int(os.environ.get("N_EPOCHS", "300"))
    
    # Load Data
    print("1. Loading Data...")
    # Use user's home directory for data
    data_root = os.path.expanduser("~/Data")
    
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor()
    ])
    
    # CelebA might fail if not downloaded, but we assume it's in ~/Data
    try:
        dataset = torchvision.datasets.CelebA(root=data_root, split='train', download=True, transform=transform)
    except RuntimeError:
        print("CelebA not found or download failed. Trying to proceed if data exists...")
        dataset = torchvision.datasets.CelebA(root=data_root, split='train', download=False, transform=transform)
    
    # Use a subset for speed; bump default for better atoms
    subset_n = int(os.environ.get("SUBSET_N", "400"))
    subset = torch.utils.data.Subset(dataset, range(subset_n))
    dataloader = torch.utils.data.DataLoader(subset, batch_size=10, shuffle=True)
    
    # Extract Patches
    print("2. Extracting Patches...")
    patcher = PatchHandler((IMAGE_SIZE, IMAGE_SIZE), PATCH_SIZE, STRIDE, channels=3)
    
    all_patches = []
    for imgs, _ in dataloader:
        patches_c, _ = patcher.image_to_patches(imgs.to(DEVICE))
        all_patches.append(patches_c)
    
    train_data = torch.cat(all_patches, dim=0)
    train_data = train_data[torch.randperm(train_data.shape[0])]
    print(f"   Training on {train_data.shape[0]} patches.")

    # Train
    print("3. Training (OMP)...")
    model = DictionaryLearning(N_FEATURES, N_ATOMS, SPARSITY_LEVEL).to(DEVICE)
    
    # Data-driven initialization to reduce garbage atoms
    with torch.no_grad():
        n_seed = min(N_ATOMS, train_data.shape[0])
        seed_idx = torch.randperm(train_data.shape[0], device=train_data.device)[:n_seed]
        seeds = train_data[seed_idx]
        seeds = seeds / seeds.norm(dim=1, keepdim=True).clamp(min=1e-6)
        model.D.data[:n_seed] = seeds
        if n_seed < N_ATOMS:
            nn.init.xavier_normal_(model.D.data[n_seed:])
        model.normalize_dictionary()
    print(f"   Seeded {n_seed} atoms from data patches.")
    
    # Slightly higher LR with weight decay; cosine annealing helps escape plateaus
    optimizer = optim.Adam([model.D], lr=0.002, weight_decay=1e-4)
    
    # Cosine schedule without upswing (one full decay over N_EPOCHS)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=3e-4)
    
    # Track atom usage
    atom_usage = torch.zeros(N_ATOMS, device=DEVICE)
    
    for epoch in range(N_EPOCHS):
        batch_size = 256  # Smaller batch size for stability
        epoch_loss = 0
        n_batches = 0
        atom_usage.zero_() # Reset usage counter for this epoch
        
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            optimizer.zero_grad()
            
            # Forward uses OMP
            batch_recon, X = model(batch)
            
            # Update usage stats
            # X is [Batch, Atoms]
            atom_usage += (X != 0).float().sum(dim=0)
            
            # Compute reconstruction loss
            recon_loss = 0.5 * ((batch - batch_recon) ** 2).sum() / batch.shape[0]

            # Dictionary regularization: mutual coherence + mild L2
            D_norm = model.D
            # Off-diagonal energy of D D^T encourages diverse atoms
            gram = D_norm @ D_norm.t()
            off_diag = gram - torch.eye(N_ATOMS, device=D_norm.device, dtype=D_norm.dtype)
            coherence_loss = (off_diag ** 2).mean()
            l2_loss = (D_norm ** 2).mean()

            loss = recon_loss + LAMBDA_COH * coherence_loss + LAMBDA_L2 * l2_loss
            
            # Skip if loss is NaN
            if not torch.isfinite(loss):
                print(f"   Warning: NaN loss at epoch {epoch+1}, batch {i}. Skipping...")
                # Reset dictionary if corrupted
                if not torch.isfinite(model.D).all():
                    print("   Reinitializing dictionary...")
                    nn.init.xavier_normal_(model.D)
                    model.normalize_dictionary()
                continue
            
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_([model.D], max_norm=1.0)
            
            optimizer.step()
            model.normalize_dictionary()
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        
        # Dead Atom Revival - more aggressive
        # Revival every 3 epochs early on, then every 10 epochs
        revival_freq = 3 if epoch < 100 else 10
        if (epoch + 1) % revival_freq == 0 and (epoch + 1) < 450:
            # More aggressive threshold: atoms used less than average of ~1 per batch
            expected_usage = len(train_data) / batch_size * SPARSITY_LEVEL / N_ATOMS
            dead_threshold = max(expected_usage * 0.1, 1)  # At least 10% of expected usage
            dead_indices = torch.where(atom_usage < dead_threshold)[0]
            
            if len(dead_indices) > 0:
                print(f"   Reviving {len(dead_indices)} dead atoms...")
                with torch.no_grad():
                    # Use ALL training data to find high-error patches
                    # Process in chunks to avoid OOM
                    all_errors = []
                    chunk_size = 512
                    for j in range(0, len(train_data), chunk_size):
                        chunk = train_data[j:j+chunk_size]
                        chunk_recon, _ = model(chunk)
                        chunk_errors = (chunk - chunk_recon).norm(dim=1)
                        all_errors.append(chunk_errors)
                    all_errors = torch.cat(all_errors)
                    
                    # Sort indices by error descending
                    sorted_indices = torch.argsort(all_errors, descending=True)
                    
                    # Take top high-error patches
                    n_dead = len(dead_indices)
                    top_error_indices = sorted_indices[:n_dead * 2]  # Get 2x candidates
                    
                    # Randomly sample from top candidates to add diversity
                    perm = torch.randperm(len(top_error_indices))[:n_dead]
                    selected_indices = top_error_indices[perm]
                    
                    # If we need more, cycle them
                    if len(selected_indices) < n_dead:
                        selected_indices = selected_indices.repeat(
                            (n_dead + len(selected_indices) - 1) // len(selected_indices)
                        )[:n_dead]
                        
                    replacements = train_data[selected_indices]
                    
                    # Add small noise for diversity
                    noise = torch.randn_like(replacements) * 0.1
                    replacements = replacements + noise
                    
                    # Normalize before inserting
                    replacements = replacements / (replacements.norm(dim=1, keepdim=True).clamp(min=1e-6))
                    
                    model.D.data[dead_indices] = replacements
        
        sparsity = (X.abs() < 1e-4).float().mean() * 100
        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"   Epoch {epoch+1}: Loss = {avg_loss:.4f} | Sparsity: {sparsity:.1f}%")

    # Visualize Atoms
    print("4. Saving Atoms...")
    atoms = model.D.detach().cpu().numpy()
    plt.figure(figsize=(12, 12))
    plt.suptitle("Learned Atoms (CelebA - OMP)")
    for i in range(100):
        plt.subplot(10, 10, i+1)
        atom = atoms[i].reshape(3, PATCH_SIZE, PATCH_SIZE)
        atom = np.transpose(atom, (1, 2, 0))
        atom = (atom - atom.min()) / (atom.max() - atom.min() + 1e-8)
        plt.imshow(atom)
        plt.axis('off')
    plt.savefig("atoms_celeba.png")

    # Reconstruction
    print("5. Reconstructing Test Image...")
    test_img, _ = subset[0]
    test_img = test_img.unsqueeze(0).to(DEVICE)
    # noisy_img = test_img + 0.1 * torch.randn_like(test_img) # Removed noise
    
    # Process
    patches_c, means = patcher.image_to_patches(test_img) # Use clean image
    with torch.no_grad():
        X_test = batch_omp(patches_c, model.D.t(), k_max=SPARSITY_LEVEL)
        clean_patches_c = X_test @ model.D
    
    recon_img = patcher.patches_to_image(clean_patches_c, means)
    
    # Plot
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1); plt.title("Original"); plt.imshow(test_img.squeeze().cpu().permute(1, 2, 0)); plt.axis('off')
    # plt.subplot(1, 3, 2); plt.title("Noisy"); plt.imshow(torch.clamp(noisy_img, 0, 1).squeeze().cpu().permute(1, 2, 0)); plt.axis('off')
    plt.subplot(1, 2, 2); plt.title("Reconstructed"); plt.imshow(torch.clamp(recon_img, 0, 1).squeeze().cpu().permute(1, 2, 0)); plt.axis('off')
    plt.savefig("reconstruction_celeba.png")
    print("Done.")

if __name__ == "__main__":
    run_fast_experiment()