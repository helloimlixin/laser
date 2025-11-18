import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    Vector Quantizer implementation for VQ-VAE.

    The Vector Quantizer maps continuous encodings to discrete codes from a learned codebook.
    This is the key component that enables VQ-VAE to learn discrete representations.
    """

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        commitment_cost=0.25,
        decay=0.0,
        epsilon=1e-5,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

        if self.decay > 0:
            self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
            self.register_buffer("_ema_w", torch.zeros(num_embeddings, embedding_dim))
            self._ema_w.data.copy_(self.embedding.weight.detach())

    def forward(self, z_e: torch.Tensor):
        """
        Args:
            z_e: Input tensor of shape [B, D, H, W]

        Returns:
            Tuple of (quantized tensor, loss, perplexity, one-hot encodings)
        """
        # Permute to [B, H, W, D] for convenience and flatten to [N, D]
        z = z_e.permute(0, 2, 3, 1).contiguous()
        flat_inputs = z.view(-1, self.embedding_dim)
        num_vectors = flat_inputs.size(0)

        embeddings = self.embedding.weight
        emb_norm_sq = (embeddings ** 2).sum(dim=1)

        # Chunked nearest neighbour search
        chunk_size = 32768
        indices_list = []
        for start in range(0, num_vectors, chunk_size):
            end = min(start + chunk_size, num_vectors)
            x = flat_inputs[start:end]
            x_norm_sq = (x ** 2).sum(dim=1, keepdim=True)
            distances = x_norm_sq + emb_norm_sq.unsqueeze(0) - 2.0 * x @ embeddings.t()
            indices_list.append(torch.argmin(distances, dim=1))
        encoding_indices = torch.cat(indices_list, dim=0)  # [N]

        # One-hot encodings and quantized latents
        encodings = F.one_hot(encoding_indices, self.num_embeddings).to(flat_inputs.dtype)
        z_q_flat = encodings @ embeddings  # [N, D]
        z_q = z_q_flat.view_as(z)

        # Loss terms
        e_latent_loss = F.mse_loss(z_q.detach(), z)
        q_latent_loss = F.mse_loss(z_q, z.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator
        z_q = z_q.to(z.dtype)
        z_st = z + (z_q - z).detach()

        # Perplexity (codebook usage)
        encoding_sum = encodings.float().sum(dim=0)
        avg_probs = encoding_sum / float(num_vectors)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # EMA updates when enabled and in training mode
        if self.training and self.decay > 0:
            self._ema_cluster_size = (
                self._ema_cluster_size * self.decay
                + (1 - self.decay) * encoding_sum
            )

            ema_w = encodings.float().t() @ flat_inputs.float()
            self._ema_w = self._ema_w * self.decay + (1 - self.decay) * ema_w

            n = self._ema_cluster_size.sum()
            cluster_size = (
                (self._ema_cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon)
                * n
            )
            self.embedding.weight.data.copy_(
                (self._ema_w / cluster_size.unsqueeze(1)).to(self.embedding.weight.data.dtype)
            )

        encodings = encodings.view(num_vectors, self.num_embeddings)
        return z_st.permute(0, 3, 1, 2).contiguous(), loss, perplexity, encodings


class DictionaryLearning(nn.Module):
    """
    Dictionary Learning Bottleneck with vectorized Batch Orthogonal Matching Pursuit (OMP) sparse coding.
    Uses a simplified greedy OMP implementation for faster and cleaner sparse coding.
    """
    def __init__(
        self,
        num_embeddings=512,
        embedding_dim=64,
        sparsity_level=5,
        commitment_cost=0.25,
        decay=0.99,
        epsilon=1e-10,
        use_ema=True,
        normalize_atoms=True,
        patch_size=1,
    ):
        super(DictionaryLearning, self).__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.sparsity_level = sparsity_level
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.use_ema = use_ema
        self.normalize_atoms = normalize_atoms
        if isinstance(patch_size, int):
            self.patch_size = (patch_size, patch_size)
        else:
            if len(patch_size) != 2:
                raise ValueError("patch_size must be an int or a tuple of two ints")
            self.patch_size = tuple(int(p) for p in patch_size)
        self.patch_area = self.patch_size[0] * self.patch_size[1]
        self.atom_dim = self.embedding_dim * self.patch_area
        
        # Initialize dictionary with random atoms shaped for flattened patches
        self.dictionary = nn.Parameter(
            torch.randn(self.atom_dim, num_embeddings), requires_grad=True
        )
        if self.normalize_atoms:
            self._normalize_dictionary()

    def _normalize_dictionary(self):
        """Normalize all dictionary atoms to have unit L2 norm."""
        with torch.no_grad():
            norms = torch.linalg.norm(self.dictionary.data, dim=0, keepdim=True)
            self.dictionary.data = self.dictionary.data / (norms + 1e-10)
    
    def orthogonality_loss(self):
        """
        Encourage dictionary atoms to be orthogonal by penalizing off-diagonal Gram entries.
        Always evaluates atoms in normalized space so the penalty is scale-invariant.
        """
        dict_norm = F.normalize(self.dictionary, dim=0)
        gram = dict_norm.t() @ dict_norm  # [num_embeddings, num_embeddings]
        eye = torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)
        loss = (gram - eye).pow(2)
        return loss.mean()

    def batch_omp(self, X, D):
        """
        Batched Orthogonal Matching Pursuit (greedy selection only).
        
        Fast implementation using greedy atom selection without LS refinement.
        Simpler and faster than Cholesky-based OMP.

        Args:
            X (torch.Tensor): Input signals of shape (M, B).
            D (torch.Tensor): Dictionary of shape (M, N), where each column is an atom of dimension M.

        Returns:
            coefficients: (N, B) Tensor with the corresponding coefficients.
        """
        M, B = X.shape
        _, N = D.shape
        device = X.device

        # Vectorized greedy atom selection
        coefficients = torch.zeros(N, B, device=device, dtype=X.dtype)
        residual = X.clone()
        
        # Mask to prevent reselecting the same atom per signal (N, B) format
        mask = torch.ones(N, B, device=device, dtype=X.dtype)  # Use float for efficient multiply
        batch_idx = torch.arange(B, device=device)
        
        for k in range(self.sparsity_level):
            # Compute correlations (keep in N, B format - avoid transpose)
            correlations = torch.mm(D.t(), residual)  # (N, B)
            abs_corr = torch.abs(correlations)
            
            # Apply mask and find argmax (avoid transpose by using dim=0)
            abs_corr_masked = abs_corr * mask  # (N, B) - already in right shape!
            idx = torch.argmax(abs_corr_masked, dim=0)  # (B,)
            
            # Update mask (vectorized) - set selected atoms to 0
            mask[idx, batch_idx] = 0.0
            
            # Gather selected atoms
            d_selected = D[:, idx]  # (M, B)
            
            # Compute coefficients using projection
            numerator = (residual * d_selected).sum(dim=0)  # (B,)
            denominator = (d_selected ** 2).sum(dim=0)  # (B,)
            alpha = numerator / (denominator + self.epsilon)  # (B,)
            
            # Update coefficient matrix
            coefficients[idx, batch_idx] = alpha
            
            # Update residual (not in-place to preserve gradients)
            residual = residual - d_selected * alpha.unsqueeze(0)
        
        return coefficients
    
    def forward(self, z_e):
        """
        Forward pass through the dictionary learning bottleneck.
        
        Args:
            z_e: Input tensor of shape [batch_size, embedding_dim, height, width]
            
        Returns:
            z_dl: latent representation reconstructed from the dictionary learning bottleneck
            loss: loss from the dictionary learning bottleneck
            coefficients: sparse coefficients
        """
        # z shape: [batch_size, embedding_dim, height, width]
        z_e = z_e.contiguous()
        batch_size, channels, height, width = z_e.shape
        patch_h, patch_w = self.patch_size
        if height % patch_h != 0 or width % patch_w != 0:
            raise ValueError(
                f"Feature map ({height}x{width}) must be divisible by patch_size {self.patch_size}"
            )
        # Extract non-overlapping patches using the same layout that PyTorch fold/unfold expect
        patches = F.unfold(
            z_e,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        ).permute(2, 0, 1).contiguous()  # [num_patches_per_sample, batch_size, atom_dim]
        patches_shape = (patches.size(0), patches.size(1), patches.size(2))
        patch_tokens = patches.view(-1, self.atom_dim).t().contiguous()  # [atom_dim, batch_size * num_patches_per_sample]
        
        '''
        Sparse coding stage
        '''
        # Run OMP in float32 to avoid AMP dtype mismatches; cast back after
        orig_dtype = patch_tokens.dtype
        with torch.amp.autocast('cuda', enabled=False):
            patch_tokens_f32 = patch_tokens.to(torch.float32)
            dictionary_f32 = self.dictionary.to(torch.float32)
            atom_norms = torch.linalg.norm(dictionary_f32, dim=0).clamp_min(1e-10)
            dict_for_omp = dictionary_f32 / atom_norms.unsqueeze(0)
            dict_for_recon = dict_for_omp if self.normalize_atoms else dictionary_f32
            
            # Use BatchOMP to compute sparse coefficients
            coefficients = self.batch_omp(patch_tokens_f32, dict_for_omp)
            if not self.normalize_atoms:
                coefficients = coefficients / atom_norms.unsqueeze(1)
            z_dl_f32 = dict_for_recon @ coefficients

        # Reshape flattened patch reconstructions back to feature maps
        z_dl_patches = (
            z_dl_f32.t()
            .view(*patches_shape)
            .permute(1, 2, 0)
            .contiguous()
        )  # [batch_size, atom_dim, num_patches_per_sample]
        z_dl_nchw = F.fold(
            z_dl_patches,
            output_size=(height, width),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        z_dl = z_dl_nchw.to(orig_dtype)  # cast back to original dtype (could be fp16 under AMP)

        # Compute the commitment loss
        z_e_nhwc = z_e.permute(0, 2, 3, 1).contiguous()
        z_dl_nhwc = z_dl.permute(0, 2, 3, 1).contiguous()
        e_latent_loss = F.mse_loss(z_dl_nhwc.detach(), z_e_nhwc)
        dl_latent_loss = F.mse_loss(z_dl_nhwc, z_e_nhwc.detach())

        # Compute the total loss
        loss = self.commitment_cost * e_latent_loss + dl_latent_loss

        # Straight-through estimator
        z_dl = z_e + (z_dl - z_e).detach()  # Allow gradients to flow back to encoder

        return z_dl, loss, coefficients  # Return the reconstructed latent representation, loss, and sparse coefficients


class KSVDDictionaryLearning(nn.Module):
    """
    K-SVD based Dictionary Learning Bottleneck.
    
    K-SVD is a classical dictionary learning algorithm that alternates between:
    1. Sparse coding: Find sparse coefficients for given dictionary (using OMP)
    2. Dictionary update: Update each atom and its coefficients using SVD
    
    The key difference from regular DL is that K-SVD updates atoms sequentially
    using rank-1 SVD approximations, which can lead to better atom learning.
    
    Reference: "K-SVD: An Algorithm for Designing Overcomplete Dictionaries 
    for Sparse Representation" by Aharon, Elad, and Bruckstein (2006)
    """
    
    def __init__(
        self,
        num_embeddings=512,
        embedding_dim=64,
        sparsity_level=5,
        commitment_cost=0.25,
        ksvd_iterations=1,
        normalize_atoms=True,
        patch_size=1,
        epsilon=1e-10,
        sparse_solver="omp",
        iht_iterations=10,
        iht_step_size=None,
        use_online_learning=True,
        dict_learning_rate=0.1,
        use_backprop_only=False,
    ):
        super(KSVDDictionaryLearning, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.sparsity_level = sparsity_level
        self.commitment_cost = commitment_cost
        self.ksvd_iterations = ksvd_iterations
        self.normalize_atoms = normalize_atoms
        self.epsilon = epsilon
        self.sparse_solver = sparse_solver.lower()
        self.iht_iterations = iht_iterations
        self.iht_step_size = iht_step_size
        
        if isinstance(patch_size, int):
            self.patch_size = (patch_size, patch_size)
        else:
            if len(patch_size) != 2:
                raise ValueError("patch_size must be an int or a tuple of two ints")
            self.patch_size = tuple(int(p) for p in patch_size)
        self.patch_area = self.patch_size[0] * self.patch_size[1]
        self.atom_dim = self.embedding_dim * self.patch_area
        
        # Use backprop-only mode for faster training
        self.use_backprop_only = use_backprop_only
        
        # Initialize dictionary with random atoms
        if self.use_backprop_only:
            # For backprop-only: make dictionary trainable and skip K-SVD updates
            self.dictionary = nn.Parameter(
                torch.randn(self.atom_dim, num_embeddings), requires_grad=True
            )
            self.enable_ksvd_update = False
        else:
            # Original K-SVD approach: dictionary not trainable, updated via K-SVD
            self.dictionary = nn.Parameter(
                torch.randn(self.atom_dim, num_embeddings), requires_grad=False
            )
            self.enable_ksvd_update = True
            
        if self.normalize_atoms and not self.use_backprop_only:
            self._normalize_dictionary()

        # Online dictionary learning parameters (only used if not backprop_only)
        self.dict_learning_rate = dict_learning_rate
        self.use_online_learning = use_online_learning
    
    def _normalize_dictionary(self):
        """Normalize all dictionary atoms to have unit L2 norm."""
        with torch.no_grad():
            norms = torch.linalg.norm(self.dictionary.data, dim=0, keepdim=True)
            self.dictionary.data = self.dictionary.data / (norms + self.epsilon)
    
    def batch_omp(self, X, D):
        """
        Batched Orthogonal Matching Pursuit with least-squares refinement.
        
        Args:
            X: Input signals of shape (M, B) where M is atom dimension, B is batch size.
            D: Dictionary of shape (M, N) with column-normalized atoms.
        
        Returns:
            coefficients: Sparse coefficient matrix of shape (N, B).
        """
        M, B = X.shape
        _, N = D.shape
        device = X.device

        coefficients = torch.zeros(N, B, device=device, dtype=X.dtype)
        selected_atoms_list = []  # Track selected atoms as list to avoid inplace modifications
        selected_mask = torch.zeros(N, B, device=device, dtype=torch.bool)
        batch_idx = torch.arange(B, device=device)

        gram = torch.matmul(D.t(), D)  # (N, N)
        dt_x = torch.matmul(D.t(), X)  # (N, B)

        for k in range(self.sparsity_level):
            residual = X - torch.matmul(D, coefficients)
            correlations = torch.matmul(D.t(), residual)
            correlations = correlations.masked_fill(selected_mask, 0.0)

            idx = torch.argmax(torch.abs(correlations), dim=0)
            selected_atoms_list.append(idx)
            # Create a new mask instead of modifying inplace to avoid breaking autograd
            new_mask = selected_mask.clone()
            new_mask[idx, batch_idx] = True
            selected_mask = new_mask

            # Stack selected atoms from list
            active = torch.stack(selected_atoms_list, dim=1)  # [B, k+1]

            gram_rows = gram.index_select(0, active.reshape(-1)).view(B, k + 1, N)
            gram_active = torch.gather(
                gram_rows,
                2,
                active.unsqueeze(1).expand(B, k + 1, k + 1),
            )
            diag_eps = max(self.epsilon, 1e-6)
            eye = torch.eye(k + 1, device=device, dtype=gram_active.dtype).unsqueeze(0)
            gram_active = gram_active + diag_eps * eye

            rhs = torch.gather(dt_x.t(), 1, active).unsqueeze(-1)  # [B, k+1, 1]
            try:
                solutions = torch.linalg.solve(gram_active, rhs).squeeze(-1)
            except RuntimeError:
                pinv = torch.linalg.pinv(gram_active)
                solutions = torch.matmul(pinv, rhs).squeeze(-1)

            # Update coefficients without inplace operation
            row_idx = active.transpose(0, 1)  # [k+1, B]
            col_idx = batch_idx.unsqueeze(0).expand_as(row_idx)
            # Create new coefficients tensor instead of modifying inplace
            coefficients = coefficients.clone()
            coefficients[row_idx, col_idx] = solutions.transpose(0, 1)

        return coefficients

    def _sparse_encode(self, tokens, dictionary):
        if self.sparse_solver == "iht":
            return self.iterative_hard_thresholding(tokens, dictionary)
        elif self.sparse_solver == "topk":
            return self.topk_sparse_coding(tokens, dictionary)
        return self.batch_omp(tokens, dictionary)

    def topk_sparse_coding(self, X, D):
        """
        Fast approximate sparse coding using top-k selection.

        Much faster than IHT/OMP: single matrix multiplication + top-k selection.
        Good approximation when dictionary atoms are well-learned.

        Args:
            X: Input signals of shape (M, B)
            D: Dictionary of shape (M, N) with normalized columns

        Returns:
            coefficients: (N, B) with top-k non-zero entries
        """
        M, B = X.shape
        _, N = D.shape
        device = X.device

        # Single matrix multiplication to get all correlations
        correlations = D.t() @ X  # (N, B)

        # Keep only top-k by absolute value
        abs_corr = torch.abs(correlations)
        topk_vals, topk_idx = torch.topk(abs_corr, self.sparsity_level, dim=0)

        # Create sparse coefficient matrix
        coefficients = torch.zeros(N, B, device=device, dtype=X.dtype)
        batch_idx = torch.arange(B, device=device).unsqueeze(0).expand(self.sparsity_level, -1)

        # Gather original correlation values (with sign)
        coefficients[topk_idx, batch_idx] = torch.gather(correlations, 0, topk_idx)

        return coefficients
    
    def _hard_threshold(self, coef):
        """Keep only the top-k coefficients (by magnitude) per sample."""
        k = self.sparsity_level
        if k >= coef.size(0):
            return coef
        abs_coef = torch.abs(coef)
        topk = torch.topk(abs_coef, k, dim=0)
        mask = torch.zeros_like(coef, dtype=torch.bool)
        mask.scatter_(0, topk.indices, True)
        return coef * mask

    def iterative_hard_thresholding(self, X, D):
        """
        Iterative Hard Thresholding (IHT) sparse coding.

        Args:
            X: Input signals of shape (M, B)
            D: Dictionary of shape (M, N)

        Returns:
            coefficients: (N, B)
        """
        M, B = X.shape
        _, N = D.shape
        device = X.device

        coefficients = torch.zeros(N, B, device=device, dtype=X.dtype)
        Dt = D.t()

        if self.iht_step_size is not None:
            step_size = self.iht_step_size
        else:
            # Estimate Lipschitz constant via spectral norm of D and pick conservative step
            try:
                spectral = torch.linalg.norm(D, ord=2)
            except RuntimeError:
                spectral = torch.sqrt(torch.linalg.eigvals(D.t() @ D).real.max())
            step_size = 1.0 / (spectral.pow(2) + self.epsilon)

        for _ in range(self.iht_iterations):
            residual = X - D @ coefficients
            gradient = Dt @ residual
            coefficients = coefficients + step_size * gradient
            coefficients = self._hard_threshold(coefficients)

        return coefficients

    def ksvd_update(self, X, coefficients):
        """
        K-SVD dictionary update step using residual recycling to avoid repeated clones.
        """
        M, _ = X.shape
        N = self.num_embeddings

        with torch.no_grad():
            residual = X - self.dictionary @ coefficients
            for k in range(N):
                omega_k = torch.abs(coefficients[k, :]) > self.epsilon

                if not omega_k.any():
                    self.dictionary.data[:, k] = torch.randn(
                        M, device=X.device, dtype=X.dtype
                    )
                    if self.normalize_atoms:
                        self.dictionary.data[:, k] /= (
                            torch.linalg.norm(self.dictionary.data[:, k]) + self.epsilon
                        )
                    continue

                current_atom = self.dictionary[:, k : k + 1]
                coeff_slice = coefficients[k : k + 1, omega_k]
                E_k = residual[:, omega_k] + current_atom @ coeff_slice

                if E_k.numel() == 0:
                    continue

                try:
                    E_k_f32 = E_k.float()
                    U, S, Vt = torch.linalg.svd(E_k_f32, full_matrices=False)
                    new_atom = U[:, 0].to(self.dictionary.dtype)
                    new_coeff = (S[0] * Vt[0, :]).to(coefficients.dtype)
                except RuntimeError:
                    continue

                residual[:, omega_k] = E_k - new_atom.unsqueeze(1) * new_coeff.unsqueeze(0)
                self.dictionary.data[:, k] = new_atom
                coefficients[k, omega_k] = new_coeff

            if self.normalize_atoms:
                self._normalize_dictionary()
    
    def online_dict_update(self, X, coefficients):
        """
        Online dictionary learning update using gradient-like updates.
        
        Fast vectorized implementation that updates all atoms in parallel.
        
        Args:
            X: Input signals of shape (M, B)
            coefficients: Current sparse coefficients of shape (N, B)
        """
        with torch.no_grad():
            # Compute reconstruction and residual
            recon = self.dictionary @ coefficients
            residual = X - recon
            
            # Vectorized update: D += lr * (residual @ coefficients.T)
            # This computes the gradient for all atoms at once
            # Shape: (M, B) @ (B, N) = (M, N)
            gradient = residual @ coefficients.t()
            
            # Normalize by usage count (number of non-zero coefficients per atom)
            usage_counts = (torch.abs(coefficients) > 1e-6).sum(dim=1, keepdim=True).t()  # (1, N)
            usage_counts = usage_counts.clamp_min(1.0)  # Avoid division by zero
            
            # Update all atoms at once
            self.dictionary.data += self.dict_learning_rate * gradient / usage_counts
            
            # Normalize dictionary atoms if required
            if self.normalize_atoms:
                self._normalize_dictionary()
    
    def forward(self, z_e):
        """
        Forward pass through K-SVD dictionary learning bottleneck.
        
        Args:
            z_e: Input tensor of shape [batch_size, embedding_dim, height, width]
        
        Returns:
            z_q: Quantized representation with straight-through gradients
            loss: Reconstruction + commitment loss
            coefficients: Sparse coefficients (num_embeddings, batch_size * num_patches)
        """
        z_e = z_e.contiguous()
        batch_size, channels, height, width = z_e.shape
        patch_h, patch_w = self.patch_size
        
        if height % patch_h != 0 or width % patch_w != 0:
            raise ValueError(
                f"Feature map ({height}x{width}) must be divisible by patch_size {self.patch_size}"
            )
        
        # Extract patches
        patches = F.unfold(
            z_e,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        ).permute(2, 0, 1).contiguous()  # [num_patches, batch_size, atom_dim]
        
        patches_shape = (patches.size(0), patches.size(1), patches.size(2))
        patch_tokens = patches.view(-1, self.atom_dim).t().contiguous()  # [atom_dim, batch_size * num_patches]
        
        # Store original dtype for consistency
        orig_dtype = patch_tokens.dtype
        
        if self.use_backprop_only:
            # Backprop-only mode: use full dictionary representation (no sparsity)
            # This is much faster but gives up sparsity constraint
            coefficients = torch.matmul(self.dictionary.t(), patch_tokens)  # [num_embeddings, batch_size * num_patches]
            z_dl_f32 = torch.matmul(self.dictionary, coefficients)  # [atom_dim, batch_size * num_patches]
            
            # Convert back to original dtype
            coefficients = coefficients.to(orig_dtype)
        else:
            # Original K-SVD sparse coding approach
            # Sparse coding stage (OMP) - operate in float32 for numerical stability
            orig_dtype = patch_tokens.dtype
            with torch.amp.autocast('cuda', enabled=False):
                patch_tokens_f32 = patch_tokens.to(torch.float32)
                dictionary_f32 = self.dictionary.to(torch.float32)

                atom_norms = torch.linalg.norm(dictionary_f32, dim=0).clamp_min(self.epsilon)
                dict_normalized = dictionary_f32 / atom_norms.unsqueeze(0)

                coefficients = self._sparse_encode(patch_tokens_f32, dict_normalized)

                if self.training and self.enable_ksvd_update:
                    if self.use_online_learning:
                        with torch.no_grad():
                            self.online_dict_update(patch_tokens_f32, coefficients.detach())
                        dictionary_f32 = self.dictionary.to(torch.float32)
                        atom_norms = torch.linalg.norm(dictionary_f32, dim=0).clamp_min(self.epsilon)
                        dict_normalized = dictionary_f32 / atom_norms.unsqueeze(0)
                        coefficients = self._sparse_encode(patch_tokens_f32, dict_normalized)
                    else:
                        for _ in range(self.ksvd_iterations):
                            with torch.no_grad():
                                self.ksvd_update(patch_tokens_f32, coefficients.detach())
                            dictionary_f32 = self.dictionary.to(torch.float32)
                            atom_norms = torch.linalg.norm(dictionary_f32, dim=0).clamp_min(self.epsilon)
                            dict_normalized = dictionary_f32 / atom_norms.unsqueeze(0)
                            coefficients = self._sparse_encode(patch_tokens_f32, dict_normalized)

                dict_for_recon = dict_normalized if self.normalize_atoms else dictionary_f32
                z_dl_f32 = dict_for_recon @ coefficients

                coefficients = coefficients.to(orig_dtype)
        
        # Reshape back to feature maps
        z_dl_patches = (
            z_dl_f32.t()
            .view(*patches_shape)
            .permute(1, 2, 0)
            .contiguous()
        )
        
        z_dl_nchw = F.fold(
            z_dl_patches,
            output_size=(height, width),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        
        z_dl = z_dl_nchw.to(orig_dtype)
        
        # Compute losses for monitoring and encoder training
        z_e_nhwc = z_e.permute(0, 2, 3, 1).contiguous()
        z_dl_nhwc = z_dl.permute(0, 2, 3, 1).contiguous()
        
        # Encoder commitment loss: encourages encoder to match reconstruction
        e_latent_loss = F.mse_loss(z_dl_nhwc.detach(), z_e_nhwc)
        # Dictionary reconstruction loss: encourages good reconstruction (gradients unused since dict not learned via backprop)
        dl_latent_loss = F.mse_loss(z_dl_nhwc, z_e_nhwc.detach())
        loss = self.commitment_cost * e_latent_loss + dl_latent_loss
        
        # Straight-through estimator: forward pass returns reconstruction, backward pass flows to encoder
        z_q = z_e + (z_dl - z_e).detach()
        
        return z_q, loss, coefficients
