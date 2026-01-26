import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """VQ baseline aligned with lucidrains vector-quantize-pytorch (no EMA)."""

    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(
            -1 / num_embeddings, 1 / num_embeddings
        )

    def forward(self, z_e: torch.Tensor):
        """
        Args:
            z_e: input tensor of shape [B, C, H, W] from the encoder

        Returns:
            Tuple of (quantized tensor, loss, perplexity, one-hot encodings)
        """
        if z_e.dim() != 4:
            raise ValueError(
                f"Expected input [B, C, H, W], got {tuple(z_e.shape)}"
            )
        if z_e.shape[1] != self.embedding_dim:
            raise ValueError(
                (
                    "Expected channel dim "
                    f"{self.embedding_dim} but received {z_e.shape[1]}"
                )
            )

        # [B, C, H, W] -> [B, H, W, C] -> [N, C]
        z_e = z_e.permute(0, 2, 3, 1).contiguous()
        z_flat = z_e.view(-1, self.embedding_dim)

        # Squared L2 distance to each codebook vector.
        distances = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.weight.pow(2).sum(dim=1)
            - 2 * z_flat @ self.embedding.weight.t()
        )
        indices = torch.argmin(distances, dim=1)

        encodings = torch.zeros(
            indices.shape[0],
            self.num_embeddings,
            device=z_flat.device,
            dtype=z_flat.dtype,
        )
        encodings.scatter_(1, indices.unsqueeze(1), 1)

        quantized = self.embedding(indices).view_as(z_e)

        # Commitment loss only (lucidrains-style); codebook learns via gradients.
        loss = self.commitment_cost * F.mse_loss(quantized, z_e)

        # Straight-through estimator: forward uses quantized, backward sees
        # identity.
        quantized = z_e + (quantized - z_e).detach()

        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        )

        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        return quantized, loss, perplexity, encodings


class VectorQuantizerEMA(nn.Module):
    """VQ baseline aligned with lucidrains vector-quantize-pytorch (EMA)."""

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        commitment_cost=0.25,
        ema_decay=0.99,
        epsilon=1e-10,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        self.epsilon = epsilon

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.normal_()
        self.embedding.weight.requires_grad = False

        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer(
            "_ema_w", torch.randn(num_embeddings, embedding_dim)
        )

    def forward(self, z_e: torch.Tensor):
        """
        Args:
            z_e: input tensor of shape [B, C, H, W] from the encoder

        Returns:
            Tuple of (quantized tensor, loss, perplexity, one-hot encodings)
        """
        if z_e.dim() != 4:
            raise ValueError(
                f"Expected input [B, C, H, W], got {tuple(z_e.shape)}"
            )
        if z_e.shape[1] != self.embedding_dim:
            raise ValueError(
                (
                    "Expected channel dim "
                    f"{self.embedding_dim} but received {z_e.shape[1]}"
                )
            )

        z_e = z_e.permute(0, 2, 3, 1).contiguous()
        z_flat = z_e.view(-1, self.embedding_dim)

        distances = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.weight.pow(2).sum(dim=1)
            - 2 * z_flat @ self.embedding.weight.t()
        )
        indices = torch.argmin(distances, dim=1)

        encodings = torch.zeros(
            indices.shape[0],
            self.num_embeddings,
            device=z_flat.device,
            dtype=z_flat.dtype,
        )
        encodings.scatter_(1, indices.unsqueeze(1), 1)

        quantized = self.embedding(indices).view_as(z_e)

        if self.training:
            with torch.no_grad():
                counts = encodings.sum(dim=0).to(
                    self.ema_cluster_size.dtype
                )
                self.ema_cluster_size.mul_(self.ema_decay).add_(
                    counts, alpha=1 - self.ema_decay
                )

                dw = (encodings.t() @ z_flat).to(self._ema_w.dtype)
                self._ema_w.mul_(self.ema_decay).add_(
                    dw, alpha=1 - self.ema_decay
                )

                n = self.ema_cluster_size.sum()
                cluster_size = (
                    (self.ema_cluster_size + self.epsilon)
                    / (n + self.num_embeddings * self.epsilon) * n
                )
                self.embedding.weight.copy_(
                    self._ema_w / cluster_size.unsqueeze(1)
                )

        loss = self.commitment_cost * F.mse_loss(quantized.detach(), z_e)

        quantized = z_e + (quantized - z_e).detach()

        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        )

        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        return quantized, loss, perplexity, encodings


class DictionaryLearning(nn.Module):
    """Per-pixel dictionary learning bottleneck with batch OMP sparse coding."""

    def __init__(
        self,
        num_embeddings=512,
        embedding_dim=64,
        sparsity_level=5,
        commitment_cost=0.25,
        dict_learning_rate=0.1,
        epsilon=1e-10,
        sparse_solver="omp",
        tolerance=None,
        omp_debug=False,
        **kwargs,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.sparsity_level = sparsity_level
        self.commitment_cost = commitment_cost
        self.epsilon = epsilon
        self.dict_learning_rate = dict_learning_rate
        self.omp_tolerance = tolerance
        self.omp_debug = omp_debug

        if sparse_solver != "omp":
            raise ValueError(
                f"Only sparse_solver='omp' is supported, got '{sparse_solver}'"
            )

        # Keep these attributes for downstream compatibility.
        self.patch_size = (1, 1)
        self.patch_stride = (1, 1)
        self.patch_flatten_order = "channel_first"

        self.dictionary = nn.Parameter(
            torch.randn(self.embedding_dim, self.num_embeddings)
        )
        self._last_diag = {}
        self._last_bottleneck_losses = {}

    def orthogonality_loss(self):
        d = F.normalize(self.dictionary, p=2, dim=0, eps=self.epsilon)
        gram = d.t() @ d
        identity = torch.eye(
            gram.shape[0], device=gram.device, dtype=gram.dtype
        )
        return (gram - identity).pow(2).mean()

    def batch_omp(self, X, D):
        """
        Batched Orthogonal Matching Pursuit adapted from
        https://github.com/amzn/sparse-vqvae/blob/main/utils/pyomp.py

        Args:
            X: Input signals of shape (M, B)
            D: Dictionary of shape (M, N) with normalized columns.

        Returns:
            coefficients: Sparse coefficient matrix of shape (N, B)
        """
        vector_dim, batch_size = X.size()
        dictionary_t = D.t()
        G = dictionary_t.mm(D)
        eps = torch.norm(X, dim=0)
        h_bar = dictionary_t.mm(X).t()

        h = h_bar
        x = torch.zeros_like(h_bar)
        L = torch.ones(batch_size, 1, 1, device=h.device)
        I = torch.ones(batch_size, 0, device=h.device).long()
        I_logic = torch.zeros_like(h_bar).bool()
        delta = torch.zeros(batch_size, device=h.device)

        def _update_logical(logical, to_add):
            running_idx = torch.arange(to_add.shape[0], device=to_add.device)
            logical[running_idx, to_add] = 1

        k = 0
        tol = self.omp_tolerance if self.omp_tolerance is not None else 1e-7
        while k < self.sparsity_level and eps.max() > tol:
            k += 1
            index = (h * (~I_logic).float()).abs().argmax(dim=1)
            _update_logical(I_logic, index)
            batch_idx = torch.arange(batch_size, device=G.device)
            expanded_batch_idx = batch_idx.unsqueeze(0).expand(k, batch_size).t()

            if k > 1:
                G_stack = G[I[batch_idx, :], index[expanded_batch_idx[..., :-1]]].view(
                    batch_size, k - 1, 1
                )
                w = torch.linalg.solve_triangular(L, G_stack, upper=False)
                w = w.view(-1, 1, k - 1)
                w_corner = torch.sqrt(1 - (w**2).sum(dim=2, keepdim=True))

                k_zeros = torch.zeros(batch_size, k - 1, 1, device=h.device)
                L = torch.cat(
                    (
                        torch.cat((L, k_zeros), dim=2),
                        torch.cat((w, w_corner), dim=2),
                    ),
                    dim=1,
                )

            I = torch.cat([I, index.unsqueeze(1)], dim=1)

            h_stack = h_bar[expanded_batch_idx, I[batch_idx, :]].view(
                batch_size, k, 1
            )
            x_stack = torch.cholesky_solve(h_stack, L)
            x[batch_idx.unsqueeze(1), I[batch_idx]] = x_stack[
                batch_idx
            ].squeeze(-1)

            beta = (
                x[batch_idx.unsqueeze(1), I[batch_idx]]
                .unsqueeze(1)
                .bmm(G[I[batch_idx], :])
                .squeeze(1)
            )
            h = h_bar - beta

            new_delta = (x * beta).sum(dim=1)
            eps += delta - new_delta
            delta = new_delta

            if self.omp_debug:
                print(
                    "Step {}, residual: {:.4f}, below tolerance: {:.4f}".format(
                        k,
                        eps.max(),
                        (eps < tol).float().mean().item(),
                    )
                )

        return x.t()

    def forward(self, z_e):
        if z_e.dim() != 4:
            raise ValueError(
                f"Expected input [B, C, H, W], got {tuple(z_e.shape)}"
            )
        B, C, H, W = z_e.shape
        if C != self.embedding_dim:
            raise ValueError(
                f"Expected channel dim {self.embedding_dim} but received {C}"
            )

        signals = z_e.permute(0, 2, 3, 1).contiguous().view(-1, C).t()
        dictionary = F.normalize(
            self.dictionary, p=2, dim=0, eps=self.epsilon
        )
        with torch.no_grad():
            coeffs = self.batch_omp(signals, dictionary)
        recon = (dictionary @ coeffs).t()
        z_dl = recon.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        dl_latent_loss = F.mse_loss(z_dl, z_e.detach())
        e_latent_loss = F.mse_loss(z_dl.detach(), z_e)
        loss = dl_latent_loss + self.commitment_cost * e_latent_loss

        self._last_bottleneck_losses = {
            "dl_latent_loss": dl_latent_loss.detach(),
            "e_latent_loss": e_latent_loss.detach(),
            "pattern_loss": z_e.new_tensor(0.0),
        }
        self._last_diag = {
            "dict_norm_mean": self.dictionary.norm(dim=0).mean().detach(),
            "coeff_abs_mean": coeffs.abs().mean().detach(),
        }

        z_dl = z_e + (z_dl - z_e).detach()
        return z_dl, loss, coeffs
