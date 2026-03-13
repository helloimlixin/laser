# `proto.py`: Theoretical and Mathematical Overview

This note explains `proto.py` as a model, not as a software artifact. It intentionally avoids a code-path walkthrough and instead describes the mathematical objects, training objectives, and generative factorization implemented by the file.

## 1. High-level picture

The file implements a two-stage generative model:

1. A sparse-latent autoencoder learns to map an image into a structured latent representation built from dictionary atoms.
2. An autoregressive transformer learns a prior over that sparse latent representation.

The basic pipeline is

```math
x \xrightarrow{E_\theta} z_e \xrightarrow{\text{sparse bottleneck}} z_q \xrightarrow{G_\psi} \hat x
```

for stage 1, and then

```math
\text{stage-1 sparse codes} \xrightarrow{\text{flatten}} y_{1:T}
\xrightarrow{\text{Transformer prior}} p(y_{1:T})
```

for stage 2.

The central idea is that the latent space is not modeled as an unconstrained dense tensor. Instead, each latent site or latent patch must lie on a sparse dictionary manifold.

## 2. Notation

Let

- $x \in [-1,1]^{3 \times R \times R}$ be an input image.
- $E_\theta$ be the encoder.
- $G_\psi$ be the decoder.
- $z_e = E_\theta(x) \in \mathbb{R}^{C \times H \times W}$ be the encoder latent.
- $z_q$ be the sparse-structured latent produced by the bottleneck.
- $K$ be the number of dictionary atoms.
- $s$ be the sparsity level.
- $D \in \mathbb{R}^{C \times K}$ be the learned atom dictionary in the per-location case.

For the patch-based case, the dictionary lives in a higher-dimensional patch space:

```math
D_p \in \mathbb{R}^{(C p^2) \times K},
```

where $p$ is the latent patch size.

The latent sequence modeled by the transformer has length

```math
T = HWD_t,
```

where $D_t$ is the token depth per spatial location:

- $D_t = s$ in the real-valued coefficient regime.
- $D_t = 2s$ in the quantized coefficient regime.

## 3. Stage 1: sparse-latent autoencoding

### 3.1 Encoder-decoder backbone

The encoder-decoder pair is a multiscale convolutional autoencoder with residual blocks and optional attention. The exact layer topology is not the main conceptual point. Mathematically, it simply defines two maps:

```math
E_\theta : \mathbb{R}^{3 \times R \times R} \to \mathbb{R}^{C \times H \times W},
```

```math
G_\psi : \mathbb{R}^{C \times H \times W} \to \mathbb{R}^{3 \times R \times R}.
```

In the default `128 x 128` configuration used here, the target latent grid is `8 x 8`, so the encoder must reduce spatial resolution by a factor of `16` along each axis. If $L$ is the number of spatial downsampling steps, then ideally

```math
H = W = \frac{R}{2^L}.
```

The implementation uses `num_downsamples = 4`, so with $R = 128$,

```math
H = W = \frac{128}{2^4} = 8.
```

#### 3.1.1 Downsampling law

Each encoder downsampling layer pads on the right and bottom by one pixel and then applies a `3 x 3` convolution with stride `2`. If the incoming spatial size is $n$ and $n$ is even, then the output size is

```math
n'
=
\left\lfloor \frac{(n+1)-3}{2} \right\rfloor + 1
=
\frac{n}{2}.
```

So the spatial path is exactly

```math
128 \to 64 \to 32 \to 16 \to 8.
```

The decoder mirrors this with nearest-neighbor interpolation by a factor of `2`, followed by a `3 x 3` convolution, so its spatial path is

```math
8 \to 16 \to 32 \to 64 \to 128.
```

#### 3.1.2 Channel schedule in this implementation

The code does not use an unrestricted doubling schedule such as $(1,2,4,8, \dots)$. Instead it defines

```math
\mathrm{ch\_mult}_\ell = \min(2^\ell, 2), \qquad \ell = 0,\dots,L,
```

so with base width `num_hiddens = 128` and $L = 4$, the channel counts are

```math
128 \cdot (1,2,2,2,2) = (128,256,256,256,256).
```

Thus the encoder expands channels once, from `128` to `256`, and then keeps the latent pyramid at `256` channels until the final projection to the bottleneck width `embedding_dim = 16`.

#### 3.1.3 Residual block as the basic nonlinear operator

At each resolution, the main computation is performed by ResNet blocks. If $h \in \mathbb{R}^{C_{\mathrm{in}} \times r \times r}$, one block has the form

```math
\mathrm{ResBlock}(h)
=
S(h)
+
W_2 \,
\mathrm{Drop}\!\left(
\phi\!\left(
N_2\!\left(
W_1 \phi(N_1(h))
\right)
\right)
\right),
```

where

- $N_1, N_2$ are group-normalization operators,
- $\phi$ is the SiLU nonlinearity,
- $W_1, W_2$ are `3 x 3` convolutions,
- $S$ is either the identity or a learned `1 x 1` / `3 x 3` shortcut if the channel dimension changes.

This means each scale learns a perturbation of the identity map rather than a wholly new transformation, which is one reason deep pyramidal autoencoders remain trainable.

#### 3.1.4 Attention at the bottleneck scale

The model also includes self-attention blocks. For a feature map $h \in \mathbb{R}^{C \times r \times r}$, let

```math
Q \in \mathbb{R}^{r^2 \times C},
\qquad
K \in \mathbb{R}^{C \times r^2},
\qquad
V \in \mathbb{R}^{C \times r^2}
```

be `1 x 1` convolutions of the normalized feature map after reshaping the spatial grid into a sequence of length $r^2$. Define the attention matrix

```math
A = \mathrm{softmax}\!\left(\frac{QK}{\sqrt{C}}\right) \in \mathbb{R}^{r^2 \times r^2}.
```

Then the attention block is

```math
\mathrm{Attn}(h)
=
h
+
W_o (V A^\top).
```

Two details matter:

- attention can optionally be inserted at user-chosen resolutions via `attn_resolutions`,
- there is always a middle attention block at the lowest-resolution feature map.

So even without high-resolution attention, the model still gets one global mixing step at the `8 x 8` bottleneck scale.

#### 3.1.5 Exact encoder geometry for `128 x 128 -> 8 x 8`

With `num_residual_layers = 2`, `num_hiddens = 128`, `num_downsamples = 4`, and `embedding_dim = 16`, the encoder shape progression is:

- input image: $x \in \mathbb{R}^{3 \times 128 \times 128}$
- input `3 x 3` convolution: $\mathbb{R}^{3 \times 128 \times 128} \to \mathbb{R}^{128 \times 128 \times 128}$
- level 0 residual stack: stays at $\mathbb{R}^{128 \times 128 \times 128}$
- downsample: $\mathbb{R}^{128 \times 128 \times 128} \to \mathbb{R}^{128 \times 64 \times 64}$
- level 1 residual stack: $\mathbb{R}^{128 \times 64 \times 64} \to \mathbb{R}^{256 \times 64 \times 64}$
- downsample: $\mathbb{R}^{256 \times 64 \times 64} \to \mathbb{R}^{256 \times 32 \times 32}$
- level 2 residual stack: stays at $\mathbb{R}^{256 \times 32 \times 32}$
- downsample: $\mathbb{R}^{256 \times 32 \times 32} \to \mathbb{R}^{256 \times 16 \times 16}$
- level 3 residual stack: stays at $\mathbb{R}^{256 \times 16 \times 16}$
- downsample: $\mathbb{R}^{256 \times 16 \times 16} \to \mathbb{R}^{256 \times 8 \times 8}$
- level 4 residual stack plus middle `Res-Attn-Res` block: stays at $\mathbb{R}^{256 \times 8 \times 8}$
- output normalization, SiLU, and final `3 x 3` projection: $\mathbb{R}^{256 \times 8 \times 8} \to \mathbb{R}^{16 \times 8 \times 8}$

So the encoder output is

```math
z_e \in \mathbb{R}^{16 \times 8 \times 8}.
```

This is a spatial compression by `16` along each axis and a dense coordinate compression from

```math
3 \cdot 128 \cdot 128 = 49152
```

input scalars to

```math
16 \cdot 8 \cdot 8 = 1024
```

latent scalars, i.e. a `48:1` reduction in raw coordinate count before the sparse dictionary bottleneck is even applied.

#### 3.1.6 Exact decoder geometry for `8 x 8 -> 128 x 128`

The decoder is not a U-Net: it has no encoder-decoder skip connections. All information that reaches image space must pass through the latent tensor $z_q \in \mathbb{R}^{16 \times 8 \times 8}$.

The decoder first lifts the `16` latent channels back to `256` channels with a `3 x 3` convolution, applies the same middle `Res-Attn-Res` computation at `8 x 8`, and then runs a multiscale upsampling tower. Because the decoder uses `num_res_blocks + 1` residual blocks per scale, it is slightly heavier than the encoder on the way back up.

Its shape progression is:

- latent input: $z_q \in \mathbb{R}^{16 \times 8 \times 8}$
- input projection: $\mathbb{R}^{16 \times 8 \times 8} \to \mathbb{R}^{256 \times 8 \times 8}$
- middle `Res-Attn-Res`: stays at $\mathbb{R}^{256 \times 8 \times 8}$
- up level at `8 x 8`: stays at $\mathbb{R}^{256 \times 8 \times 8}$, then upsample to $\mathbb{R}^{256 \times 16 \times 16}$
- up level at `16 x 16`: stays at $\mathbb{R}^{256 \times 16 \times 16}$, then upsample to $\mathbb{R}^{256 \times 32 \times 32}$
- up level at `32 x 32`: stays at $\mathbb{R}^{256 \times 32 \times 32}$, then upsample to $\mathbb{R}^{256 \times 64 \times 64}$
- up level at `64 x 64`: stays at $\mathbb{R}^{256 \times 64 \times 64}$, then upsample to $\mathbb{R}^{256 \times 128 \times 128}$
- final level at `128 x 128`: maps $\mathbb{R}^{256 \times 128 \times 128} \to \mathbb{R}^{128 \times 128 \times 128}$
- output normalization, SiLU, and final `3 x 3` RGB projection: $\mathbb{R}^{128 \times 128 \times 128} \to \mathbb{R}^{3 \times 128 \times 128}$

If `out_tanh=True`, the final image is

```math
\hat x = \tanh(G_\psi(z_q)) \in [-1,1]^{3 \times 128 \times 128}.
```

So mathematically the decoder is a learned map from an `8 x 8` latent field back to image space, with all high-resolution detail reconstructed from the compressed representation rather than copied through skips.

The important constraint is what happens between encoder and decoder: the dense latent $z_e$ is projected onto a sparse dictionary model before decoding.

### 3.2 Per-location sparse bottleneck

In the non-patch formulation, each spatial latent vector

```math
z_e(u) \in \mathbb{R}^C, \qquad u \in \{1,\dots,H\} \times \{1,\dots,W\}
```

is represented as a sparse linear combination of atoms:

```math
z_q(u) = \sum_{j=1}^{s} a_j(u)\, d_{i_j(u)},
```

where

- $d_k \in \mathbb{R}^C$ is the $k$-th column of $D$,
- $i_j(u) \in \{1,\dots,K\}$ are selected atom indices,
- $a_j(u) \in \mathbb{R}$ are sparse coefficients.

Equivalently, if $\alpha(u) \in \mathbb{R}^K$ is a sparse coefficient vector, then

```math
z_q(u) = D \alpha(u), \qquad \|\alpha(u)\|_0 \le s.
```

The sparse coding problem at each site is

```math
\alpha^*(u)
=
\arg\min_{\alpha \in \mathbb{R}^K}
\|z_e(u) - D\alpha\|_2^2
\quad
\text{subject to}
\quad
\|\alpha\|_0 \le s.
```

This is the core representation learned by the model.

### 3.3 Orthogonal Matching Pursuit

The sparse approximation is computed with batched Orthogonal Matching Pursuit (OMP). At a conceptual level, OMP solves the constrained least-squares problem greedily.

Given a signal $z \in \mathbb{R}^C$, initialize

```math
r^{(0)} = z, \qquad S^{(0)} = \emptyset.
```

At iteration $m = 1,\dots,s$:

1. Select the atom with largest correlation to the current residual:

   ```math
   k^{(m)} = \arg\max_{k \notin S^{(m-1)}} |d_k^\top r^{(m-1)}|.
   ```

2. Expand the support:

   ```math
   S^{(m)} = S^{(m-1)} \cup \{k^{(m)}\}.
   ```

3. Refit the coefficients on the active support:

   ```math
   \alpha_{S^{(m)}}^{(m)}
   =
   \arg\min_{\beta}
   \|z - D_{S^{(m)}} \beta\|_2^2.
   ```

4. Update the residual:

   ```math
   r^{(m)} = z - D_{S^{(m)}} \alpha_{S^{(m)}}^{(m)}.
   ```

After $s$ steps, OMP returns a support of size exactly $s$ and aligned coefficients.

Two details matter conceptually:

1. The support size is fixed, so every latent site emits the same number of sparse slots.
2. The selected slots are reordered by descending coefficient magnitude before stage 2 sees them.

That second point removes arbitrary solver order and makes the tokenization more canonical.

### 3.4 Dictionary geometry

The dictionary atoms are constrained to unit norm:

```math
\|d_k\|_2 = 1 \quad \text{for all } k.
```

This matters because it separates direction from magnitude:

- the atom identity carries the latent direction,
- the coefficient carries the scale and sign.

If atom norms were unconstrained, scale could drift between atoms and coefficients, making the sparse representation less identifiable.

The implementation monitors dictionary coherence, which is the largest absolute off-diagonal cosine similarity:

```math
\mu(D) = \max_{i \ne j} |d_i^\top d_j|.
```

Low coherence is generally desirable for sparse coding because highly aligned atoms make support selection less stable.

### 3.5 Straight-through sparse projection

The sparse coding map itself is non-differentiable in practice because OMP involves hard support selection. The model therefore uses a straight-through surrogate.

Let $z_q$ be the sparse reconstruction obtained from OMP. The latent passed to the decoder is

```math
z_{\text{ste}} = z_e + \mathrm{sg}(z_q - z_e),
```

where $\mathrm{sg}$ denotes stop-gradient.

Forward pass:

```math
z_{\text{ste}} = z_q.
```

Backward pass:

```math
\frac{\partial z_{\text{ste}}}{\partial z_e} = I.
```

So the decoder sees the sparse-projected latent, while the encoder receives an identity-style backward signal through the bottleneck.

This is analogous in spirit to VQ-VAE, except the discrete nearest-neighbor map is replaced by sparse reconstruction from a learned dictionary.

### 3.6 Stage-1 objective

The stage-1 loss has two parts:

1. image reconstruction loss
2. bottleneck alignment loss

The image reconstruction term is

```math
\mathcal{L}_{\text{recon}}
=
\| \hat x - x \|_2^2,
\qquad
\hat x = G_\psi(z_{\text{ste}}).
```

The bottleneck term has a dictionary-side part and an encoder-side commitment part:

```math
\mathcal{L}_{\text{dict}}
=
\| z_q - \mathrm{sg}(z_e) \|_2^2,
```

```math
\mathcal{L}_{\text{commit}}
=
\| \mathrm{sg}(z_q) - z_e \|_2^2.
```

The bottleneck loss is

```math
\mathcal{L}_{\text{bottleneck}}
=
\mathcal{L}_{\text{dict}}
+
\beta \mathcal{L}_{\text{commit}},
```

where $\beta$ is the commitment weight.

The full stage-1 objective is

```math
\mathcal{L}_{\text{stage1}}
=
\mathcal{L}_{\text{recon}}
+
\lambda_b \mathcal{L}_{\text{bottleneck}},
```

where $\lambda_b$ is the external bottleneck weight.

Conceptually:

- $\mathcal{L}_{\text{recon}}$ trains the end-to-end autoencoder,
- $\mathcal{L}_{\text{dict}}$ trains the sparse manifold to approximate encoder latents,
- $\mathcal{L}_{\text{commit}}$ discourages the encoder from wandering away from that manifold.

### 3.7 What is and is not differentiated

It is important to be precise here.

The sparse coefficients and supports are obtained by solving a hard sparse-coding problem. The training signal does not differentiate through the combinatorial support-selection map itself. Instead:

- the encoder is optimized through the straight-through approximation,
- the dictionary is optimized through the reconstruction induced by the selected support,
- support selection is treated as frozen inside each forward pass.

So the learned dictionary is trained under a piecewise-defined approximation to the true sparse coding operator.

That is mathematically imperfect, but it is a practical and common compromise for structured latent models.

## 4. Tokenization of sparse codes

Stage 2 does not see dense latents. It sees a tokenized version of the sparse representation.

At each latent site $u$, the sparse code consists of

```math
\{(i_1(u), a_1(u)), \dots, (i_s(u), a_s(u))\}.
```

There are two ways to expose this to the transformer.

### 4.1 Real-valued coefficient regime

In the real-valued regime, only the atom identities are discretized into tokens:

```math
t_j(u) = i_j(u), \qquad j=1,\dots,s.
```

The coefficients remain continuous side information:

```math
c_j(u) = a_j(u).
```

So each spatial location emits $s$ discrete tokens and $s$ real-valued scalars.

The stage-2 sequence length becomes

```math
T = HWs.
```

The vocabulary contains:

- $K$ atom tokens,
- one padding token,
- one beginning-of-sequence token.

### 4.2 Quantized coefficient regime

In the quantized regime, each coefficient is itself discretized into one of $B$ bins:

```math
q(a_j(u)) \in \{1,\dots,B\}.
```

The token stream alternates atom identity and coefficient bin:

```math
(i_1(u), q(a_1(u)), i_2(u), q(a_2(u)), \dots, i_s(u), q(a_s(u))).
```

So each location emits $2s$ tokens and the sequence length becomes

```math
T = HW(2s).
```

The vocabulary contains:

- $K$ atom tokens,
- $B$ coefficient-bin tokens,
- one padding token,
- one beginning-of-sequence token.

This converts stage 2 into a purely discrete sequence-modeling problem.

### 4.3 Coefficient clipping

When coefficients are used as decoder inputs in the real-valued regime, they are projected into a bounded interval:

```math
\Pi_{[-c_{\max}, c_{\max}]}(a) = \min(\max(a,-c_{\max}), c_{\max}).
```

This means the decoder is trained and sampled on a bounded sparse-code manifold rather than an unbounded regression target.

That is mathematically important because it defines the actual latent image manifold learned by stage 1.

## 5. Coefficient quantization

When the quantized regime is used, the model supports two coefficient discretizations.

### 5.1 Uniform quantization

Let $a \in [-c_{\max}, c_{\max}]$. The uniform map is

```math
u(a) = \frac{a + c_{\max}}{2c_{\max}} \in [0,1],
```

followed by discretization into $B$ bins:

```math
b(a) = \mathrm{round}\left(u(a)(B-1)\right).
```

Decoding maps the bin index back to a fixed bin center.

### 5.2 Mu-law quantization

To allocate more resolution near zero, the model also supports mu-law companding. Define the normalized coefficient

```math
\bar a = a / c_{\max} \in [-1,1].
```

Then apply

```math
f_\mu(\bar a)
=
\mathrm{sign}(\bar a)
\frac{\log(1 + \mu |\bar a|)}{\log(1+\mu)}.
```

This companded value is quantized uniformly in $[-1,1]$, then inverted at decode time.

The effect is to use finer effective resolution near zero, which is well matched to sparse code distributions whose mass often concentrates on small magnitudes.

## 6. Patch-based sparse bottleneck

The file also implements a patch-based sparse bottleneck. Here the sparse object is not a single latent vector $z_e(u)$, but a local latent patch.

### 6.1 Patch extraction

Let

```math
P_u(z_e) \in \mathbb{R}^{Cp^2}
```

be the flattened latent patch centered around location $u$, with patch size $p$ and stride $r$.

For each patch, the sparse coding problem becomes

```math
\alpha^*(u)
=
\arg\min_{\alpha}
\|P_u(z_e) - D_p \alpha\|_2^2
\quad
\text{subject to}
\quad
\|\alpha\|_0 \le s.
```

The reconstructed patch is

```math
\hat P_u = D_p \alpha^*(u).
```

So the dictionary now lives in patch space rather than per-site latent-vector space.

### 6.2 Why patches change the representation

Per-location sparse coding assumes each spatial site can be reconstructed independently once the convolutional encoder has done its work.

Patch-based coding instead treats a small local neighborhood as the primitive object. This allows a single atom to represent a small structured motif in latent space rather than one isolated feature vector.

That can increase expressivity, but it changes both the geometry and the sequence semantics:

- one token stack corresponds to one latent patch, not one latent site,
- reconstruction requires a stitching rule because patches overlap.

### 6.3 Patch reconstruction operators

Two overlap rules are implemented mathematically.

#### Center-crop stitching

Each reconstructed patch contributes only its central $r \times r$ region. If the patch size is $p$ and stride is $r$, define the crop margin

```math
c = \frac{p-r}{2}.
```

Then each reconstructed patch $\hat P_u$ is reshaped into a $C \times p \times p$ tensor, cropped to

```math
\hat P_u[:, c:c+r, c:c+r],
```

and tiled into the output grid without averaging.

This makes every output latent pixel come from exactly one patch center.

#### Hann overlap-add

Let $w \in \mathbb{R}^{p \times p}$ be a separable Hann window. Each reconstructed patch is weighted by $w$ and overlap-added:

```math
\tilde z = \sum_u W_u \odot \hat P_u,
```

where $W_u$ is the shifted window at patch location $u$.

The final reconstruction divides by the accumulated weight map:

```math
z_q = \frac{\sum_u W_u \odot \hat P_u}{\sum_u W_u + \varepsilon}.
```

This is a weighted partition-of-unity style reconstruction. It is smoother than naive averaging and behaves especially well at 50 percent overlap.

## 7. Stage 2: autoregressive prior over sparse codes

Once stage 1 is trained, it defines a sparse-code distribution over the training set. Stage 2 fits an autoregressive prior to that induced representation.

## 7.1 Sequence geometry

The sparse code tensor has shape

```math
H \times W \times D_t,
```

where $D_t$ is the token depth per spatial position.

Flattening gives a sequence

```math
y_{1:T}, \qquad T = H W D_t.
```

The ordering is raster order over spatial sites with the sparse-slot depth varying within each site.

This matters because the transformer is not invariant to flattening order. The chosen ordering defines the causal factorization it learns.

### 7.2 Transformer factorization

In the purely discrete case, the model is simply

```math
p(y_{1:T}) = \prod_{t=1}^{T} p(y_t \mid y_{<t}).
```

In the real-valued coefficient case, the stage-2 representation consists of atoms and coefficients:

```math
(a_1,c_1), \dots, (a_T,c_T).
```

The model factorizes this as

```math
p(a_{1:T}, c_{1:T})
=
\prod_{t=1}^{T}
p(a_t \mid a_{<t}, c_{<t})
\,
p(c_t \mid a_t, a_{<t}, c_{<t}).
```

This is the key probabilistic design choice of the real-valued path.

It says:

1. predict which atom comes next,
2. then predict the coefficient conditioned on that atom and the prior history.

## 7.3 Positional structure in the transformer

Each sequence position corresponds to:

- a spatial site index,
- a depth-slot index,
- a token type.

If position $t$ corresponds to spatial index $s(t)$ and depth slot $d(t)$, then the input embedding is conceptually

```math
e_t
=
e_{\text{token}}(y_t)
+
e_{\text{space}}(s(t))
+
e_{\text{depth}}(d(t))
+
e_{\text{type}}(t).
```

The type embedding distinguishes the BOS position from regular content positions. The spatial and depth embeddings tell the transformer where a token lies in the latent grid and which sparse slot it represents.

This is mathematically important because the flattened sequence is not just a 1D token string. It is a serialized 3D grid with known geometry.

## 8. Stage 2 in the quantized regime

In the quantized regime, stage 2 is a standard categorical autoregressive model.

### 8.1 Likelihood

Let $y_t$ be the next token. The transformer produces logits $\ell_t \in \mathbb{R}^V$ over the vocabulary. The model distribution is

```math
p(y_t = v \mid y_{<t})
=
\frac{\exp(\ell_{t,v})}{\sum_{v'} \exp(\ell_{t,v'})}.
```

### 8.2 Loss

Training minimizes the negative log-likelihood:

```math
\mathcal{L}_{\text{CE}}
=
-
\sum_{t=1}^{T}
\log p(y_t \mid y_{<t}).
```

In practice this is the usual cross-entropy over the shifted sequence.

### 8.3 Structural masking

Because the quantized representation alternates atom tokens and coefficient-bin tokens, valid outputs depend on the parity of the position inside each local sparse stack.

So the generative process is not merely "sample any token from the vocabulary." It is constrained to alternate between:

- atom identity slots,
- coefficient-bin slots.

Mathematically, the support of $p(y_t \mid y_{<t})$ depends on whether $t$ is an atom step or a coefficient step.

## 9. Stage 2 in the real-valued regime

The real-valued path is more interesting mathematically.

### 9.1 Autoregressive conditioning

At training time, the transformer receives a shifted atom sequence

```math
[\text{BOS}, a_1, a_2, \dots, a_{T-1}]
```

and a shifted coefficient sequence

```math
[0, c_1, c_2, \dots, c_{T-1}].
```

The hidden state at time $t$ therefore summarizes

```math
(a_{<t}, c_{<t}).
```

From that hidden state the model predicts:

- a categorical distribution for $a_t$,
- a scalar coefficient for $c_t$, conditioned on the chosen atom.

### 9.2 Atom-conditioned coefficient normalization

The coefficient regression problem is made easier by normalizing coefficients separately for each atom.

For atom $k$, compute empirical statistics $(\mu_k, \sigma_k)$ from stage-1 sparse codes. Then the normalized coefficient target is

```math
\tilde c_t
=
\mathrm{clip}
\left(
\frac{c_t - \mu_{a_t}}{\sigma_{a_t}},
-M,
M
\right),
```

where $M$ is a fixed normalization bound.

The model predicts $\hat{\tilde c}_t$, and decoding maps it back via

```math
\hat c_t
=
\hat{\tilde c}_t \sigma_{a_t} + \mu_{a_t}.
```

This is a strong modeling choice. The coefficient head is not asked to learn one global scalar distribution for all atoms. Instead, it learns residual variation after atom-specific centering and scaling.

### 9.3 Coefficient head factorization

Let $h_t$ be the transformer hidden state before the output heads. The coefficient predictor uses both:

- the historical context summarized by $h_t$,
- the embedding of the selected atom $a_t$.

So the coefficient map is conceptually

```math
\hat{\tilde c}_t = f_{\text{coeff}}(h_t, e(a_t)).
```

This matches the factorization

```math
p(c_t \mid a_t, a_{<t}, c_{<t}).
```

Without conditioning on $a_t$, the scalar regression target would mix the statistics of all atoms and become substantially less coherent.

## 10. Stage-2 objectives in the real-valued regime

The real-valued path always includes atom cross-entropy and then adds one auxiliary coefficient objective.

### 10.1 Atom cross-entropy

The categorical term is

```math
\mathcal{L}_{\text{atom}}
=
-
\sum_{t=1}^{T} \log p(a_t \mid a_{<t}, c_{<t}).
```

### 10.2 Direct coefficient regression

Two direct scalar objectives are supported.

Mean squared error:

```math
\mathcal{L}_{\text{coeff-MSE}}
=
\sum_{t=1}^{T} (\hat{\tilde c}_t - \tilde c_t)^2.
```

Huber loss:

```math
\mathcal{L}_{\text{coeff-Huber}}
=
\sum_{t=1}^{T}
\mathrm{Huber}_\delta(\hat{\tilde c}_t - \tilde c_t).
```

These are natural if coefficient accuracy itself is the desired target.

### 10.3 Reconstruction-aware coefficient losses

The more interesting options measure coefficient quality through the induced sparse latent reconstruction.

Let

```math
R(a_{1:T}, c_{1:T})
```

be the operator that reshapes a sequence into the latent sparse grid and reconstructs the corresponding latent tensor through the stage-1 bottleneck.

Then the target sparse latent is

```math
z^* = R(a_{1:T}, c_{1:T}).
```

There are two reconstruction-style losses.

#### Predicted atoms plus predicted coefficients

Sample or argmax the model's atom predictions $\hat a_{1:T}$ and use predicted coefficients $\hat c_{1:T}$:

```math
\hat z = R(\hat a_{1:T}, \hat c_{1:T}).
```

Then minimize

```math
\mathcal{L}_{\text{recon-MSE}}
=
\| \hat z - z^* \|_2^2.
```

This couples atom and coefficient quality through the geometry of the sparse manifold.

#### Ground-truth atoms plus predicted coefficients

Use the true atoms but predicted coefficients:

```math
\hat z = R(a_{1:T}, \hat c_{1:T}),
```

and minimize

```math
\mathcal{L}_{\text{gt-atom-recon-MSE}}
=
\| \hat z - z^* \|_2^2.
```

This isolates coefficient quality while evaluating it in the latent geometry actually used by the decoder.

### 10.4 Full real-valued stage-2 loss

The full loss is

```math
\mathcal{L}_{\text{stage2}}
=
\mathcal{L}_{\text{atom}}
+
\lambda_c \mathcal{L}_{\text{coeff}},
```

where $\mathcal{L}_{\text{coeff}}$ is one of the coefficient objectives above.

## 11. Scheduled sampling

The real-valued path includes scheduled sampling to reduce exposure bias.

Teacher forcing trains on the true history

```math
(a_{<t}, c_{<t}),
```

but generation uses model-sampled history. To shrink that train-test mismatch, previous-step inputs are sometimes replaced with model predictions during training.

If $\rho(\tau)$ is the replacement probability at training progress $\tau$, then for a previous time step $m < t$ the training input is

```math
(\tilde a_m, \tilde c_m)
=
\begin{cases}
(\hat a_m, \hat c_m), & \text{with probability } \rho(\tau), \\
(a_m, c_m), & \text{with probability } 1 - \rho(\tau).
\end{cases}
```

The schedule increases from $0$ toward a chosen final probability.

Mathematically, this means the conditional contexts used during optimization gradually move from the empirical data distribution toward the model's own rollout distribution.

## 12. Sampling from the model

After training, generation proceeds entirely in sparse-code space before decoding back to pixels.

### 12.1 Quantized regime

Sample autoregressively:

```math
y_t \sim p(y_t \mid y_{<t}),
```

subject to the structural constraint that atom and coefficient-bin positions alternate appropriately.

Reshape the sampled sequence into a sparse token grid, reconstruct the latent tensor $z_q$, and decode:

```math
\hat x = G_\psi(z_q).
```

### 12.2 Real-valued regime

At each step:

1. sample an atom

   ```math
   a_t \sim p(a_t \mid a_{<t}, c_{<t}),
   ```

2. predict or sample a coefficient

   ```math
   \hat c_t \approx \mathbb{E}[c_t \mid a_t, a_{<t}, c_{<t}]
   ```

through the coefficient head.

Then reconstruct the stage-1 sparse latent from $(a_{1:T}, \hat c_{1:T})$ and decode it to image space.

So even though stage 2 is a transformer, the final image model is still fundamentally mediated by the stage-1 sparse manifold.

## 13. Conceptual interpretation

The model can be understood as imposing three layers of structure.

### 13.1 Convolutional structure

The encoder-decoder pair extracts a multiscale latent representation adapted to images.

### 13.2 Sparse linear structure

The bottleneck forces local latent content to lie near a union of low-dimensional sparse subspaces generated by the dictionary.

For fixed support $S$, the latent lives in

```math
\mathrm{span}(D_S).
```

Across all supports of size at most $s$, the bottleneck defines a union-of-subspaces model:

```math
\mathcal{M}
=
\bigcup_{|S| \le s} \mathrm{span}(D_S).
```

This is a useful way to think about the latent manifold. The autoencoder does not learn an arbitrary nonlinear latent geometry. It learns a decoder whose inputs must lie in or near $\mathcal{M}$.

### 13.3 Autoregressive structure

Stage 2 learns a probability law over the discrete or hybrid coordinates that index points on this sparse manifold.

So the full generative model is:

1. sample a sparse code sequence from the transformer,
2. map that sequence to a sparse latent point on $\mathcal{M}$,
3. decode the latent point back to image space.

## 14. Why the quantized and real-valued variants differ

The two stage-2 modes correspond to two different statistical assumptions.

### 14.1 Quantized variant

Everything is turned into symbols. This gives a clean categorical model:

```math
p(y_{1:T}) = \prod_t p(y_t \mid y_{<t}).
```

Advantages:

- simpler likelihood,
- simpler sampling,
- no scalar regression instability.

Cost:

- coefficient values are discretized,
- reconstruction fidelity may be limited by binning resolution.

### 14.2 Real-valued variant

Only atom identities are discretized, while amplitudes remain continuous.

Advantages:

- potentially better fidelity because coefficients need not be quantized,
- more faithful use of sparse coding geometry.

Cost:

- stage 2 becomes a hybrid discrete-continuous model,
- training requires normalization, auxiliary losses, and careful conditioning.

So the real-valued path is mathematically richer but operationally more delicate.

## 15. Main approximation built into the model

The deepest approximation in the whole system is not the transformer. It is the treatment of sparse coding during stage 1.

The exact sparse-coding map

```math
z_e \mapsto \alpha^*(z_e; D)
```

is non-smooth because support selection changes discontinuously. The model therefore uses a surrogate training procedure:

- solve for sparse codes with hard OMP,
- reconstruct the latent from those codes,
- treat the discrete support choice as fixed during backpropagation.

This means the learned system is best viewed as an alternating structured approximation rather than a fully differentiable end-to-end probabilistic model.

That does not make it invalid. It simply clarifies what kind of mathematics is actually being optimized.

## 16. Bottom line

`proto.py` implements a sparse-latent generative model whose key object is a learned dictionary-based latent manifold.

Stage 1 learns:

```math
x \mapsto z_e \mapsto z_q \mapsto \hat x
```

with $z_q$ constrained to be a sparse reconstruction from learned atoms.

Stage 2 learns a prior over the coordinates of that sparse representation:

- fully discrete coordinates in the quantized regime,
- discrete atom identities plus continuous coefficients in the real-valued regime.

The most useful compact summary is:

```math
\text{autoencoder backbone}
+
\text{sparse dictionary manifold}
+
\text{autoregressive prior over sparse coordinates}.
```

That is the mathematical content of the implementation.
