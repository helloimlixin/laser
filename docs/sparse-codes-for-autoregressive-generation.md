# Tokenization of Real-Valued Sparse Codes

Let $D=[d_1,\ldots,d_M]\in\mathbb R^{m\times M}$ be a fixed dictionary with
$\|d_j\|_2=1$. A continuous sparse representation approximates a vector
$h\in\mathbb R^m$ by $Dc$, where $c\in\mathbb R^M$ and $\|c\|_0\leq K$.
Its discrete support specifies the active atoms, while its real-valued
coefficients specify their contributions. We construct a finite token vocabulary
that jointly represents atom identity and coefficient value, and encode each
vector as a sequence of $K$ additive contributions.

For each atom $d_j$, we associate $L$ distinct nonzero coefficient levels
$\{\ell_{j,b}\}_{b=0}^{L-1}$. These levels are real-valued, may be signed, and
are shared across the $K$ residual steps. Define the vocabulary
$\mathcal V=\{0,\ldots,ML\}$ and its embedding function by

$$
e(0)=0,\qquad
e\bigl(1+(j-1)L+b\bigr)=\ell_{j,b}d_j,
\quad
j\in\{1,\ldots,M\},\quad b\in\{0,\ldots,L-1\}.
$$

Thus, $|\mathcal V|=ML+1$, and every nonzero token specifies one scaled atom.
The coefficient levels are fitted separately for each atom, allowing their
magnitudes to adapt to the dictionary geometry without requiring symmetric
positive and negative values. The token index completely determines its real
vector contribution once the dictionary and coefficient table are fixed. Token
zero denotes the zero contribution.

We obtain a token sequence by greedy residual quantization. With $r_0=h$, the
$k$-th step selects

$$
t_k=\arg\min_{v\in\mathcal V}
\|r_{k-1}-e(v)\|_2^2,
\qquad
r_k=r_{k-1}-e(t_k),
\qquad k=1,\ldots,K,
$$

using a fixed rule to resolve ties. The reconstruction is

$$
\widetilde h=\sum_{k=1}^{K}e(t_k).
$$

Assignments are computed jointly over atom identity and coefficient level.
Consequently, the selected support may differ from that of the continuous sparse
representation of $h$. If only a sparse code $c$ is available, the same procedure
can be applied to its reconstruction $Dc$. In either case, each selected
contribution remains fixed during subsequent residual steps. The sequence order
is the residual selection order, and repeated atoms are permitted.

The resulting representation retains a sparsity constraint. Specifically,
define the aggregated coefficient of atom $j$ as

$$
\widetilde c_j
=
\sum_{k=1}^{K}\sum_{b=0}^{L-1}
\ell_{j,b}\,
\mathbf 1\!\left[t_k=1+(j-1)L+b\right].
$$

Then $\widetilde h=D\widetilde c$ and $\|\widetilde c\|_0\leq K$.
Repeated selections of the same atom add their coefficients, while zero tokens
leave the reconstruction unchanged. The discrete representation therefore
belongs to a finite subset of the dictionary's union of subspaces. It
approximates continuous sparse reconstructions while making both support and
coefficient choices recoverable from token indices.

The structure of the vocabulary also permits efficient assignment. For a
residual $r$, selecting the closest nonzero codeword is equivalent to maximizing

$$
s_{j,b}(r)
=
2\ell_{j,b}\,d_j^\top r-\ell_{j,b}^2.
$$

The correlations $D^\top r$ are computed once and reused across all coefficient
levels, giving an assignment cost of $O(Mm+ML)$ per residual step. The zero token
has score zero. Its inclusion guarantees
$\|r_k\|_2^2\leq\|r_{k-1}\|_2^2$ under exact deterministic assignment.
This is a stepwise residual-error property; the greedy procedure does not in
general minimize the final reconstruction error over all length-$K$ sequences.

We fit the coefficient table with the dictionary fixed. Initial levels are
estimated from continuous pursuit coefficients. We then alternate residual
assignment and regularized scalar updates, keeping the coefficient table
unchanged during each assignment pass. Let $\mathcal I_{j,b}$ contain the
residuals assigned to the pair $(j,b)$ in a pass. Since the atoms are normalized,
the optimal continuous projection coefficient for residual $r_i$ is
$d_j^\top r_i$. Given a fixed prior level $\ell_{j,b}^{(0)}$ and
$\lambda>0$, the update is

$$
\ell_{j,b}^{\mathrm{new}}
=
\frac{
\sum_{i\in\mathcal I_{j,b}}d_j^\top r_i
+\lambda\ell_{j,b}^{(0)}
}{
|\mathcal I_{j,b}|+\lambda
}.
$$

For fixed residuals and assignments, this update minimizes a regularized scalar
least-squares objective. The prior stabilizes estimates for infrequently
selected entries, including entries with no assignments. The coefficient levels
are kept distinct and nonzero, and residual assignments are recomputed between
passes. These updates account for the residuals encountered by the discrete
encoder; they do not imply monotonic minimization of the complete multi-step
reconstruction objective. The dictionary and fitted levels are subsequently
held fixed during prior training.

For autoregressive training, we additionally use stochastic assignments and
soft targets, following [Lee et al. (2022)](https://arxiv.org/abs/2203.01941).
For temperature $\tau>0$, define

$$
q_\tau(v\mid r)
=
\frac{
\exp\!\left(-\|r-e(v)\|_2^2/\tau\right)
}{
\sum_{w\in\mathcal V}
\exp\!\left(-\|r-e(w)\|_2^2/\tau\right)
}.
$$

At each residual step, we sample $t_k\sim q_\tau(\cdot\mid r_{k-1})$ and
update $r_k=r_{k-1}-e(t_k)$. Thus, subsequent assignments are conditioned on the
sampled prefix. The distribution assigns greater mass to contributions that
better approximate the current residual and concentrates on the nearest
codeword as $\tau\to0$ when the minimizer is unique. Unlike deterministic
assignment, stochastic assignment need not reduce residual error at every step.
The temperature is expressed in squared-distance units and is calibrated to the
additional reconstruction distortion induced by sampling.

Let $H=(h_1,\ldots,h_T)$ be an ordered collection of vectors, and let
$S=(t_{u,k})\in\mathcal V^{T\times K}$ denote its token representation. An
autoregressive prior factorizes as

$$
p_\theta(S)
=
\prod_{u=1}^{T}\prod_{k=1}^{K}
p_\theta\!\left(t_{u,k}\mid S_{<u,:},S_{u,<k}\right).
$$

Writing $Q_\tau(S\mid H)$ for the sequential distribution induced by the
stochastic quantizer, we train the prior with

$$
\mathcal L(\theta)
=
-\mathbb E_{H,\,S\sim Q_\tau(\cdot\mid H)}
\left[
\frac{1}{TK}
\sum_{u=1}^{T}\sum_{k=1}^{K}
\sum_{v\in\mathcal V}
q_\tau(v\mid r_{u,k-1})
\log p_\theta\!\left(v\mid S_{<u,:},S_{u,<k}\right)
\right].
$$

The sampled tokens provide the causal context, while the corresponding soft
distributions provide the prediction targets. Assigning target mass according
to reconstruction distance exposes the prior to alternative discrete
decompositions and encodes the geometric relationship among candidate
contributions. This objective is a categorical cross-entropy over token
sequences. During generation, the prior predicts tokens from previously
generated tokens without access to the target vectors or their residuals.

The same embedding function can supply the prior with continuous context:
each token is represented by a learned projection of $e(t_{u,k})$, and the
partial sum $\sum_{i<k}e(t_{u,i})$ describes the reconstruction available before
the next residual prediction. After sampling, the token sequence is converted
to $\widetilde h_u=\sum_{k=1}^{K}e(t_{u,k})$ using only fixed codeword lookups
and addition. This gives real-valued sparse representations a finite categorical
interface while preserving their additive dictionary structure. The coefficient
resolution $L$ controls vocabulary size, and the residual depth $K$ controls
sequence length and the number of available contributions; together they
determine the tradeoff between approximation fidelity and autoregressive
prediction cost.
