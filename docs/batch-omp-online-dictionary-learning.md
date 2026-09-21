# Sparse Coding with Batch OMP and Alternating Online Dictionary Updates

Let $H=[h_1,\ldots,h_N]\in\mathbb R^{m\times N}$ contain a minibatch of
vectors, and let $D=[d_1,\ldots,d_M]\in\mathbb R^{m\times M}$ be a learned
dictionary with $\|d_j\|_2=1$. We approximate each $h_i$ by $Dc_i$, where
$c_i\in\mathbb R^M$ has at most $K$ nonzero entries. The corresponding
dictionary-learning objective is

$$
\min_{D,C}\frac{1}{2}\|H-DC\|_F^2
\quad\text{subject to}\quad
\|c_i\|_0\leq K,\qquad \|d_j\|_2=1,
$$

where $C=[c_1,\ldots,c_N]$. We alternate between sparse inference with the
dictionary fixed and a dictionary update with the inferred codes fixed. Sparse
inference uses batch orthogonal matching pursuit (OMP). The dictionary update
uses normalized residual least-squares targets, a shared relaxation parameter,
and a reconstruction-error acceptance test. When $H$ is produced by a learned
encoder, these operations are interleaved with gradient updates of the
surrounding model.

We implement sparse inference using the shared-Gram construction and incremental
Cholesky factorization of batch OMP
([Rubinstein et al., 2008](https://csaws.cs.technion.ac.il/~ronrubin/Publications/KSVD-OMP-v2.pdf)).
For a fixed dictionary, we first compute

$$
G=D^\top D\in\mathbb R^{M\times M},
\qquad
B=D^\top H\in\mathbb R^{M\times N}.
$$

The Gram matrix is shared by all vectors in the batch. Each vector has its own
ordered support $S_i^{(k)}$ and coefficient vector after pursuit step $k$.
Starting with $S_i^{(0)}=\varnothing$, $c_i^{(0)}=0$, and
$a_i^{(0)}=B_{:,i}$, we select

$$
j_i^{(k)}
=
\arg\max_{j\notin S_i^{(k-1)}}|a_{i,j}^{(k-1)}|,
\qquad
S_i^{(k)}=(S_i^{(k-1)},j_i^{(k)}).
$$

Selected indices are masked, so each support contains distinct atoms. Ties are
resolved deterministically. The absolute correlation permits signed
coefficients; atom selection does not constrain coefficient sign.

After extending the support, OMP refits every active coefficient:

$$
\gamma_i^{(k)}
=
\arg\min_{\gamma\in\mathbb R^k}
\|h_i-D_{S_i^{(k)}}\gamma\|_2^2.
$$

For a full-column-rank selected subdictionary, the coefficients solve

$$
G_{S_i^{(k)},S_i^{(k)}}\gamma_i^{(k)}
=B_{S_i^{(k)},i}.
$$

The vector $c_i^{(k)}$ places $\gamma_i^{(k)}$ on the selected support and
zeros elsewhere. Re-estimation at each depth is essential: introducing a new
atom can change the coefficients of all earlier selections. In exact arithmetic,
the resulting residual is orthogonal to the selected span,
$D_{S_i^{(k)}}^\top(h_i-D_{S_i^{(k)}}\gamma_i^{(k)})=0$.

We solve these systems by maintaining a Cholesky factor
$L_i^{(k)}(L_i^{(k)})^\top=G_{S_i^{(k)},S_i^{(k)}}$. At the first step,
$L_i^{(1)}=[1]$. For $k>1$, define

$$
g_i=G_{S_i^{(k-1)},j_i^{(k)}},
\qquad
L_i^{(k-1)}w_i=g_i,
\qquad
q_i=1-w_i^\top w_i,
\qquad
\delta_i=\sqrt{q_i}.
$$

The augmented factor is

$$
L_i^{(k)}
=
\begin{bmatrix}
L_i^{(k-1)} & 0\\
w_i^\top & \delta_i
\end{bmatrix}.
$$

Two triangular solves then recover the coefficients:

$$
L_i^{(k)}u_i=B_{S_i^{(k)},i},
\qquad
(L_i^{(k)})^\top\gamma_i^{(k)}=u_i.
$$

The Cholesky solve is used only when the Schur complement $q_i$ is sufficiently
positive relative to the diagonal and the precision used to form the Gram
matrix. Small or nonfinite pivots trigger a double-precision singular-value
decomposition of the selected subdictionary. In the unregularized case, singular
values below a relative precision threshold are discarded. With ridge
regularization, the inverse singular values are replaced by
$\sigma/(\sigma^2+\rho)$. A vector that requires this fallback remains on that
path for subsequent pursuit depths.

We also evaluate the reconstructed residual directly after each coefficient
solve. If its objective increases beyond numerical tolerance or becomes
nonfinite, the stable solve is attempted. If that candidate still fails after
conversion to the output precision, the previous coefficients are retained and
the newly selected slot receives a zero coefficient. For $\rho>0$, this check
uses the regularized objective, including $\rho\|\gamma\|_2^2$. A zero coefficient
denotes an inactive slot. Exact least-squares identities apply when the selected
system is nonsingular and no truncation or rejection is needed.

Residual correlations are updated directly from the precomputed matrices:

$$
a_i^{(k)}
=
B_{:,i}-G_{:,S_i^{(k)}}\gamma_i^{(k)}
=D^\top\!\left(h_i-D_{S_i^{(k)}}\gamma_i^{(k)}\right).
$$

This avoids recomputing a dictionary projection of the explicit residual on
the Cholesky path. For vectors using the stable fallback, correlations are
recomputed from the explicit residual in double precision. Support selection,
factor updates, and triangular solves
are batched across vectors. The procedure executes $K$ selections and returns
the support and its fitted real coefficients; the number of nonzero
coefficients can be smaller than $K$.

Precomputing $G$ costs $O(mM^2)$, and computing $B$ costs $O(mMN)$.
For the direct correlation updates above, the remaining work over all pursuit
depths, including direct reconstruction checks, is $O(N(M+m)K^2+NK^3)$
on the Cholesky path. A fallback at depth $k$ adds a selected-subdictionary
decomposition and residual projection, costing $O(mk^2+mM)$ for that vector
when $k\leq m$. The shared Gram matrix requires $O(M^2)$ storage,
the batch correlation workspace requires $O(NM)$ storage, and the per-vector
Cholesky factors require $O(NK^2)$ storage. The Gram matrix is recomputed when
the dictionary changes.

Numerical conditioning can additionally be controlled by a ridge parameter
$\rho\geq0$. In this variant, the coefficient solve uses
$G_{S,S}+\rho I$, and the Cholesky diagonal term becomes $1+\rho$.
The coefficients then minimize
$\|h_i-D_S\gamma\|_2^2+\rho\|\gamma\|_2^2$; ordinary OMP is recovered at
$\rho=0$. An optional coherence threshold $\mu<1$ restricts new candidates to
atoms satisfying $|d_j^\top d_\ell|\leq\mu$ for every previously selected
atom $d_\ell$. If no unselected atom satisfies the threshold, selection falls
back to the remaining unselected atoms. These conditioning variants preserve
the fixed support budget, but the regularized solve does not have the exact
residual-orthogonality property of unregularized OMP.

Algorithm 1 gives the batched pursuit procedure. Setting $\rho=0$ and $\mu=1$
recovers the unregularized, unrestricted selection rule. The parallel loop
denotes batched operations over independent vectors, and each saved prefix is
the reconstruction after refitting that depth's coefficients.

~~~text
Algorithm 1: Batch OMP with incremental Cholesky solves

Input: H in R^(m x N); unit-column dictionary D in R^(m x M);
       sparsity K <= M; ridge rho >= 0; coherence threshold 0 < mu <= 1;
       relative pivot threshold tau_piv; solve tolerance tau_obj;
       relative singular-value threshold tau_svd.
Output: ordered supports S, coefficient matrix C, prefix reconstructions P.

G <- transpose(D) D
B <- transpose(D) H
C <- zeros(M, N)
For each i:
    S[i] <- empty; L[i] <- empty; a[i] <- B[:, i]
    gamma_old[i] <- empty; stable[i] <- false
    objective_old[i] <- squared_norm(H[:, i])

For k = 1, ..., K:
    In parallel for i = 1, ..., N:
        J <- {1, ..., M} excluding S[i]
        If mu < 1 and S[i] is nonempty:
            J_allowed <- {j in J : max(abs(G[S[i], j])) <= mu}
            If J_allowed is nonempty: J <- J_allowed
        j <- index in J maximizing abs(a[i][j])
             (choose the smallest index in a tie)

        diagonal <- G[j, j] + rho
        If k = 1: L[i] <- [sqrt(diagonal)]
        Else if not stable[i]:
            w <- solve_lower_triangular(L[i], G[S[i], j])
            pivot <- diagonal - dot(w, w)
            If pivot is nonfinite or pivot <= tau_piv diagonal:
                stable[i] <- true
            Else:
                L[i] <- block_matrix([[L[i], 0], [transpose(w), sqrt(pivot)]])

        Append j to S[i]
        If not stable[i]:
            gamma <- solve_cholesky(L[i], B[S[i], i])
            If objective(gamma) is nonfinite or exceeds objective_old[i]
               by more than tau_obj max(objective_old[i], 1):
                stable[i] <- true
        If stable[i]:
            gamma <- double_precision_SVD_solve(D[:, S[i]], H[:, i],
                                               rho, tau_svd)
            Cast gamma to the coefficient output precision
        If objective(gamma) is nonfinite or exceeds objective_old[i]
           by more than tau_obj max(objective_old[i], 1):
            gamma <- concatenate(gamma_old[i], [0])
        C[S[i], i] <- gamma
        P[k][:, i] <- D[:, S[i]] gamma
        objective_old[i] <- squared_norm(H[:, i] - P[k][:, i])
                            + rho squared_norm(gamma)
        gamma_old[i] <- gamma
        If stable[i]: a[i] <- transpose(D) (H[:, i] - P[k][:, i])
        Else: a[i] <- B[:, i] - G[:, S[i]] gamma

Return S, C, P
~~~

When sparse coding is embedded in a trainable model, pursuit is evaluated
without differentiating through support selection or the coefficient solves.
Let $\widehat h_i^{(k)}=D_{S_i^{(k)}}\gamma_i^{(k)}$ denote the fitted
reconstruction at depth $k$. We pass the final reconstruction through a
straight-through estimator,

$$
h_i^{\mathrm{ST}}
=
h_i+\mathrm{sg}\!\left(\widehat h_i^{(K)}-h_i\right),
$$

where $\mathrm{sg}$ denotes stop-gradient. The forward value is
$\widehat h_i^{(K)}$, while the surrogate derivative with respect to $h_i$
is the identity. A downstream differentiable objective can therefore train the
encoder through the reconstructed representation.

We also apply a commitment objective to encourage encoder outputs to admit
accurate sparse approximations. With progressive supervision, this objective is

$$
\mathcal L_{\mathrm{commit}}
=
\frac{\beta}{NmK}
\sum_{i=1}^{N}\sum_{k=1}^{K}
\left\|h_i-\mathrm{sg}\!\left(\widehat h_i^{(k)}\right)\right\|_2^2,
$$

where $\beta$ controls its weight. Each prefix uses the coefficients refitted
at that pursuit depth. Truncating the final coefficient vector would generally
produce different prefix reconstructions. Final-depth supervision is obtained
by retaining only the $k=K$ term and normalizing by $Nm$. The dictionary is held
outside the gradient update in the alternating formulation. Its explicit update
uses the final-depth codes, even when the encoder receives progressive
commitment supervision.

After a gradient update of the surrounding model, we update the dictionary
using detached vectors and codes recorded during all forward passes contributing
to that gradient update. Accumulated gradients are normalized by the actual
number of examples, including a final partial accumulation window. A dictionary
update may use one such complete minibatch or a finite window of recorded
minibatches. A skipped gradient step discards pending dictionary statistics and
does not advance the dictionary schedules.
Denote the concatenated vectors and fixed codes in this window again by $H$
and $C$. The dictionary objective for this step is

$$
F(D;H,C)=\|H-DC\|_F^2.
$$

The recorded vectors are not recomputed after the encoder update, and the
coefficients are not refitted while evaluating candidate dictionary updates.
Each window is discarded after its update. The procedure is online in the
sense that it adapts the dictionary from successive finite batches of
representations, without optimizing over the full training history.

Let $R=H-DC$ and let $v_j=C_{j,:}^\top$ collect the coefficients of atom $j$
across the window. The residual with this atom's contribution removed is

$$
R_{-j}=R+d_jv_j^\top.
$$

Holding all other atoms and all coefficients fixed, the unit-norm coordinate
subproblem is

$$
\overline d_j
=
\arg\min_{\|d\|_2=1}
\|R_{-j}-dv_j^\top\|_F^2.
$$

Define the coefficient energy and residual statistic

$$
s_j=v_j^\top v_j,
\qquad
u_j=Rv_j+s_jd_j=R_{-j}v_j.
$$

For nonzero $u_j$, the subproblem has the closed-form solution

$$
\overline d_j=\frac{u_j}{\|u_j\|_2}.
$$

Indeed, its objective is
$\|R_{-j}\|_F^2+s_j-2d^\top u_j$ on the unit sphere, so minimizing it
amounts to maximizing $d^\top u_j$. The update therefore points each atom
toward the coefficient-weighted residual after accounting for that atom's
existing contribution. The sufficient statistics are computed from selected
support entries; dense coefficient-covariance matrices are unnecessary.

We restrict updates to sufficiently observed atoms. Let
$n_j=\sum_i\mathbf 1[C_{j,i}\ne0]$ be the active selection count in the update
window. Candidate atoms satisfy $n_j\geq n_{\min}$, and a maximum of
$B_{\max}$ candidates are retained in descending count order with deterministic
tie handling. Candidates with negligible coefficient energy or negligible
$\|u_j\|_2$ keep their current values. Let $\mathcal J$ denote the remaining
eligible atoms.

All targets are computed from the same pre-update dictionary and residual.
Because these atoms are subsequently moved together, independently optimal
coordinate targets need not reduce the joint objective when applied at full
strength. We therefore form normalized relaxed candidates,

$$
d_j(\eta)
=
\frac{(1-\eta)d_j+\eta\overline d_j}
{\|(1-\eta)d_j+\eta\overline d_j\|_2},
\qquad j\in\mathcal J,
$$

with other columns unchanged. Starting from $\eta_0\in(0,1]$, we test
$\eta=\eta_0,\eta_0/2,\ldots,\eta_0/2^{B_{\mathrm{bt}}}$, where
$B_{\mathrm{bt}}$ bounds the number of backtracking reductions. Let
$\Delta D(\eta)=D(\eta)-D$. The candidate error is evaluated exactly for the
stored codes as

$$
F(D(\eta);H,C)
=
\|R-\Delta D(\eta)C\|_F^2.
$$

We accept the first candidate satisfying

$$
F(D(\eta);H,C)
\leq
F(D;H,C)+\varepsilon_{\mathrm{acc}}
\max\!\left(F(D;H,C),1\right),
$$

where $\varepsilon_{\mathrm{acc}}$ is a numerical tolerance. If no candidate
passes, the dictionary is unchanged. The global check includes interactions
among simultaneously updated atoms. The construction preserves unit-norm
columns for nondegenerate candidates and controls the fixed-code reconstruction
error on the recorded update window. It does not imply monotonic improvement
of the complete learning objective, of subsequent minibatches, or of
reconstructions obtained by rerunning greedy pursuit after the update.

For a distributed update over a pooled batch, counts, coefficient energies,
residual statistics, and reconstruction errors are aggregated over the same
participating vectors. Gathering the recorded vectors and codes before applying
the update is equivalent to aggregating these quantities. A single accepted
dictionary is then shared across workers. The acceptance test refers to that
pooled update window.

Algorithm 2 specifies the dictionary step. All targets and all backtracking
trials use the same input dictionary, recorded vectors, and coefficients.
Here, $\mathrm{normalize}_{\epsilon}(v)=v/\max(\|v\|_2,\epsilon)$ applies
normalization with a numerical floor. Support slots with zero coefficients are
inactive and do not contribute to usage counts.

~~~text
Algorithm 2: Alternating dictionary update with fixed codes

Input: unit-column dictionary D; recorded vectors H and codes C;
       final supports S; minimum count n_min; candidate cap B_max;
       initial relaxation eta_0; backtracking budget B_bt;
       numerical floor epsilon; acceptance tolerance epsilon_acc.
Output: updated dictionary D_new.

R <- H - D C
E_0 <- squared_Frobenius_norm(R)
For j = 1, ..., M:
    n[j] <- number of i with C[j, i] != 0
A <- indices with n[j] >= n_min, sorted by decreasing n[j]
     (break ties by increasing index)
A <- first min(B_max, length(A)) indices of A
J <- empty

For j in A:
    v <- transpose(C[j, :])
    s <- dot(v, v)
    u <- R v + s D[:, j]
    If s > epsilon and norm(u) > epsilon:
        target[j] <- normalize_epsilon(u)
        Add j to J

If J is empty: return D

For b = 0, ..., B_bt:
    eta <- eta_0 / 2^b
    Delta <- zeros_like(D)
    In parallel for j in J:
        candidate <- normalize_epsilon(
                         (1 - eta) D[:, j] + eta target[j])
        Delta[:, j] <- candidate - D[:, j]
    E_trial <- squared_Frobenius_norm(R - Delta C)
    If E_trial <= E_0 + epsilon_acc max(E_0, 1):
        Return column_normalize_epsilon(D + Delta)

Return D
~~~

Dictionary initialization and maintenance are separate from the alternating
least-squares step. At initialization, atoms may be sampled from nonzero
training vectors and normalized, with small perturbations when repeated
candidates are needed. During learning, usage is tracked over finite intervals.
An optional replacement rule reinitializes a bounded number of atoms that have
remained unused for a prescribed number of consecutive intervals, prioritizing
the longest-unused atoms. Replacements use normalized recent training vectors
with optional perturbations, or normalized random directions when candidates
are unavailable. Such replacements are not subject to the fixed-code
backtracking test, so its error guarantee applies only to accepted
least-squares updates.

One training iteration consequently holds the dictionary fixed while computing
batch OMP codes, updates the surrounding model through the straight-through and
commitment objectives, and then applies the dictionary step when an update
window is complete. The next sparse inference pass uses the resulting
dictionary and refits the coefficients. The output remains a support with
signed real coefficients; any subsequent conversion to a finite token vocabulary
is a separate operation.

Algorithm 3 summarizes this ordering for a finite update window of $W$
recorded batches. The downstream objective is any differentiable objective of
the surrounding model. Stop-gradient copies preserve the vectors and codes
associated with each forward pass, and only the surrounding model parameters
$\phi$ are passed to the gradient optimizer.

~~~text
Algorithm 3: Online training with alternating sparse inference

Input: surrounding model parameters phi; unit-column dictionary D;
       update-window length W; commitment weight beta;
       settings for Algorithms 1 and 2.

window <- empty
For each optimization step:
    H <- encoder_phi(current_minibatch)
    H_record <- stop_gradient_copy(H)
    With gradient recording disabled:
        S, C, P <- BatchOMP(H_record, D)                 // Algorithm 1
    C_record <- copy(C)
    S_record <- copy(S)

    H_ST <- H + stop_gradient(P[K] - H)
    L_commit <- beta / (N m K)
                * sum over k=1,...,K of
                  squared_Frobenius_norm(H - stop_gradient(P[k]))
    L <- downstream_objective(phi, H_ST) + L_commit
    phi, updated <- optimizer_step(phi, L)              // D is fixed
    If not updated:
        Clear window
        Continue

    Append (H_record, C_record, S_record) to window
    If length(window) = W:
        H_window, C_window, S_window <- concatenate recorded batches
        If using a distributed pooled update:
            Pool these records over participating workers
        D <- FixedCodeDictionaryUpdate(
                 D, H_window, C_window, S_window)       // Algorithm 2
        Share the accepted D with participating workers if needed
        Clear window

    Apply optional unused-atom maintenance and synchronize D if needed
~~~

Here, the current minibatch includes every microbatch contributing to one
optimizer step. The pseudocode uses continuous coefficients and progressive commitment
supervision. Retaining only the final prefix yields the final-depth commitment
variant. An incomplete window is deferred until $W$ records are available;
unused-atom maintenance remains separate from the accepted least-squares step.
