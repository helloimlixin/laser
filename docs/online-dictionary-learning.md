# Online Dictionary Learning

Our algorithm is an **alternating online dictionary learner**: it infers sparse representations with batch orthogonal matching pursuit (OMP), then updates the dictionary using coefficient-weighted residuals. The dictionary update uses a relaxed step and accepts it only when the reconstruction error on the current batch does not increase beyond numerical tolerance.

This description uses general dimensions and hyperparameters while preserving the algorithm used in the ImageNet run with reconstruction FID 4.210914.

Let

$$
H=[h_1,\ldots,h_N]\in\mathbb R^{m\times N}
$$

contain a minibatch of encoder-produced vectors, and let

$$
D=[d_1,\ldots,d_M]\in\mathbb R^{m\times M},
\qquad \|d_j\|_2=1,
$$

be a shared dictionary. Each vector is approximated by a sparse linear combination,

$$
h_i\approx Dc_i,
\qquad \|c_i\|_0\le K.
$$

The coefficients are signed real numbers. The dictionary can be overcomplete, and its atoms can be correlated. The underlying sparse reconstruction objective is

$$
\min_{D,C}\frac12\|H-DC\|_F^2
\quad\text{subject to}\quad
\|c_i\|_0\le K,\qquad \|d_j\|_2=1.
$$

We alternate between estimating $C$ with $D$ fixed and updating $D$ with the inferred codes fixed.

**Sparse inference proceeds through $K$ OMP selections.** For each vector, we begin with an empty support and residual $r_i^{(0)}=h_i$. At pursuit depth $k$, we select the unused atom with the largest absolute residual correlation:

$$
j_i^{(k)}
=
\arg\max_{j\notin S_i^{(k-1)}}
\left|d_j^\top r_i^{(k-1)}\right|.
$$

The selected atom is appended to the support,

$$
S_i^{(k)}=S_i^{(k-1)}\cup\{j_i^{(k)}\}.
$$

Absolute correlation permits either positive or negative coefficients. Masking previously selected atoms ensures that each support contains distinct indices.

After every selection, we jointly refit **all coefficients on the active support**:

$$
\gamma_i^{(k)}
=
\arg\min_\gamma
\left\|h_i-D_{S_i^{(k)}}\gamma\right\|_2^2.
$$

We then form the reconstruction and residual,

$$
\widehat h_i^{(k)}
=
D_{S_i^{(k)}}\gamma_i^{(k)},
\qquad
r_i^{(k)}
=
h_i-\widehat h_i^{(k)}.
$$

Consequently, adding an atom can change the coefficients of earlier selections. For a nonsingular selected system in exact arithmetic, the fitted residual is orthogonal to the selected span.

To implement this efficiently across a batch, we precompute

$$
G=D^\top D,
\qquad
B=D^\top H.
$$

The active least-squares systems become

$$
G_{S_i^{(k)},S_i^{(k)}}\gamma_i^{(k)}
=
B_{S_i^{(k)},i}.
$$

Each vector maintains an incremental Cholesky factor of its selected Gram matrix. Adding an atom extends this small factor, and triangular solves recover the refitted coefficients. Residual correlations are updated using

$$
D^\top r_i^{(k)}
=
B_{:,i}
-
G_{:,S_i^{(k)}}\gamma_i^{(k)}.
$$

The Gram matrix is shared across the batch, while supports and coefficient solves are independent for each vector. The reference implementation uses FP32 pursuit arithmetic and a small numerical floor on Cholesky pivots.

**The surrounding model learns through a straight-through reconstruction and progressive commitment loss.** OMP runs without differentiating through atom selection or coefficient fitting. The decoder receives the final sparse reconstruction through

$$
h_i^{\mathrm{ST}}
=
h_i+\operatorname{sg}
\left(\widehat h_i^{(K)}-h_i\right),
$$

where $\operatorname{sg}$ denotes stop-gradient. Its forward value equals the sparse reconstruction, while its surrogate derivative with respect to the encoder output is the identity. Downstream reconstruction, perceptual, and adversarial objectives can therefore train the encoder.

Progressive commitment supervision encourages encoder outputs to be approximated well at every pursuit depth:

$$
\mathcal L_{\mathrm{commit}}
=
\frac{\beta}{NmK}
\sum_{i=1}^{N}\sum_{k=1}^{K}
\left\|
h_i-\operatorname{sg}\left(\widehat h_i^{(k)}\right)
\right\|_2^2.
$$

Here, $\beta$ represents the effective commitment weight. Each prefix reconstruction uses the coefficients fitted at that depth. This matters because truncating the final coefficient vector generally gives a different reconstruction.

The dictionary remains fixed during the gradient update of the surrounding model. Its explicit update uses the **final-depth codes**, even though the encoder receives supervision from all pursuit depths.

**After the model optimizer step, we compute residual-based dictionary targets.** We retain detached copies of the encoder vectors, supports, and coefficients from the preceding forward pass. These recorded quantities remain paired: the dictionary update uses the vectors that generated the stored codes.

With those codes fixed, define

$$
R=H-DC.
$$

For atom $j$, let $v_j=C_{j,:}^{\top}$ contain its coefficients across the batch. Its coefficient energy and residual correlation are

$$
q_j=v_j^\top v_j,
\qquad
b_j=Rv_j.
$$

Adding back that atom's existing contribution gives

$$
u_j=b_j+q_jd_j
=
\sum_i c_{j,i}\left(r_i+d_jc_{j,i}\right).
$$

The expression $r_i+d_jc_{j,i}$ is the error remaining after accounting for all other atoms. Thus $u_j$ identifies the direction that best explains the portions of the batch assigned to atom $j$.

Holding all other atoms and all coefficients fixed, the unit-norm coordinate target is

$$
\overline d_j
=
\frac{u_j}{\|u_j\|_2},
$$

provided the coefficient energy and target norm are sufficiently large. This follows by expanding the coordinate objective: under $\|d_j\|_2=1$, its variable part is $-2d_j^\top u_j$, which is minimized by aligning $d_j$ with $u_j$.

Updates are restricted to sufficiently observed atoms. We count each atom's occurrences in the selected supports, apply a minimum usage threshold, and optionally cap the candidate set by retaining the most frequently selected atoms. Candidates with negligible coefficient energy or target norm retain their existing values.

These statistics are accumulated directly from sparse support entries and coefficients.

**Eligible atoms move together using a shared relaxation parameter.** All targets are computed from the same pre-update dictionary and residual. For an eligible atom,

$$
d_j(\eta)
=
\frac{(1-\eta)d_j+\eta\overline d_j}
{\left\|(1-\eta)d_j+\eta\overline d_j\right\|_2}.
$$

This interpolates toward the coordinate target and restores unit norm. The parameter $\eta$ controls how strongly the dictionary responds to the current batch.

Although each target is optimal for its individual coordinate, applying several targets simultaneously introduces interactions between atoms. We therefore evaluate the complete candidate dictionary before accepting it.

Let

$$
\Delta D(\eta)=D(\eta)-D.
$$

For the stored coefficients, the candidate reconstruction error is

$$
E(\eta)
=
\left\|R-\Delta D(\eta)C\right\|_F^2.
$$

Starting from an initial relaxation $\eta_0$, we accept the first candidate satisfying

$$
E(\eta)
\le
\|R\|_F^2
+
\varepsilon_{\mathrm{acc}}
\max\!\left(\|R\|_F^2,1\right).
$$

If a candidate fails, we halve the relaxation and retry, up to a bounded number of reductions. If every trial fails, the dictionary is unchanged.

This check controls reconstruction error for the **recorded batch with its codes held fixed**. It does not establish monotonic improvement of the complete neural training objective or of future batches. Subsequent forward passes rerun OMP and refit coefficients using the updated dictionary.

In distributed training, workers sum usage counts, residual correlations, coefficient energies, and candidate reconstruction errors across their local batches. Dictionary targets and acceptance decisions therefore reflect the pooled batch.

**Initialization and unused-atom replacement maintain dictionary coverage.** At initialization, atoms are sampled from nonzero encoder outputs and normalized. When there are fewer distinct candidates than required atoms, sampled directions can be repeated with small perturbations.

During training, usage is monitored over fixed intervals. Atoms that remain unused for a prescribed number of consecutive intervals become eligible for replacement. A bounded fraction is replaced with normalized recent encoder vectors, with perturbations to encourage diversity; random normalized directions provide a fallback when suitable vectors are unavailable. Replacement is a separate maintenance operation from the residual update and its acceptance check.

One training iteration follows this order:

1. Encode the current minibatch and initialize the dictionary if needed.
2. Run batch OMP with the dictionary fixed, retaining each fitted prefix.
3. Record detached encoder vectors and final sparse codes.
4. Update the surrounding model using downstream losses and progressive commitment.
5. Compute dictionary targets from the recorded final-code residuals.
6. Apply a normalized relaxed update, backtracking until the fixed-code error check passes.
7. Perform scheduled unused-atom maintenance.

The method is online because each dictionary step uses the current minibatch's sparse assignments and residuals. The learned dictionary carries information forward across iterations, while codes are inferred afresh whenever a new batch is processed.

This description follows the [preserved reference configuration](../outputs/ffhq-k4-continue150-20260923/online-reference.json) and the [source comparison verifying the reference dictionary updater](../outputs/ffhq-k4-continue150-20260923/online-update-code-comparison.json).
