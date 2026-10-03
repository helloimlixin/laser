The user clarified that both stochastic sampling and soft labeling must account for the vector reconstructed by the whole sparse combination. A change to spatial input embeddings alone does not satisfy this requirement. This document records the target design; it is not an implemented or validated training recipe.

For a site code S=((a_1,c_1),...,(a_K,c_K)), write v(S)=sum_d c_d D[a_d], with coefficients in physical latent units. A candidate combination-aware teacher is

    q(S | z) proportional to mu(S) * exp(-||z - v(S)||^2 / tau).

Here mu specifies valid sequences and the base measure. It must be explicit: uniform weighting of token sequences need not imply uniform weighting of reconstructed vectors, because some vectors have more equivalent encodings. Existing coefficient grids, depth scales, support constraints and any preference over partial reconstructions belong in the specification; they should not be changed implicitly.

Stochastic training sequences and soft labels must come from this same teacher. At depth d, condition on the actual sampled prefix h=S_<d. The next-pair conditional sums the weight of valid remaining completions:

    q(a_d,c_d | h,z)
      proportional to sum_suffix mu(h,(a_d,c_d),suffix)
        * exp(-||z - v(h) - c_d D[a_d] - v(suffix)||^2 / tau).

An exact factorization into existing heads is q(a_d|h,z) and q(c_d|a_d,h,z). Sample the atom from the first distribution, sample its coefficient from the second, and supervise each head with its matching soft conditional. These are coupled marginals of one joint target. Independent smoothing of cached atom/coeff labels does not implement this teacher. An extra unequal atom-loss multiplier changes the joint-likelihood objective and must be treated as an explicit departure.

Previously sampled atom/coefficient pairs remain frozen. Sampling or marginalizing possible suffixes does not require refitting earlier coefficients. Candidate generators must also obey this restriction; no least-squares revision of an emitted prefix is permitted.

The earlier q(a,c|r) proportional to exp(-||r-cD[a]||^2/tau) already used the physical contribution of a pair. Its limitation is that it scores immediate residual error without accounting for the quality or multiplicity of possible complete sparse reconstructions. Original RQ uses a greedy residual-conditioned codeword teacher; the joint-combination target above is a principled extension rather than an exact reproduction of that algorithm. Reference: https://arxiv.org/html/2203.01941#S3.SS2.SSS3.

A combined-vector MSE on the model's mean prediction is also insufficient to specify this distribution: distinct incorrect predictions can average to the right mean. The intended teacher should retain valid alternative combinations while assigning labels and samples consistently.

Exact marginalization over the full compound vocabulary is expensive. A finite candidate-completion distribution can make the conditional computations exact within that candidate set, while remaining an approximation to the intended full teacher. Candidates must remain consistent across depths; conditioning cannot silently switch to incompatible suffix targets. Proposal probabilities, candidate coverage, coefficient noise, reconstructed-vector distortion and prefix invariance require explicit validation before any paid training launch.

The physical-spatial-context proposal remains an optional separate architecture ablation. It is superseded as a complete response to the user's teacher requirement. No production training, refitting, replacement teacher, or new training launch was performed by this clarification. Existing FID/generalization measurements do not establish that this missing teacher mechanism is the unique cause of the plateau.
