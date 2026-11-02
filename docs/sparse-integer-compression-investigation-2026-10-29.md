# Sparse codes as integer sequences: investigation, 2026-10-29

The existing adaptive joint atom/coefficient quantizer is the best established
route to a modest vocabulary increase. On the historical ImageNet 8x8, K=4,
16,384-atom tokenizer, a 20,481-symbol vocabulary is 25% larger than the original
atom alphabet and transmits 256 integers per image. Its measured full-validation
reconstruction FID is 4.6894, compared with 4.2150 for the source tokenizer.
This is **lossy re-encoding**, including a learned replacement dictionary with
8,192 directions; it does not preserve every original support/coefficient pair.

For higher fidelity at the same number of integers, the later 32,768-symbol
free-vector residual book reached 4.4778 rFID. Keeping the original sparse support
and quantizing coefficients to 32 bins did better at 4.2516, but needs two
component integers per contribution or a much larger packed alphabet.

The results here concern the historical ImageNet tokenizer with source checkpoint
SHA256 `dd28db9306d526bdc9fbf8016403e106c4f317fd8f5c04792f0953e53310fdab`.
They should not be transferred numerically to the newer multiscale VAR experiments.

## Evidence checked

Read-only W&B access retrieved 300 recent summaries from
[`helloimlixin-rutgers/laser`](https://wandb.ai/helloimlixin-rutgers/laser), spanning
2026-09-30 through 2026-10-29, plus full configurations for four focused runs.
Local codebooks, fitting implementations, reconstruction reports, and the
low-bit coefficient study were also inspected. Snapshots and fresh verification
results are under
[`outputs/sparse-integer-investigation-20261029`](../outputs/sparse-integer-investigation-20261029/).
The API credential was not written to these artifacts.

The principal W&B evidence is:

- [20,481-class, four-depth training run](https://wandb.ai/helloimlixin-rutgers/laser/runs/imagenet-laser-joint20481-d4-rqrecipe-scratch100-8a100-20261027):
  records the source and compressed tokenizer identities and 4.6894 rFID.
- [Joint/neural rate-distortion study](https://wandb.ai/helloimlixin-rutgers/laser/runs/imagenet-laser-joint-neural-rate-distortion-20261024):
  matched four-depth 8k/16k/32k vector books and 32k/49k/65k scaled-atom controls.
- [Entropy-constrained dictionary study](https://wandb.ai/helloimlixin-rutgers/laser/runs/imagenet-laser-entropy-constrained-rd-20261024):
  tests rate penalties and coordinate assignment refinement.
- [Low-bit coefficient study](https://wandb.ai/helloimlixin-rutgers/laser/runs/imagenet-rqmatched-lowbit-20261027):
  reports 32 coefficient bins passing the paired full-50k reconstruction gate.
- [Exactly shared physical vocabulary study](https://wandb.ai/helloimlixin-rutgers/laser/runs/imagenet-laser-sharedjoint20k-depth4-codebook-20261022):
  separates one shared physical dictionary from depth-specific lookup tables.

## What a small integer alphabet can represent

A sparse code has support `a[k]` and real values `c[k]`, reconstructing
`z = sum_k c[k] * D[:, a[k]]`. Atom IDs alone omit the coefficients. With a finite
vocabulary and bounded sequence length, arbitrary real coefficients cannot be
represented exactly. Exact preservation of stored floating-point values requires
retaining their bits, potentially with lossless entropy coding.

After scalar quantization to L coefficient levels, the pair can be packed as
`t = a * L + b`. This is one integer but has M*L possible values; relabeling does
not reduce the pair's information content. For M=16,384 and L=32, the packed
alphabet has 524,288 symbols and each pair takes 19 fixed-width bits. A flat
524,288-way classifier would be 32 times the original atom classifier width.
Factorized atom and conditional-coefficient heads avoid that output-layer growth.

Keeping the two fields in a serialized sequence instead needs 512 component
integers per image. Disjoint atom/bin token ranges need only M+L=16,416 symbols;
with known alternating positions, the two fields can even reuse integer labels.
That serialized sequence length should not be confused with the existing
compound model's transformer context: it can make both decisions within one
pair event rather than doubling its spatial/body sequence length.

There is also a simple constraint on single-contribution scaled-atom books:
retaining every original direction with at least one positive and one negative
coefficient needs at least 2M nonzero entries. With M=16,384, a 20k alphabet cannot
offer both signs on every original direction in that interface. It must replace
or reduce directions, restrict coefficient choices, or spend more tokens.

## The working compact construction

[`JointAtomCoefficientRQ`](../src/models/laser_joint_tokenizer.py) stores an atom ID
and coefficient for each token and residual depth:

`e_d(t) = coefficient[d,t] * dictionary[:, atom_id[d,t]]`.

At each site, initialize `r = target`; select the codeword minimizing
`||r - e_d(t)||^2`, subtract it, and continue for K depths. Decode using only
`sum_d e_d(t[d])`. There are no transmitted continuous coefficients. Depth is
known from sequence position and costs no extra per-image index.

For a non-unit atom, exact assignment maximizes
`2*c*dot(r,D[:,a]) - c*c*||D[:,a]||^2`; the null token scores zero.
Earlier selected contributions stay fixed during this greedy encoding pass.
This is a residual-quantization interface, consistent with the stacked discrete
representation in [Lee et al., CVPR 2022](https://arxiv.org/abs/2203.01941).

The saved 20,481-entry book allocates coefficient levels unevenly:

| Levels per direction | Number of directions |
| --- | ---: |
| 2 | 6,190 |
| 4 | 1,957 |
| 6 | 44 |
| 8 | 1 |

Thus there are 8,192 directions, 20,480 nonzero joint entries, and one null ID:
an average of 2.5 signed levels per direction. The allocator spends extra entries
where training coefficient distortion drops most. Offline fitting alternates
assignment with dictionary/scalar updates against final summed latent error.
Implementations are in
[`laser_joint_vocabulary.py`](../src/training/laser_joint_vocabulary.py),
[`laser_compact_joint_vocabulary.py`](../src/training/laser_compact_joint_vocabulary.py),
and, for physically identical codewords across depths,
[`laser_shared_joint_vocabulary.py`](../src/training/laser_shared_joint_vocabulary.py).

These are two distinct goals:

- To compress an existing sparse reconstruction, fit/encode `target = D*c`.
- To build an alternative image tokenizer, fit/encode the pre-bottleneck encoder
  latent. The historical joint-vocabulary studies use this route and may change
  the support entirely. They do not establish error bounds on the original c.

## Measured tradeoffs

All rFID figures below use 50,000 reconstructed ImageNet validation images and
the original RQ Inception metric. They span several studies, not one controlled
sweep of all methods. The free-vector 16k/32k and scaled-atom 49k rows come from
the same October 24 study. Image encoder/decoder weights were frozen.

| Representation | Alphabet | Integers/image | Nominal bits/image | rFID |
| --- | ---: | ---: | ---: | ---: |
| Original support + 32-bin coefficient | 16,384 atom IDs; 32 bins | 512 component IDs, or 256 packed pairs | 4,864 | 4.2516 |
| Adaptive scaled atoms, depth-specific values, K=4 | 20,481 | 256 | 3,840 | 4.6894 |
| Adaptive scaled atoms, depth-specific values, K=5 | 20,481 | 320 | 4,800 | 4.1924 |
| Adaptive scaled atoms, depth-specific values, K=6 | 20,481 | 384 | 5,760 | 4.0409 |
| Exactly shared physical scaled-atom book, refitted K=4 | 20,481 | 256 | 3,840 | 4.9503 |
| Free vector tables, K=4 | 16,384 | 256 | 3,584 | 4.7448 |
| Free vector tables, K=4 | 32,768 | 256 | 3,840 | 4.4778 |
| Original-direction scaled-atom book, K=4 | 49,153 | 256 | 4,096 | 4.4354 |

Nominal sizes use `ceil(log2(V))` bits per joint token and exclude shared model
parameters, headers, and side information. They are not measured compressed-file
sizes. A 20,481-symbol stream costs the same 15 fixed-width bits per token as a
32,768-symbol stream. Adding a fifth depth spends 25% more tokens; its 4.1924
rFID is not a four-token result. The physically shared five-depth construction
is another book, with 4.3224 rFID, not 4.1924.

The original 20k book shares the atom/level layout across depth, but **not the
physical vectors**: its coefficient tensor is `[4,20481]` with different values
at each depth. The free-vector 32k book goes further, storing four unconstrained
`[32768,256]` tables. That is 128 MiB in FP32, versus 16 MiB for the original
`[256,16384]` direction matrix. Its token alphabet doubles, but its dense vector
storage is eight times the original matrix. It also no longer guarantees a
K-sparse representation in the original dictionary.

The compact factorized 20k book occupies approximately 8.94 MiB of tensor data
because directions are reused across scalar levels; it need not materialize
every expanded vector for assignment.

## What earlier attempts rule out

- The 32-bin coefficient study increased rFID by only 0.0360 versus its 512-bin
  control. This is the strongest checked option for preserving original supports.
- A small neural decoder did not improve the October 24 16k/32k books' rFID:
  4.7448 to 4.7696 at 16k and 4.4778 to 4.5051 at 32k, despite lower latent MSE.
  The tested neural repair is not a demonstrated substitute for vocabulary size.
- A separate learned 16k scalar-tuple codec reached approximately 7.75 rFID;
  its much worse result does not establish that every 16k representation fails.
- In the entropy-constrained study, a 16k book scored 4.8099 without a rate
  penalty and 4.8115 with beta=5e-5 plus two coordinate sweeps. Held-out model
  cost changed from 13.8804 to 13.8744 bits/token: little rate reduction in the
  tested regime. This does not establish a general limit on learned AR coding.
- The 20k redundancy audit found no codeword neighbors within 10% relative
  physical-vector RMSE; almost every ID was used at each depth. Frequency pruning
  or merging near-duplicates is not supported as a large, low-loss reduction.

Generation quality is a separate result. The completed 49,153-class, 20-epoch
[prior run](https://wandb.ai/helloimlixin-rutgers/laser/runs/imagenet-laser-joint49153-rfid4436-scratch20-8h100-20261023)
reports FID50k 31.2756. The 20,481-class K=4 run's retrieved summary reports
46.7840 under its benchmark sampler and 38.2776 under its tuned sampler. These
are different training/sampling conditions, not a matched vocabulary ablation.
Neither reconstruction rFID nor a short causal coding probe proves better
generation or an actual entropy-coded bitrate.

## Recommendation

1. If a 25% alphabet increase and 256 integers/image are the primary constraints,
   use the established **20,481-entry adaptive joint construction** as the
   baseline. Budget explicitly for its measured reconstruction loss and its
   learned 8,192-direction replacement dictionary.
2. If retaining reconstruction fidelity is more important, retain the original
   supports with **32 coefficient bins and factorized predictions**. A packed
   integer is still possible, but its alphabet is 524,288, not approximately 20k.
   This already has stronger reconstruction evidence than the compact books.
3. If 2x alphabet growth is acceptable, the **32,768-entry free-vector book** is
   the best checked four-token result below 4.5 rFID. Account for its four dense
   depth tables and changed sparse semantics.
4. The useful untested middle ground is a matched **20,480 / 24,576 / 32,768**
   four-depth sweep, comparing adaptive scaled atoms with free vectors, first
   on fixed training/heldout latents. Include an exactly shared-codeword arm only
   if identical vector meaning at every depth is required. Existing results do
   not establish how the free-vector method behaves at 20k or 24k.

For that next sweep, fix the source checkpoint, 256 integer budget, image
transforms, fitting images, and decoder; select on held-out training data. Report
validation rFID, paired PSNR/LPIPS, latent distortion, dictionary bytes, and
held-out code length. If compression of the existing sparse code is the goal,
fit against `D*c` and additionally report its distortion and support changes.
Generation comparisons need their own matched prior budget and sampler.

## Fresh verification

The saved adaptive 20k book was loaded on CPU. On 16 synthetic 256-dimensional
signals, its greedy IDs exactly matched explicit expanded-codebook nearest
neighbors at all four depths. Summing lookups from its integer IDs reproduced
the quantizer output within `2.3842e-7` maximum absolute error. This checks the
integer interface, not image reconstruction quality.

Seventeen existing focused tests passed for adaptive allocation/refitting,
physical sharing, integer residual books, and entropy-constrained assignment.
An unrelated report-writer multiprocessing test was excluded. Pytest was
installed under `/tmp` for this check; the environment lacks the dependencies
needed for the broader upstream integration suite. No model training or new
image-quality evaluation was performed during this investigation.
