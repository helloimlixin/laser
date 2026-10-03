"""Experimental candidate-only extension of the immutable Church vector teacher."""
import hashlib

import torch

from .combination_search import POLICY_VERSION, search_combinations, wide_pair_pool


class ExpandedVectorTeacher:
    """Preserve baseline contexts/events; append four joint-support proposals.

The context sampler retains its exact RNG stream. The loss remains one finite
complete-vector likelihood with the same kernel width. Additional candidates
are sampled using a separate derived RNG and each uses its own causal prefix in
the existing model. No auxiliary loss or coefficient fitting is introduced.
    """
    def __init__(self, aux, plan, *, baseline_class=None, target_builder=None):
        if baseline_class is None:
            from fast_teacher_cached_norm import CachedNormVectorTeacher
            baseline_class = CachedNormVectorTeacher
        if target_builder is None:
            from vector_targets import build_vector_targets
            target_builder = build_vector_targets
        self.baseline = baseline_class(aux, plan)
        self.target_builder = target_builder
        self.plan = dict(plan)
        self.dictionary = self.baseline.dictionary
        self.scales = self.baseline.scales
        self.bins = self.baseline.bins
        self.extra = plan.get('combination_search_extra_candidates', 4)
        self.alternatives = plan.get('combination_search_alternatives_per_depth', 2)
        self.width = plan.get('combination_search_beam_width', 16)
        self.quota = plan.get('combination_search_support_quota', self.width // 2)
        if self.extra != 4:
            raise ValueError('the registered ablation adds exactly four complete candidate slots')

    @torch.no_grad()
    def __call__(self, atoms, normalized_coefficients, signals=None, generator=None):
        if generator is None:
            raise ValueError('an explicit baseline generator is required to preserve context RNG')
        digest = hashlib.sha256(generator.get_state().numpy().tobytes()).digest()
        seed = int.from_bytes(digest[:8], 'little') & ((1 << 63) - 1)
        candidate_generator = torch.Generator(device=atoms.device).manual_seed(seed)
        original, telemetry = self.baseline(atoms, normalized_coefficients,
                                            signals=signals, generator=generator)
        physical = normalized_coefficients.to(self.dictionary) * self.scales
        pool = wide_pair_pool(atoms, physical, self.dictionary, self.bins,
            temperature=self.plan['teacher_temperature'], alternatives_per_depth=self.alternatives,
            uniform_mixture=self.plan.get('teacher_proposal_uniform_mixture', .001),
            site_chunk_size=self.plan.get('combination_search_site_chunk_size', 128),
            generator=candidate_generator)
        bank = search_combinations(original.target_vectors, self.dictionary,
            pool.atoms, pool.coefficient_ids, self.bins, beam_width=self.width,
            support_quota=self.quota,
            site_chunk_size=self.plan.get('combination_search_site_chunk_size', 128))
        extra_atoms, extra_bins, usable = bank.sample_novel_supports(atoms,
            self.plan['teacher_temperature'], draws=self.extra, generator=candidate_generator)
        # The baseline teacher always appends four stochastic tuples after its
        # 24 permutation slots. Retain them, including the original context.
        targets = self.target_builder(atoms, physical, self.dictionary, self.bins,
            context_atoms=original.context_atoms,
            context_coefficient_ids=original.context_coefficient_ids,
            extra_atoms=torch.cat((original.atoms[..., -4:, :], extra_atoms), -2),
            extra_coefficient_ids=torch.cat((original.coefficient_ids[..., -4:, :], extra_bins), -2),
            site_chunk_size=self.plan.get('vector_geometry_site_chunk_size', 256))
        # Missing novel supports are padding, never new training events.
        targets.valid[..., -self.extra:] &= usable
        telemetry = dict(telemetry)
        telemetry.update(valid_candidates=targets.valid.float().sum(-1).mean(),
            expanded_usable_candidates=usable.float().sum(-1).mean(),
            expanded_candidate_error=(targets.errors[..., -self.extra:] * usable).sum() / usable.sum().clamp_min(1),
            expanded_candidate_kernel_mass=(torch.exp(-targets.errors[..., -self.extra:]
                / self.plan['vector_kernel_temperature']) * usable).sum(-1).mean(),
            expanded_search_evaluations=bank.evaluated_candidates.float().mean())
        return targets, telemetry

    def metadata(self):
        metadata = dict(self.baseline.metadata())
        metadata.update(candidate_policy=POLICY_VERSION,
            candidate_set='all baseline24 permutations and4 stochastic events, plus4 distance-weighted distinct-support complete-vector proposals',
            scored_candidate_slots=32,
            baseline_context_and_generator_stream_preserved=True,
            candidate_rng='SHA256-derived independent generator from pre-call baseline RNG state',
            additional_losses=False, no_coefficient_refitting=True,
            candidate_alternatives_per_depth=self.alternatives,
            candidate_beam_width=self.width, candidate_support_quota=self.quota,
            candidate_search='two-slot replacements in unobserved training candidates; full-vector errors, support-stratified bounded beam; finite coverage')
        return metadata
