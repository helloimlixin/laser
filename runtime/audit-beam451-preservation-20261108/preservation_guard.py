"""Fail closed if the audited frozen representation or teacher bound changes."""
import torch

EXPECTED_DIGEST='1dae97e3fa2302799c8f02f6a5f8ca0e950ed88235bdce39e84a621369a14dc2'


@torch.no_grad()
def verify_frozen_codec(profile,aux):
    assert not aux.training and all(not module.training for module in aux.modules())
    assert all(not parameter.requires_grad for parameter in aux.parameters())
    assert profile.digest(aux)==EXPECTED_DIGEST, 'Frozen Stage 1 codec differs from the full-validation reference'
    assert aux.reconstruction_beam_width==16 and aux.physical_book.encoding_beam_width==16
    assert aux.physical_book.vocab_size==16384 and aux.physical_book.depth==4
    for depth,table in enumerate(aux.quantizer.codebooks):
        torch.testing.assert_close(table.weight[:-1],aux.physical_book.expanded_codebook(depth),rtol=0,atol=0)
    return dict(stage1_preservation_guard_passed=True,frozen_tokenizer_digest=EXPECTED_DIGEST,
                stage1_frozen=True,physical_and_native_codewords_bitwise_equal=True)


@torch.no_grad()
def verify_teacher(pool,weights,chosen):
    row=torch.arange(len(weights),device=weights.device)
    canonical_cost=pool['costs'][row,pool['canonical_index']]
    excess=(pool['costs']-canonical_cost[:,None]).clamp_min(0)
    assert torch.isfinite(weights).all() and (weights>=0).all()
    assert ((weights==0)|(excess<=.01*canonical_cost[:,None].clamp_min(1e-8)+1e-6)).all(), 'Teacher exceeds verified latent-error bound'
    relative=excess/canonical_cost[:,None].clamp_min(1e-8)
    return dict(teacher_preservation_guard_passed=True,
        expected_relative_latent_error=float((relative*weights).sum(-1).mean()),
        sampled_relative_latent_error=float(relative[row,chosen].mean()),
        maximum_supported_relative_latent_error=float(relative.masked_fill(weights==0,0).max()))
