import pytest
import torch

from src.models.laser_tts import LaserTTS, TTSConfig
from src.tts_pairing import batch_chain, common_state, state_sha


@pytest.fixture(params=['laser', 'rvq'])
def prior(request):
    torch.manual_seed(4)
    return LaserTTS(TTSConfig(phone_vocab=12, speakers=2, width=32, heads=4,
        text_layers=1, audio_layers=1, dropout=0, codec=request.param, depth_layers=2)).eval()


def test_depth_teacher_forcing_has_no_current_or_future_field_leakage(prior):
    context = torch.randn(2, 3, 32)
    teacher = torch.tensor([1, 60, 2, 64]).expand(2, 3, 4).clone()
    expected = prior.depth_logits(context, teacher)
    for changed_field in range(4):
        changed = teacher.clone()
        changed[..., changed_field] += 1
        actual = prior.depth_logits(context, changed)
        for d in range(changed_field + 1):
            torch.testing.assert_close(actual[d], expected[d])


def test_cached_depth_matches_parallel_teacher_forcing(prior):
    context = torch.randn(2, 3, 32)
    teacher = torch.tensor([1, 60, 2, 64]).expand(2, 3, 4).clone()
    expected = prior.depth_logits(context, teacher)
    caches = [{} for _ in prior.depth_blocks]
    fields = []
    for d in range(4):
        actual = prior.depth_next_logits(context.flatten(0, 1), fields, caches)
        if d == 2 and prior.cfg.codec == 'laser':
            actual[:, 1] = -1e4
        torch.testing.assert_close(actual.reshape_as(expected[d]), expected[d], atol=1e-5, rtol=1e-5)
        fields.append(teacher[..., d].flatten())


def test_generation_respects_each_codec_vocabulary_and_eos(prior):
    with torch.no_grad():
        for head in prior.heads:
            head.weight.zero_(); head.bias.fill_(-100); head.bias[0] = 100
    codes, info = prior.generate(torch.tensor([[3, 2]]), torch.tensor([0]),
        min_frames=3, max_frames=3, temperature=0)
    assert codes.shape == (3, 4) and not info['eos_reached']
    if prior.cfg.codec == 'rvq':
        assert (codes == 0).all(), 'Different RVQ codebooks may emit the same integer'
    else:
        assert (codes[:, 0] != codes[:, 2]).all()
    with torch.no_grad():
        prior.heads[0].bias[prior.EOS] = 200
    codes, info = prior.generate(torch.tensor([[3, 2]]), torch.tensor([0]),
        min_frames=0, max_frames=3, temperature=0)
    assert codes.shape == (0, 4) and info['eos_reached']


def test_shared_weight_copy_includes_depth_transformer(prior):
    other = LaserTTS(TTSConfig(**{**vars(prior.cfg), 'codec': 'rvq' if prior.cfg.codec == 'laser' else 'laser'}))
    other.load_state_dict(common_state(prior), strict=False)
    assert state_sha(common_state(prior)) == state_sha(common_state(other))
    assert any(k.startswith('depth_blocks.') for k in common_state(prior))


def test_data_chain_detects_reordered_utterances():
    b = {k: torch.tensor([1, 2]) for k in ('indices', 'phones', 'lengths', 'text_lengths', 'speakers')}
    changed = {**b, 'indices': torch.tensor([2, 1])}
    assert batch_chain('00'*32, b) != batch_chain('00'*32, changed)
