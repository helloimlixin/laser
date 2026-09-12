import torch

from src.models.laser_tts import LaserTTS, TTSConfig, position_encoding
from src.tts_data import FrameBatchSampler, collate_tts, text_split


def model():
    torch.manual_seed(12)
    return LaserTTS(TTSConfig(phone_vocab=12, speakers=3, width=32, heads=4,
                              text_layers=1, audio_layers=2, dropout=0)).eval()


def batch():
    codes = torch.tensor([[1, 60, 2, 65], [3, 64, 4, 62], [5, 55, 6, 70]])
    return collate_tts([{'codes': codes, 'phones': torch.tensor([3, 4, 2]), 'speaker': 0},
                        {'codes': codes[:1], 'phones': torch.tensor([5, 2]), 'speaker': 1}])


def test_temporal_causality_and_cached_generation_equivalence():
    m, b = model(), batch()
    with torch.inference_mode():
        memory, mask = m.text_memory(b['phones'], b['speakers'])
        full, _ = m.temporal(b['codes'], memory, mask, b['speakers'])
        changed = b['codes'].clone(); changed[:, 2, 0] = 123
        other, _ = m.temporal(changed, memory, mask, b['speakers'])
        torch.testing.assert_close(full[:, :3], other[:, :3])
        caches = [{} for _ in m.blocks]
        static = [block.cross_attention.kv(memory) for block in m.blocks]
        outputs = []
        for t in range(4):
            x = m.bos.expand(2, -1, -1) if t == 0 else m.frame_embedding(b['codes'][:, t-1:t])
            x = x + position_encoding(1, 32, x.device, t) + m.speakers(b['speakers'])[:, None]
            for block, cache, kv in zip(m.blocks, caches, static):
                x, _ = block(x, memory, mask, cache=cache, static_kv=kv)
            outputs.append(m.out_norm(x))
        torch.testing.assert_close(torch.cat(outputs, dim=1), full, atol=1e-5, rtol=1e-5)


def test_padding_is_not_a_training_target():
    m, b = model(), batch()
    first = m(b)['nll']
    changed = {k: v.clone() for k, v in b.items()}
    changed['codes'][1, 1:, 0] = 42
    changed['codes'][1, 1:, 2] = 43
    torch.testing.assert_close(first, m(changed)['nll'])


def test_text_and_speaker_receive_gradients():
    m, b = model(), batch()
    result = m(b, guide_weight=.2)
    result['loss'].backward()
    assert torch.isfinite(result['loss'])
    assert m.phones.weight.grad[3:].abs().sum() > 0
    assert m.speakers.weight.grad.abs().sum() > 0
    changed = {k: v.clone() for k, v in b.items()}
    changed['phones'][0, 0] = 8
    assert abs(float(m(changed)['nll'] - m(b)['nll'])) > 1e-7


def test_eos_terminates_without_emitting_invalid_codec_fields():
    m = model()
    with torch.no_grad():
        m.heads[0].weight.zero_(); m.heads[0].bias.fill_(-100)
        m.heads[0].bias[m.EOS] = 100
    codes, info = m.generate(torch.tensor([[3, 2]]), torch.tensor([0]), min_frames=0, max_frames=5, temperature=0)
    assert codes.shape == (0, 4) and info['eos_reached']


def test_transcript_groups_and_frame_batches():
    assert text_split('Hello, world!') == text_split('HELLO WORLD.')
    lengths = [3, 50, 12, 90, 8, 4, 20]
    sampler = FrameBatchSampler(lengths, frame_budget=100, max_batch=3)
    batches = list(sampler)
    assert sorted(i for group in batches for i in group) == list(range(len(lengths)))
    assert all(max(lengths[i] for i in group) * len(group) <= 100 for group in batches)
