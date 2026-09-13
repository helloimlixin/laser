import pytest
from scripts.tools.benchmark_mdctcodec_tts import aggregate, edit_distance


def test_word_error_counts_insertions_and_omissions():
    assert edit_distance('a short sentence'.split(), 'a very short sentence today'.split()) == 2
    assert edit_distance('a short sentence'.split(), []) == 3
    assert edit_distance('a short sentence'.split(), 'a short sentence'.split()) == 0


def test_corpus_wer_and_speed_are_weighted_by_words_and_duration():
    common = dict(char_errors=1, characters=10, speaker_similarity_ecapa=.8,
        utmos=3., peak_gpu_allocated_gib=1., truncated_at_15_seconds=False)
    rows = [dict(common, word_errors=1, words=2, synthesis_seconds=2, seconds=1, eos_reached=False),
            dict(common, word_errors=0, words=8, synthesis_seconds=2, seconds=3, eos_reached=True)]
    result = aggregate(rows)
    assert result['wer'] == pytest.approx(.1)
    assert result['rtf'] == pytest.approx(1.)
    assert result['cap_fraction'] == pytest.approx(.5)
