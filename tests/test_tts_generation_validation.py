import pytest
from src.tts_generation_validation import select_generation_records


def record(speaker, text, split='validation'):
    return {'speaker': speaker, 'text_key': text, 'split': split, 'seconds': 4,
            'path': f'/{split}/{speaker}/{text}.flac'}


def test_generation_selection_is_fixed_balanced_and_validation_only():
    rows = [record(f'p{i}', f'this is validation sentence number {j}') for i in range(10) for j in range(3)]
    rows += [record('train', 'a training sentence with five words', 'train'),
             record('test', 'a withheld test sentence with words', 'test')]
    first = select_generation_records(rows, 8, 42)
    assert first == select_generation_records(list(reversed(rows)), 8, 42)
    assert len({r['speaker'] for r in first}) == 8
    assert all(r['split'] == 'validation' for r in first)


def test_generation_selection_rejects_text_leakage_and_insufficient_speakers():
    text = 'this text accidentally appears in training'
    with pytest.raises(AssertionError, match='leaks'):
        select_generation_records([record('p1', text), record('p2', text, 'train')], 1, 1)
    with pytest.raises(ValueError, match='eligible'):
        select_generation_records([record('p1', text)], 2, 1)
