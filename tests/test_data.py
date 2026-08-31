from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from gpt_from_scratch.data import load_token_tensors, prepare_text_corpus, split_ids


def test_split_ids_preserves_order_and_partition():
    train, validation = split_ids([0, 1, 2, 3, 4, 5], train_fraction=0.5)

    assert train == [0, 1, 2]
    assert validation == [3, 4, 5]


@pytest.mark.parametrize('fraction', [0, 1, -0.1, 1.1])
def test_split_ids_rejects_invalid_fraction(fraction):
    with pytest.raises(ValueError, match='train_fraction'):
        split_ids([0, 1, 2, 3], train_fraction=fraction)


def test_prepare_corpus_writes_long_token_tensors():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / 'corpus.txt'
        source.write_text('abcd ' * 40, encoding='utf-8')
        output = root / 'prepared'

        info = prepare_text_corpus(source, output, train_fraction=0.75)
        train, validation = load_token_tensors(output)

        assert info.total_tokens == train.numel() + validation.numel()
        assert train.dtype == torch.long
        assert validation.dtype == torch.long
        assert (output / 'meta.json').exists()
