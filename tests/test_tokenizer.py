from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from gpt_from_scratch.tokenizer import (
    BytePairTokenizer,
    CharacterTokenizer,
    tokenizer_from_file,
)


class TokenizerTests(unittest.TestCase):
    def test_character_tokenizer_roundtrip_and_persistence(self) -> None:
        text = "hello small transformer\n"
        tokenizer = CharacterTokenizer.train(text)
        ids = tokenizer.encode(text)
        self.assertEqual(tokenizer.decode(ids), text)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tokenizer.json"
            tokenizer.save(path)
            loaded = tokenizer_from_file(path)
            self.assertEqual(loaded.decode(loaded.encode(text)), text)

    def test_character_tokenizer_rejects_unknown_character(self) -> None:
        tokenizer = CharacterTokenizer.train("abc")
        with self.assertRaises(ValueError):
            tokenizer.encode("abcd")

    def test_byte_pair_tokenizer_roundtrip_and_persistence(self) -> None:
        text = "the theater theory theme then there\n" * 4
        tokenizer = BytePairTokenizer.train(text, vocab_size=280, min_frequency=2)
        ids = tokenizer.encode(text)
        self.assertEqual(tokenizer.decode(ids), text)
        self.assertGreaterEqual(tokenizer.vocab_size, 256)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tokenizer.json"
            tokenizer.save(path)
            loaded = tokenizer_from_file(path)
            self.assertEqual(loaded.decode(loaded.encode(text)), text)


if __name__ == "__main__":
    unittest.main()

