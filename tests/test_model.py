from __future__ import annotations

import unittest

import torch

from gpt_from_scratch.config import GPTConfig
from gpt_from_scratch.model import GPT, sample_next_token


class ModelTests(unittest.TestCase):
    def test_forward_shape_and_loss_are_valid(self) -> None:
        config = GPTConfig(
            vocab_size=17,
            block_size=8,
            n_layer=2,
            n_head=2,
            n_embd=16,
            dropout=0.0,
        )
        model = GPT(config)
        idx = torch.randint(0, config.vocab_size, (4, config.block_size))
        logits, loss = model(idx, idx)
        self.assertEqual(logits.shape, (4, config.block_size, config.vocab_size))
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))

    def test_causal_mask_prevents_future_leakage(self) -> None:
        torch.manual_seed(7)
        config = GPTConfig(
            vocab_size=23,
            block_size=6,
            n_layer=2,
            n_head=2,
            n_embd=16,
            dropout=0.0,
        )
        model = GPT(config)
        model.eval()
        idx = torch.randint(0, config.vocab_size, (2, config.block_size))
        changed_future = idx.clone()
        changed_future[:, -1] = (changed_future[:, -1] + 1) % config.vocab_size

        with torch.no_grad():
            logits_a, _ = model(idx)
            logits_b, _ = model(changed_future)

        self.assertTrue(torch.allclose(logits_a[:, :-1], logits_b[:, :-1], atol=1e-5))

    def test_generation_extends_sequence(self) -> None:
        config = GPTConfig(
            vocab_size=11,
            block_size=4,
            n_layer=1,
            n_head=1,
            n_embd=8,
            dropout=0.0,
        )
        model = GPT(config)
        idx = torch.tensor([[1, 2, 3]], dtype=torch.long)
        out = model.generate(idx, max_new_tokens=5, temperature=0, top_p=1.0)
        self.assertEqual(out.shape, (1, 8))

    def test_sampling_validates_filters(self) -> None:
        logits = torch.randn(2, 5)
        sampled = sample_next_token(logits, temperature=1.0, top_k=3, top_p=0.8)
        self.assertEqual(sampled.shape, (2, 1))
        with self.assertRaises(ValueError):
            sample_next_token(logits, top_p=0)
        with self.assertRaises(ValueError):
            sample_next_token(logits, top_k=0)


if __name__ == "__main__":
    unittest.main()
