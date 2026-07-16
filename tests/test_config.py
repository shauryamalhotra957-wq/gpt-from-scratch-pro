from __future__ import annotations

import unittest

from gpt_from_scratch.config import TrainConfig


class TrainConfigTests(unittest.TestCase):
    def test_rejects_minimum_lr_above_peak_lr(self) -> None:
        with self.assertRaisesRegex(ValueError, "min_lr must not exceed learning_rate"):
            TrainConfig(learning_rate=1e-4, min_lr=2e-4)

    def test_rejects_negative_warmup(self) -> None:
        with self.assertRaisesRegex(ValueError, "warmup_iters must be non-negative"):
            TrainConfig(warmup_iters=-1)

    def test_rejects_decay_before_warmup_finishes(self) -> None:
        with self.assertRaisesRegex(ValueError, "lr_decay_iters"):
            TrainConfig(warmup_iters=20, lr_decay_iters=10)


if __name__ == "__main__":
    unittest.main()
