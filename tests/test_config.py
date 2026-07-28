from __future__ import annotations

import unittest

from gpt_from_scratch.config import TrainConfig
from gpt_from_scratch.train import get_lr


class TrainConfigTests(unittest.TestCase):
    def test_rejects_inverted_learning_rate_floor(self) -> None:
        with self.assertRaisesRegex(ValueError, "min_lr must not exceed learning_rate"):
            TrainConfig(learning_rate=1e-4, min_lr=2e-4)

    def test_rejects_negative_warmup(self) -> None:
        with self.assertRaisesRegex(ValueError, "warmup_iters must be non-negative"):
            TrainConfig(warmup_iters=-1)

    def test_rejects_decay_that_ends_before_warmup(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "lr_decay_iters must be greater than or equal to warmup_iters",
        ):
            TrainConfig(warmup_iters=20, lr_decay_iters=10)

    def test_learning_rate_schedule_hits_documented_boundaries(self) -> None:
        config = TrainConfig(
            learning_rate=1e-3,
            min_lr=1e-4,
            warmup_iters=2,
            lr_decay_iters=6,
        )

        self.assertAlmostEqual(get_lr(0, config), 5e-4)
        self.assertAlmostEqual(get_lr(1, config), 1e-3)
        self.assertAlmostEqual(get_lr(2, config), 1e-3)
        self.assertAlmostEqual(get_lr(6, config), 1e-4)
        self.assertAlmostEqual(get_lr(7, config), 1e-4)

    def test_zero_length_warmup_is_supported(self) -> None:
        config = TrainConfig(
            learning_rate=1e-3,
            min_lr=1e-4,
            warmup_iters=0,
            lr_decay_iters=4,
        )

        self.assertAlmostEqual(get_lr(0, config), 1e-3)


if __name__ == "__main__":
    unittest.main()
