from __future__ import annotations

import pytest

from gpt_from_scratch.config import TrainConfig


def test_train_config_accepts_matching_schedule():
    config = TrainConfig(
        learning_rate=0.001,
        min_lr=0.0001,
        warmup_iters=10,
        lr_decay_iters=20,
    )

    assert config.min_lr < config.learning_rate


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"learning_rate": 0.001, "min_lr": 0.002}, "min_lr"),
        ({"warmup_iters": 11, "lr_decay_iters": 10}, "warmup_iters"),
        ({"lr_decay_iters": 0}, "lr_decay_iters"),
    ],
)
def test_train_config_rejects_contradictory_schedule(kwargs, message):
    with pytest.raises(ValueError, match=message):
        TrainConfig(**kwargs)
