from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from typing import Any, TypeVar


T = TypeVar("T")


@dataclass
class GPTConfig:
    """Architecture settings for a decoder-only GPT language model."""

    vocab_size: int
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    bias: bool = True

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.n_layer <= 0:
            raise ValueError("n_layer must be positive")
        if self.n_head <= 0:
            raise ValueError("n_head must be positive")
        if self.n_embd <= 0:
            raise ValueError("n_embd must be positive")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainConfig:
    """Training settings for next-token language modeling."""

    batch_size: int = 32
    max_iters: int = 1000
    eval_interval: int = 100
    eval_iters: int = 20
    log_interval: int = 10
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 100
    lr_decay_iters: int = 1000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    seed: int = 1337
    device: str = "auto"
    dtype: str = "auto"
    compile: bool = False
    always_save_checkpoint: bool = False

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_iters < 0:
            raise ValueError("max_iters must be non-negative")
        if self.eval_interval <= 0:
            raise ValueError("eval_interval must be positive")
        if self.eval_iters <= 0:
            raise ValueError("eval_iters must be positive")
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.min_lr < 0:
            raise ValueError("min_lr must be non-negative")
        if self.min_lr > self.learning_rate:
            raise ValueError("min_lr must not exceed learning_rate")
        if self.warmup_iters < 0:
            raise ValueError("warmup_iters must be non-negative")
        if self.lr_decay_iters < self.warmup_iters:
            raise ValueError("lr_decay_iters must be greater than or equal to warmup_iters")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("AdamW betas must be in [0, 1)")
        if self.grad_clip < 0:
            raise ValueError("grad_clip must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def dataclass_from_dict(cls: type[T], payload: dict[str, Any]) -> T:
    """Create a dataclass instance and reject misspelled configuration keys."""

    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        joined = ", ".join(unknown)
        raise ValueError(f"Unknown {cls.__name__} field(s): {joined}")
    return cls(**payload)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

