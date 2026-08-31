from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import inspect
from typing import Literal

import torch

from gpt_from_scratch.config import save_json
from gpt_from_scratch.tokenizer import (
    BytePairTokenizer,
    CharacterTokenizer,
    Tokenizer,
)


@dataclass
class PreparedDataInfo:
    input_path: str
    tokenizer_path: str
    train_path: str
    val_path: str
    tokenizer_type: str
    vocab_size: int
    total_tokens: int
    train_tokens: int
    val_tokens: int
    train_fraction: float


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def split_ids(ids: list[int], train_fraction: float = 0.9) -> tuple[list[int], list[int]]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if len(ids) < 4:
        raise ValueError("Need at least 4 tokens to create train and validation splits")
    split_idx = int(len(ids) * train_fraction)
    split_idx = min(max(split_idx, 1), len(ids) - 1)
    return ids[:split_idx], ids[split_idx:]


def prepare_text_corpus(
    input_path: str | Path,
    out_dir: str | Path,
    tokenizer_type: Literal["char", "byte-bpe"] = "char",
    vocab_size: int = 512,
    train_fraction: float = 0.9,
    min_frequency: int = 2,
) -> PreparedDataInfo:
    """Train a tokenizer, encode text, and save train/validation token tensors."""

    input_path = Path(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    text = read_text(input_path)
    if tokenizer_type == "char":
        tokenizer: Tokenizer = CharacterTokenizer.train(text)
    elif tokenizer_type == "byte-bpe":
        tokenizer = BytePairTokenizer.train(
            text,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
        )
    else:
        raise ValueError(f"Unsupported tokenizer_type: {tokenizer_type!r}")

    ids = tokenizer.encode(text)
    train_ids, val_ids = split_ids(ids, train_fraction=train_fraction)

    train_path = out_dir / "train.pt"
    val_path = out_dir / "val.pt"
    tokenizer_path = out_dir / "tokenizer.json"

    torch.save(torch.tensor(train_ids, dtype=torch.long), train_path)
    torch.save(torch.tensor(val_ids, dtype=torch.long), val_path)
    tokenizer.save(tokenizer_path)

    info = PreparedDataInfo(
        input_path=str(input_path),
        tokenizer_path=str(tokenizer_path),
        train_path=str(train_path),
        val_path=str(val_path),
        tokenizer_type=tokenizer_type,
        vocab_size=tokenizer.vocab_size,
        total_tokens=len(ids),
        train_tokens=len(train_ids),
        val_tokens=len(val_ids),
        train_fraction=train_fraction,
    )
    save_json(out_dir / "meta.json", asdict(info))
    return info


def load_token_tensors(data_dir: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    data_dir = Path(data_dir)
    train_path = data_dir / "train.pt"
    val_path = data_dir / "val.pt"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Expected {train_path.name} and {val_path.name} inside {data_dir}"
        )
    load_kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = True
    train_data = torch.load(train_path, **load_kwargs)
    val_data = torch.load(val_path, **load_kwargs)
    if train_data.dtype != torch.long or val_data.dtype != torch.long:
        raise TypeError("Prepared token tensors must have dtype torch.long")
    return train_data, val_data


class TokenBatcher:
    """Random contiguous next-token batches for autoregressive training."""

    def __init__(
        self,
        train_data: torch.Tensor,
        val_data: torch.Tensor,
        block_size: int,
        batch_size: int,
        device: torch.device,
    ) -> None:
        self.train_data = train_data
        self.val_data = val_data
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self._validate("train", train_data)
        self._validate("val", val_data)

    def _validate(self, split: str, data: torch.Tensor) -> None:
        if data.ndim != 1:
            raise ValueError(f"{split} data must be a 1D token tensor")
        if len(data) <= self.block_size:
            raise ValueError(
                f"{split} data has {len(data)} tokens, but block_size is "
                f"{self.block_size}; provide more data or lower block_size"
            )

    def get_batch(self, split: Literal["train", "val"]) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.train_data if split == "train" else self.val_data
        upper = len(data) - self.block_size
        starts = torch.randint(0, upper, (self.batch_size,))
        x = torch.stack([data[start : start + self.block_size] for start in starts])
        y = torch.stack([data[start + 1 : start + self.block_size + 1] for start in starts])
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

