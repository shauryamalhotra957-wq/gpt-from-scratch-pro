from __future__ import annotations

from pathlib import Path
from typing import Any
import inspect

import torch

from gpt_from_scratch.config import GPTConfig, TrainConfig
from gpt_from_scratch.model import GPT
from gpt_from_scratch.tokenizer import Tokenizer, tokenizer_from_dict


def save_checkpoint(
    path: str | Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    model_config: GPTConfig,
    train_config: TrainConfig,
    tokenizer: Tokenizer,
    iter_num: int,
    best_val_loss: float,
    history: list[dict[str, float | int]],
    meta: dict[str, Any] | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config": model_config.to_dict(),
        "train_config": train_config.to_dict(),
        "tokenizer": tokenizer.to_dict(),
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
        "history": history,
        "meta": meta or {},
    }
    tmp_path = target.with_name(f"{target.name}.tmp")
    torch.save(checkpoint, tmp_path)
    tmp_path.replace(target)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    kwargs: dict[str, Any] = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(Path(path), **kwargs)


def load_model_from_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[GPT, Tokenizer, dict[str, Any]]:
    checkpoint = load_checkpoint(path, map_location=map_location)
    model = GPT(GPTConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(map_location)
    model.eval()
    tokenizer = tokenizer_from_dict(checkpoint["tokenizer"])
    return model, tokenizer, checkpoint
