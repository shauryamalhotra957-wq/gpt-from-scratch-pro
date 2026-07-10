from __future__ import annotations

from contextlib import nullcontext
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, ContextManager

import torch

from gpt_from_scratch.checkpoint import load_checkpoint, load_model_from_checkpoint, save_checkpoint
from gpt_from_scratch.config import GPTConfig, TrainConfig, save_json
from gpt_from_scratch.data import TokenBatcher, load_token_tensors
from gpt_from_scratch.model import GPT
from gpt_from_scratch.tokenizer import tokenizer_from_file
from gpt_from_scratch.utils import device_type, resolve_dtype, select_device, set_seed


def get_lr(iter_num: int, config: TrainConfig) -> float:
    if iter_num < config.warmup_iters:
        return config.learning_rate * (iter_num + 1) / max(1, config.warmup_iters)
    if iter_num > config.lr_decay_iters:
        return config.min_lr
    decay_ratio = (iter_num - config.warmup_iters) / max(
        1,
        config.lr_decay_iters - config.warmup_iters,
    )
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


def safe_perplexity(loss: float) -> float:
    return float("inf") if loss > 80 else math.exp(loss)


def make_autocast_context(
    device: torch.device,
    dtype: torch.dtype,
) -> Callable[[], ContextManager[Any]]:
    dev_type = device_type(device)
    use_amp = dev_type == "cuda" and dtype in {torch.float16, torch.bfloat16}

    def autocast_context() -> ContextManager[Any]:
        if use_amp:
            return torch.amp.autocast(device_type=dev_type, dtype=dtype)
        return nullcontext()

    return autocast_context


def _move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _ensure_resume_compatible(
    checkpoint: dict[str, Any],
    model_config: GPTConfig,
) -> None:
    checkpoint_model_config = checkpoint.get("model_config")
    if checkpoint_model_config != model_config.to_dict():
        raise ValueError(
            "Resume checkpoint architecture does not match the requested model config. "
            "Use the same config file or start a fresh run."
        )


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    batcher: TokenBatcher,
    eval_iters: int,
    autocast_context: Callable[[], ContextManager[Any]],
) -> dict[str, float]:
    out: dict[str, float] = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for idx in range(eval_iters):
            x, y = batcher.get_batch(split)  # type: ignore[arg-type]
            with autocast_context():
                _, loss = model(x, y)
            if loss is None:
                raise RuntimeError("Expected a loss during evaluation")
            losses[idx] = loss.item()
        out[split] = float(losses.mean())
    model.train()
    return out


def train_model(
    data_dir: str | Path,
    out_dir: str | Path,
    model_config: GPTConfig,
    train_config: TrainConfig,
    resume_from: str | Path | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    """Run a complete next-token training job and return the final summary."""

    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(train_config.seed)

    tokenizer_path = data_dir / "tokenizer.json"
    tokenizer = tokenizer_from_file(tokenizer_path)
    if tokenizer.vocab_size != model_config.vocab_size:
        raise ValueError(
            f"Tokenizer vocab size is {tokenizer.vocab_size}, but model config "
            f"uses {model_config.vocab_size}"
        )

    train_data, val_data = load_token_tensors(data_dir)
    device = select_device(train_config.device)
    dtype = resolve_dtype(device, train_config.dtype)
    dev_type = device_type(device)
    autocast_context = make_autocast_context(device, dtype)

    batcher = TokenBatcher(
        train_data=train_data,
        val_data=val_data,
        block_size=model_config.block_size,
        batch_size=train_config.batch_size,
        device=device,
    )
    raw_model = GPT(model_config).to(device)
    model = raw_model
    optimizer = raw_model.configure_optimizers(
        weight_decay=train_config.weight_decay,
        learning_rate=train_config.learning_rate,
        betas=(train_config.beta1, train_config.beta2),
        device_type=dev_type,
    )

    best_val_loss = float("inf")
    history: list[dict[str, float | int]] = []
    start_iter = 0
    if resume_from is not None:
        checkpoint = load_checkpoint(resume_from, map_location=device)
        _ensure_resume_compatible(checkpoint, model_config)
        raw_model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        _move_optimizer_state_to_device(optimizer, device)
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
        history = list(checkpoint.get("history", []))
        start_iter = int(checkpoint.get("iter_num", -1)) + 1
        set_seed(train_config.seed + start_iter)
        if progress:
            print(f"Resuming from {resume_from} at iteration {start_iter}")

    if train_config.compile and hasattr(torch, "compile"):
        model = torch.compile(raw_model)  # type: ignore[assignment]
    scaler_enabled = dev_type == "cuda" and dtype == torch.float16
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    except TypeError:
        scaler = torch.amp.GradScaler(enabled=scaler_enabled)

    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)
    tokens_per_iter = (
        train_config.gradient_accumulation_steps
        * train_config.batch_size
        * model_config.block_size
    )

    for iter_num in range(start_iter, train_config.max_iters + 1):
        lr = get_lr(iter_num, train_config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        should_eval = (
            iter_num % train_config.eval_interval == 0
            or iter_num == train_config.max_iters
        )
        if should_eval:
            losses = estimate_loss(
                model=model,
                batcher=batcher,
                eval_iters=train_config.eval_iters,
                autocast_context=autocast_context,
            )
            record = {
                "iter": iter_num,
                "train_loss": losses["train"],
                "val_loss": losses["val"],
                "val_perplexity": safe_perplexity(losses["val"]),
                "lr": lr,
                "elapsed_seconds": time.time() - start_time,
            }
            history.append(record)
            is_best = losses["val"] < best_val_loss
            if is_best:
                best_val_loss = losses["val"]
            save_checkpoint(
                out_dir / "last.pt",
                model=raw_model,
                optimizer=optimizer,
                model_config=model_config,
                train_config=train_config,
                tokenizer=tokenizer,
                iter_num=iter_num,
                best_val_loss=best_val_loss,
                history=history,
                meta={"data_dir": str(data_dir), "checkpoint_kind": "last"},
            )
            if is_best or train_config.always_save_checkpoint:
                save_checkpoint(
                    out_dir / "best.pt",
                    model=raw_model,
                    optimizer=optimizer,
                    model_config=model_config,
                    train_config=train_config,
                    tokenizer=tokenizer,
                    iter_num=iter_num,
                    best_val_loss=best_val_loss,
                    history=history,
                    meta={"data_dir": str(data_dir), "checkpoint_kind": "best"},
                )
            save_json(out_dir / "history.json", {"history": history})
            if progress:
                print(
                    "iter "
                    f"{iter_num:>6} | train {losses['train']:.4f} | "
                    f"val {losses['val']:.4f} | ppl {record['val_perplexity']:.2f} | "
                    f"lr {lr:.2e}"
                )

        if iter_num == train_config.max_iters:
            break

        for _ in range(train_config.gradient_accumulation_steps):
            x, y = batcher.get_batch("train")
            with autocast_context():
                _, loss = model(x, y)
                if loss is None:
                    raise RuntimeError("Expected a loss during training")
                loss = loss / train_config.gradient_accumulation_steps
            scaler.scale(loss).backward()

        if train_config.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), train_config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    elapsed = time.time() - start_time
    summary = {
        "out_dir": str(out_dir),
        "best_checkpoint": str(out_dir / "best.pt"),
        "last_checkpoint": str(out_dir / "last.pt"),
        "best_val_loss": best_val_loss,
        "best_val_perplexity": safe_perplexity(best_val_loss),
        "elapsed_seconds": elapsed,
        "model_parameters": raw_model.num_parameters(),
        "tokens_per_iter": tokens_per_iter,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "resumed_from": str(resume_from) if resume_from is not None else None,
        "history": history,
    }
    save_json(out_dir / "summary.json", summary)
    return summary


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    data_dir: str | Path,
    batch_size: int = 32,
    eval_iters: int = 20,
    device: str = "auto",
    dtype: str = "auto",
) -> dict[str, Any]:
    selected_device = select_device(device)
    selected_dtype = resolve_dtype(selected_device, dtype)
    model, _, checkpoint = load_model_from_checkpoint(
        checkpoint_path,
        map_location=selected_device,
    )
    model_config = GPTConfig(**checkpoint["model_config"])
    train_data, val_data = load_token_tensors(data_dir)
    batcher = TokenBatcher(
        train_data=train_data,
        val_data=val_data,
        block_size=model_config.block_size,
        batch_size=batch_size,
        device=selected_device,
    )
    losses = estimate_loss(
        model=model,
        batcher=batcher,
        eval_iters=eval_iters,
        autocast_context=make_autocast_context(selected_device, selected_dtype),
    )
    return {
        "checkpoint": str(checkpoint_path),
        "data_dir": str(data_dir),
        "train_loss": losses["train"],
        "val_loss": losses["val"],
        "train_perplexity": safe_perplexity(losses["train"]),
        "val_perplexity": safe_perplexity(losses["val"]),
        "eval_iters": eval_iters,
        "batch_size": batch_size,
        "device": str(selected_device),
        "dtype": str(selected_dtype).replace("torch.", ""),
    }


def load_config_bundle(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    model_payload = dict(payload.get("model", {}))
    train_payload = dict(payload.get("train", {}))
    unknown = sorted(set(payload) - {"model", "train"})
    if unknown:
        raise ValueError(f"Unknown top-level config section(s): {', '.join(unknown)}")
    return model_payload, train_payload
