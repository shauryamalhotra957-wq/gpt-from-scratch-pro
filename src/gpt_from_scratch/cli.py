from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

from gpt_from_scratch.checkpoint import load_model_from_checkpoint
from gpt_from_scratch.config import GPTConfig, TrainConfig, dataclass_from_dict, load_json
from gpt_from_scratch.data import load_token_tensors, prepare_text_corpus
from gpt_from_scratch.tokenizer import tokenizer_from_file
from gpt_from_scratch.train import evaluate_checkpoint, load_config_bundle, train_model
from gpt_from_scratch.utils import select_device, set_seed


def _add_model_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--n-layer", type=int)
    parser.add_argument("--n-head", type=int)
    parser.add_argument("--n-embd", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--bias", action=argparse.BooleanOptionalAction)


def _add_train_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-iters", type=int)
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--eval-iters", type=int)
    parser.add_argument("--log-interval", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--min-lr", type=float)
    parser.add_argument("--warmup-iters", type=int)
    parser.add_argument("--lr-decay-iters", type=int)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--grad-clip", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--dtype", type=str)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction)
    parser.add_argument("--always-save-checkpoint", action=argparse.BooleanOptionalAction)


def _collect_present(args: argparse.Namespace, names: list[str]) -> dict[str, object]:
    values: dict[str, object] = {}
    for name in names:
        value = getattr(args, name.replace("-", "_"))
        if value is not None:
            values[name.replace("-", "_")] = value
    return values


def command_prepare(args: argparse.Namespace) -> int:
    info = prepare_text_corpus(
        input_path=args.input,
        out_dir=args.out_dir,
        tokenizer_type=args.tokenizer,
        vocab_size=args.vocab_size,
        train_fraction=args.train_fraction,
        min_frequency=args.min_frequency,
    )
    print(f"Prepared {info.total_tokens} tokens")
    print(f"Tokenizer: {info.tokenizer_path}")
    print(f"Train: {info.train_tokens} tokens | Val: {info.val_tokens} tokens")
    print(f"Vocab size: {info.vocab_size}")
    return 0


def command_train(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    meta = load_json(data_dir / "meta.json")

    model_payload = {
        "vocab_size": int(meta["vocab_size"]),
        "block_size": 128,
        "n_layer": 4,
        "n_head": 4,
        "n_embd": 128,
        "dropout": 0.1,
        "bias": True,
    }
    train_payload = {}
    if args.config is not None:
        config_model, config_train = load_config_bundle(args.config)
        model_payload.update(config_model)
        train_payload.update(config_train)

    model_payload.update(
        _collect_present(
            args,
            ["block-size", "n-layer", "n-head", "n-embd", "dropout", "bias"],
        )
    )
    train_payload.update(
        _collect_present(
            args,
            [
                "batch-size",
                "max-iters",
                "eval-interval",
                "eval-iters",
                "log-interval",
                "gradient-accumulation-steps",
                "learning-rate",
                "min-lr",
                "warmup-iters",
                "lr-decay-iters",
                "weight-decay",
                "grad-clip",
                "seed",
                "device",
                "dtype",
                "compile",
                "always-save-checkpoint",
            ],
        )
    )

    model_config = dataclass_from_dict(GPTConfig, model_payload)
    train_config = dataclass_from_dict(TrainConfig, train_payload)
    summary = train_model(
        data_dir=data_dir,
        out_dir=args.out_dir,
        model_config=model_config,
        train_config=train_config,
        resume_from=args.resume_from,
        progress=not args.quiet,
    )
    print(f"Best validation loss: {summary['best_val_loss']:.4f}")
    print(f"Best validation perplexity: {summary['best_val_perplexity']:.2f}")
    print(f"Checkpoint: {summary['best_checkpoint']}")
    print(f"Last checkpoint: {summary['last_checkpoint']}")
    print(f"Parameters: {summary['model_parameters']:,}")
    return 0


def command_generate(args: argparse.Namespace) -> int:
    if args.seed is not None:
        set_seed(args.seed)
    device = select_device(args.device)
    model, tokenizer, _ = load_model_from_checkpoint(args.checkpoint, map_location=device)
    model.to(device)
    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        raise ValueError("Prompt encoded to zero tokens")
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    output = model.generate(
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    text = tokenizer.decode(output[0].tolist())
    print(text)
    if args.out is not None:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


def command_inspect_data(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    meta = load_json(data_dir / "meta.json")
    tokenizer = tokenizer_from_file(data_dir / "tokenizer.json")
    train_data, val_data = load_token_tensors(data_dir)
    print(f"Input: {meta['input_path']}")
    print(f"Tokenizer: {meta['tokenizer_type']} | vocab size: {tokenizer.vocab_size}")
    print(f"Tokens: total {meta['total_tokens']} | train {len(train_data)} | val {len(val_data)}")
    print(f"Train fraction: {meta['train_fraction']}")
    if args.preview_tokens > 0:
        preview_ids = train_data[: args.preview_tokens].tolist()
        preview = tokenizer.decode(preview_ids)
        print("Preview:")
        print(preview)
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    metrics = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        eval_iters=args.eval_iters,
        device=args.device,
        dtype=args.dtype,
    )
    print(f"Checkpoint: {metrics['checkpoint']}")
    print(f"Train loss: {metrics['train_loss']:.4f} | ppl {metrics['train_perplexity']:.2f}")
    print(f"Val loss: {metrics['val_loss']:.4f} | ppl {metrics['val_perplexity']:.2f}")
    print(f"Device: {metrics['device']} | dtype: {metrics['dtype']}")
    return 0


def command_sample_data(args: argparse.Namespace) -> int:
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(SAMPLE_CORPUS, encoding="utf-8")
    print(f"Wrote sample corpus to {target}")
    return 0


SAMPLE_CORPUS = """First Engineer:
We build the small machine carefully, wire by wire.

Second Engineer:
Then we teach it to predict the next mark, and the marks become music.

First Engineer:
Does it understand?

Second Engineer:
Not as we do. But it learns the shape of our language, and that is a beginning.

Narrator:
The workshop grows quiet. The model listens to every symbol before it speaks.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpt-from-scratch",
        description="Train and sample a GPT-style Transformer from scratch.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Tokenize a text corpus")
    prepare.add_argument("--input", required=True, help="Path to UTF-8 text")
    prepare.add_argument("--out-dir", required=True, help="Directory for prepared data")
    prepare.add_argument(
        "--tokenizer",
        choices=["char", "byte-bpe"],
        default="char",
        help="Tokenizer type",
    )
    prepare.add_argument("--vocab-size", type=int, default=512)
    prepare.add_argument("--min-frequency", type=int, default=2)
    prepare.add_argument("--train-fraction", type=float, default=0.9)
    prepare.set_defaults(func=command_prepare)

    inspect_data = subparsers.add_parser("inspect-data", help="Inspect a prepared dataset")
    inspect_data.add_argument("--data-dir", required=True, help="Directory created by prepare")
    inspect_data.add_argument("--preview-tokens", type=int, default=160)
    inspect_data.set_defaults(func=command_inspect_data)

    train = subparsers.add_parser("train", help="Train a GPT model")
    train.add_argument("--data-dir", required=True, help="Directory created by prepare")
    train.add_argument("--out-dir", required=True, help="Directory for checkpoints")
    train.add_argument("--config", help="JSON file with model/train sections")
    train.add_argument("--resume-from", help="Checkpoint to resume from")
    train.add_argument("--quiet", action="store_true", help="Disable progress logs")
    _add_model_overrides(train)
    _add_train_overrides(train)
    train.set_defaults(func=command_train)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a checkpoint on prepared data")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--data-dir", required=True, help="Directory created by prepare")
    evaluate.add_argument("--batch-size", type=int, default=32)
    evaluate.add_argument("--eval-iters", type=int, default=20)
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--dtype", default="auto")
    evaluate.set_defaults(func=command_evaluate)

    generate = subparsers.add_parser("generate", help="Generate text from a checkpoint")
    generate.add_argument("--checkpoint", required=True)
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--max-new-tokens", type=int, default=100)
    generate.add_argument("--temperature", type=float, default=0.8)
    generate.add_argument("--top-k", type=int)
    generate.add_argument("--top-p", type=float)
    generate.add_argument("--device", default="auto")
    generate.add_argument("--seed", type=int)
    generate.add_argument("--out")
    generate.set_defaults(func=command_generate)

    sample_data = subparsers.add_parser("sample-data", help="Write a tiny demo corpus")
    sample_data.add_argument("--out", required=True)
    sample_data.set_defaults(func=command_sample_data)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        raise SystemExit(args.func(args))
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
