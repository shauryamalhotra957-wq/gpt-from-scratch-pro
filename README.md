# GPT From Scratch Pro

[![tests](https://github.com/shauryamalhotra957-wq/gpt-from-scratch-pro/actions/workflows/tests.yml/badge.svg)](https://github.com/shauryamalhotra957-wq/gpt-from-scratch-pro/actions/workflows/tests.yml)

This is a complete, professional GPT-style language-model project inspired by
Andrej Karpathy's reference video, ["Let's build GPT: from scratch, in code,
spelled out"](https://www.youtube.com/watch?v=kCc8FmEb1nY).

It is intentionally small enough to run locally, but it is not a toy script.
It includes a real package layout, a decoder-only Transformer, a character
tokenizer, a byte-level BPE tokenizer, corpus preparation, training,
checkpointing, text generation, tests, configs, and documentation.

## What Makes This Better Than The Tutorial Baseline

- Modern attention path: uses PyTorch scaled-dot-product attention when
  available, with a readable manual causal-attention fallback.
- Resumable training: saves both `best.pt` and `last.pt`, including optimizer
  state, tokenizer metadata, configs, training history, and validation metrics.
- Evaluation workflow: score a checkpoint on prepared train/validation splits
  without starting a new training run.
- Better sampling controls: greedy, temperature, top-k, and top-p nucleus
  sampling with optional deterministic seeds.
- Dataset inspection: verify tokenizer type, vocabulary size, split sizes, and
  decoded previews before burning time on training.
- CI-ready quality gate: pytest and Ruff run locally and in GitHub Actions.

## What Is Inside

- Decoder-only GPT with token embeddings, position embeddings, masked
  multi-head self-attention, MLP blocks, residual streams, pre-LayerNorm,
  dropout, tied output embeddings, AdamW, gradient clipping, and cosine LR decay.
- From-scratch tokenizers:
  - `CharacterTokenizer` for transparent educational runs.
  - `BytePairTokenizer` for reversible byte-level BPE experiments.
- CLI workflow:
  - `sample-data`
  - `prepare`
  - `train`
  - `evaluate`
  - `generate`
- Checkpoints that include model weights, model config, training config,
  tokenizer metadata, training history, and best validation loss.
- Fast tests using `unittest`, with optional pytest compatibility.

## Quickstart

From this project directory:

```powershell
$env:PYTHONPATH = "src"
python -m gpt_from_scratch sample-data --out data/demo.txt
python -m gpt_from_scratch prepare --input data/demo.txt --out-dir runs/demo-data --tokenizer char
python -m gpt_from_scratch inspect-data --data-dir runs/demo-data
python -m gpt_from_scratch train --data-dir runs/demo-data --out-dir runs/demo-run --config configs/smoke.json
python -m gpt_from_scratch evaluate --checkpoint runs/demo-run/best.pt --data-dir runs/demo-data --batch-size 8 --eval-iters 2
python -m gpt_from_scratch generate --checkpoint runs/demo-run/best.pt --prompt "First Engineer:" --max-new-tokens 80 --temperature 0.8 --top-k 20 --top-p 0.95 --seed 7
```

Run the tests:

```powershell
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
```

Install as an editable package:

```powershell
python -m pip install -e .
gpt-from-scratch --help
```

## Train On Tiny Shakespeare

```powershell
python scripts/download_tiny_shakespeare.py
$env:PYTHONPATH = "src"
python -m gpt_from_scratch prepare --input data/tiny_shakespeare.txt --out-dir runs/shakespeare-char --tokenizer char
python -m gpt_from_scratch train --data-dir runs/shakespeare-char --out-dir runs/shakespeare-tiny --config configs/tiny.json
python -m gpt_from_scratch generate --checkpoint runs/shakespeare-tiny/best.pt --prompt "ROMEO:" --max-new-tokens 500 --temperature 0.8 --top-k 40
```

For BPE:

```powershell
$env:PYTHONPATH = "src"
python -m gpt_from_scratch prepare --input data/tiny_shakespeare.txt --out-dir runs/shakespeare-bpe --tokenizer byte-bpe --vocab-size 1024
python -m gpt_from_scratch train --data-dir runs/shakespeare-bpe --out-dir runs/shakespeare-bpe-tiny --config configs/tiny.json
```

## Resume Training

If a run is interrupted, continue from `last.pt`:

```powershell
$env:PYTHONPATH = "src"
python -m gpt_from_scratch train --data-dir runs/shakespeare-char --out-dir runs/shakespeare-tiny --config configs/tiny.json --resume-from runs/shakespeare-tiny/last.pt --max-iters 2000
```

`--max-iters` is the total target iteration, not the number of extra iterations.
For example, if `last.pt` was saved at iteration `1000`, setting
`--max-iters 2000` continues through iteration `2000`.

## Command Reference

```powershell
python -m gpt_from_scratch --help
python -m gpt_from_scratch prepare --help
python -m gpt_from_scratch inspect-data --help
python -m gpt_from_scratch train --help
python -m gpt_from_scratch evaluate --help
python -m gpt_from_scratch generate --help
```

## Project Boundaries

"From scratch" here means the tokenizer, Transformer architecture, training
loop, checkpoint format, and generation loop are implemented directly in this
repo. PyTorch is used for tensor operations, automatic differentiation, and
optimized kernels. That is the right engineering boundary for a serious
software-engineering project: you can inspect the model logic while still
training something real.

This project will not produce ChatGPT-quality output without enormous data,
compute, evaluation, alignment work, and serving infrastructure. It will give
you a clean, inspectable foundation that demonstrates how GPT-style language
models actually work.

## Useful Files

- `src/gpt_from_scratch/model.py`: Transformer model.
- `src/gpt_from_scratch/tokenizer.py`: character and byte-level BPE tokenizers.
- `src/gpt_from_scratch/train.py`: training and evaluation loop.
- `src/gpt_from_scratch/cli.py`: command-line interface.
- `.github/workflows/tests.yml`: CI test workflow.
- `tests/`: correctness and smoke tests.
- `docs/ARCHITECTURE.md`: deeper implementation notes.
- `docs/ENGINEERING.md`: engineering decisions and extension points.
