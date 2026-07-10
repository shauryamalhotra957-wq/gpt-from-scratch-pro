# Engineering Notes

## Design Choices

- The public API is small: `GPTConfig`, `TrainConfig`, `GPT`, tokenizers, and
  the CLI.
- Config JSON is validated strictly so misspelled fields fail early.
- Tests use `unittest` so the project works even when pytest is not installed.
- Checkpoints include tokenizer metadata so generation is reproducible.
- `last.pt` includes optimizer state so interrupted runs can resume properly.
- The CLI exposes inspect, train, resume, evaluate, and generate workflows.
- The tokenizer and model are decoupled through a small protocol.
- The model uses explicit masked attention instead of hiding the central idea
  behind a high-level Transformer wrapper, while also using PyTorch's fused
  scaled-dot-product attention when available.

## Extension Points

- Add dataset streaming for corpora too large to fit in memory.
- Add distributed data-parallel training.
- Add a GPT-2 compatible tokenizer adapter.
- Add evaluation metrics such as perplexity reports by split.
- Add richer CUDA training profiles for specific GPU classes.
- Add a small web UI for prompt sampling from local checkpoints.

## Quality Gates

Before treating a change as complete:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests
python -m gpt_from_scratch prepare --input data/tiny_corpus.txt --out-dir runs/manual-data --tokenizer char
python -m gpt_from_scratch inspect-data --data-dir runs/manual-data
python -m gpt_from_scratch train --data-dir runs/manual-data --out-dir runs/manual-run --config configs/smoke.json
python -m gpt_from_scratch evaluate --checkpoint runs/manual-run/best.pt --data-dir runs/manual-data --batch-size 8 --eval-iters 2
python -m gpt_from_scratch generate --checkpoint runs/manual-run/best.pt --prompt "First Engineer:" --max-new-tokens 40 --top-p 0.95 --seed 7
```

Or run the local smoke script:

```powershell
.\scripts\smoke_test.ps1
```

For hosted checks, `.github/workflows/tests.yml` runs the same unittest suite on
Python 3.11.
