# Contributing

Thanks for improving GPT From Scratch Pro.

## Development setup

1. Fork or clone the repository.
2. Create a focused branch from `main`.
3. Create and activate a supported Python virtual environment.
4. Install the project with `python -m pip install -e .`.

## Quality checks

Run:

```bash
python -m pytest
python -m ruff check .
```

Add deterministic tests for model, tokenizer, data-pipeline, and checkpoint behavior. Keep fixtures compact and avoid committing downloaded corpora or model weights.

## Pull requests

Describe the problem, implementation, compatibility impact, and validation. Include benchmark results when changing performance-sensitive code.
