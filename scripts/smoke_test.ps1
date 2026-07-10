$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$env:PYTHONPATH = "src"

python -m unittest discover -s tests
python -m gpt_from_scratch prepare --input data/tiny_corpus.txt --out-dir runs/smoke-data --tokenizer char
python -m gpt_from_scratch inspect-data --data-dir runs/smoke-data --preview-tokens 80
python -m gpt_from_scratch train --data-dir runs/smoke-data --out-dir runs/smoke-run --config configs/smoke.json
python -m gpt_from_scratch evaluate --checkpoint runs/smoke-run/best.pt --data-dir runs/smoke-data --batch-size 8 --eval-iters 2
python -m gpt_from_scratch generate --checkpoint runs/smoke-run/best.pt --prompt "First Engineer:" --max-new-tokens 80 --temperature 0.8 --top-k 20 --top-p 0.95 --seed 7

