# CLI Experience Standard

The command line is this project's interface. Commands should be predictable in a local terminal, readable in logs, and safe to automate in CI.

## Command journey

1. `sample-data` or `prepare` establishes an inspectable corpus.
2. `inspect-data` confirms tokenizer, split, token count, and preview.
3. `train` reports iteration, loss, timing, and checkpoint decisions.
4. `evaluate` separates train and validation evidence.
5. `generate` prints generated text without decorative noise.

## Output rules

- Start long operations by naming the resolved device, dtype, data path, and output directory.
- Use stable labels that are easy for humans to scan and scripts to parse.
- Report progress at meaningful intervals; avoid decorative spinners in captured logs.
- A saved checkpoint message must include its path and whether it is `best` or `last`.
- Errors go to stderr, name the failed input, and include one actionable correction.
- Successful commands exit `0`; invalid configuration or missing assets exit non-zero.

## Accessibility and automation

Never rely on terminal color to communicate state. If color is added later, honor `NO_COLOR` and keep the text labels intact. Progress output must remain understandable in monochrome logs and when copied into issue reports.

## Recommended state vocabulary

`preparing`, `ready`, `training`, `evaluating`, `checkpoint saved`, `complete`, `error`

These labels should remain consistent across CLI commands, documentation, and future experiment dashboards.
