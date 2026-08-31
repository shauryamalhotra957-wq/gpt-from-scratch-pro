# Security policy

GPT From Scratch Pro is an educational model-training project.

- Do not commit private corpora, API keys, tokens, checkpoints containing restricted data, or generated logs with prompts.
- Treat downloaded datasets and model checkpoints as untrusted inputs; verify their source before loading.
- Review tokenizer and checkpoint changes for unexpected code execution or resource exhaustion.
- Do not represent model outputs as verified facts or use them for high-impact decisions without independent evaluation.
- Keep training runs reproducible by recording dataset and dependency versions without exposing private data.

Report credential exposure, unsafe deserialization, prompt/data leakage, or denial-of-service issues privately to the repository owner.
