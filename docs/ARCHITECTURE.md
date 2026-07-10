# Architecture

The model is a decoder-only Transformer trained with next-token prediction.
For a token sequence `x[0:t]`, the training target is the same sequence shifted
one position to the left. Causal masking prevents each position from attending
to future tokens.

## Data Flow

1. A UTF-8 text corpus is read from disk.
2. A tokenizer is trained on the corpus.
3. The full corpus is encoded into token ids.
4. Token ids are split into train and validation tensors.
5. `TokenBatcher` samples random contiguous windows.
6. `GPT.forward` returns logits and cross-entropy loss.
7. The training loop optimizes with AdamW and saves the best validation
   checkpoint plus the latest resumable checkpoint.

## Model Components

- Token embedding: maps token ids to vectors.
- Position embedding: adds learned position information up to `block_size`.
- Transformer block:
  - Pre-LayerNorm.
  - Masked multi-head self-attention.
  - Residual connection.
  - Pre-LayerNorm.
  - MLP with GELU.
  - Residual connection.
- Final LayerNorm.
- Linear language-modeling head tied to token embeddings.

## Causal Attention

The attention matrix has shape `[batch, heads, time, time]`. A lower-triangular
mask sets logits for future positions to negative infinity before softmax. This
enforces autoregressive generation and is covered by a regression test that
changes a future token and checks that earlier logits are unchanged.

On PyTorch versions that expose `torch.nn.functional.scaled_dot_product_attention`,
the model uses that fused SDPA path with `is_causal=True`. The hand-written
masked attention path remains in the code as a readable fallback and as an
educational reference.

## Tokenization

`CharacterTokenizer` is best for transparent learning runs because every token
is visible as a character.

`BytePairTokenizer` starts with raw byte ids `0..255` and learns repeated byte
pairs. Since every UTF-8 string can be represented as bytes, it is reversible
without an unknown token.

## Checkpoints

Each checkpoint stores:

- Model state dict.
- Optimizer state dict.
- Model config.
- Training config.
- Tokenizer definition.
- Iteration number.
- Best validation loss.
- Training history.
- Metadata such as the prepared data path.

The training loop writes two checkpoint files:

- `best.pt`: the checkpoint with the lowest validation loss so far.
- `last.pt`: the latest checkpoint, intended for `--resume-from`.

That makes generation possible with only the checkpoint file, and makes training
resumable without losing optimizer state.

## Generation

Generation supports:

- Greedy decoding with `--temperature 0`.
- Temperature sampling.
- Top-k sampling.
- Top-p nucleus sampling.
- Reproducible sampling with `--seed`.

The sampler applies filters directly to the final-position logits, then samples
one token at a time autoregressively.

## Evaluation

The `evaluate` command reloads a checkpoint and estimates train/validation loss
on random batches from prepared tensors. It also reports perplexity, which is
`exp(loss)` when the loss is numerically small enough to avoid overflow.
