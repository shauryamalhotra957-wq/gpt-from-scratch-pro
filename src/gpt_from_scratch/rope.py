"""
Rotary Position Embedding (RoPE) implementation for Transformer architectures.
Based on Su et al. (2021) 'RoFormer: Enhanced Transformer with Rotary Position Embedding'.
"""
import torch
import torch.nn as nn

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Compute inverse frequencies: theta_i = 10000 ^ (-2(i-1)/dim)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos and sin embeddings
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        # Shape: [max_seq_len, dim / 2] -> duplicate for full dim: [max_seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        half_dim = x.shape[-1] // 2
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Applies rotary embeddings to input tensor x of shape [batch, num_heads, seq_len, head_dim]
        or [batch, seq_len, num_heads, head_dim].
        """
        cos = self.cos_cached[:seq_len, :].to(x.dtype)
        sin = self.sin_cached[:seq_len, :].to(x.dtype)

        # Match dimensions based on input rank
        if x.dim() == 4:
            cos = cos.unsqueeze(0).unsqueeze(1) # [1, 1, seq_len, dim]
            sin = sin.unsqueeze(0).unsqueeze(1)
        elif x.dim() == 3:
            cos = cos.unsqueeze(0) # [1, seq_len, dim]
            sin = sin.unsqueeze(0)

        return (x * cos) + (self._rotate_half(x) * sin)
