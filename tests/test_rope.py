import pytest
import torch
from gpt_from_scratch.rope import RotaryPositionEmbedding

def test_rope_initialization_and_shapes():
    dim = 64
    max_seq_len = 128
    rope = RotaryPositionEmbedding(dim=dim, max_seq_len=max_seq_len)
    
    batch_size = 2
    seq_len = 16
    x = torch.randn(batch_size, seq_len, dim)
    out = rope(x, seq_len=seq_len)
    
    assert out.shape == x.shape
    assert not torch.allclose(out, x)

def test_rope_4d_attention_tensor():
    batch = 2
    heads = 4
    seq_len = 32
    head_dim = 32
    
    rope = RotaryPositionEmbedding(dim=head_dim, max_seq_len=64)
    q = torch.randn(batch, heads, seq_len, head_dim)
    q_rot = rope(q, seq_len=seq_len)
    
    assert q_rot.shape == (batch, heads, seq_len, head_dim)
    # Energy / norm should remain approximately preserved
    norm_orig = torch.norm(q, dim=-1)
    norm_rot = torch.norm(q_rot, dim=-1)
    assert torch.allclose(norm_orig, norm_rot, atol=1e-4)
