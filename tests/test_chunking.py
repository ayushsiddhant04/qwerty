"""
Unit Tests for Stage 1: ChunkKV Candidate Construction.
"""

import pytest
import torch

from flashcache.core.cache_state import BlockState, ChunkStatus
from flashcache.chunking import ChunkBuilder, ChunkMetadata, KVChunk, ImportanceScorer


def test_chunk_builder_partitioning_coverage():
    """Verify that chunks seamlessly partition the full sequence without gaps or overlaps."""
    batch_size = 1
    num_heads = 2
    seq_len = 100
    head_dim = 64
    chunk_size = 16
    sink_tokens = 4
    recent_tokens = 20

    keys = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    values = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)

    builder = ChunkBuilder(
        chunk_size=chunk_size,
        sink_tokens=sink_tokens,
        recent_tokens=recent_tokens,
    )
    chunks = builder.partition_layer(keys, values, layer_idx=0)

    # Filter by head 0
    head0_chunks = [c for c in chunks if c.metadata.head_idx == 0]
    assert len(head0_chunks) > 0

    # Verify sink chunk
    sink = head0_chunks[0]
    assert sink.metadata.is_sink is True
    assert sink.metadata.status == ChunkStatus.PROTECTED
    assert sink.metadata.start_pos == 0
    assert sink.metadata.end_pos == sink_tokens

    # Verify recent chunk
    recent = head0_chunks[-1]
    assert recent.metadata.is_recent is True
    assert recent.metadata.status == ChunkStatus.PROTECTED
    assert recent.metadata.start_pos == seq_len - recent_tokens
    assert recent.metadata.end_pos == seq_len

    # Verify contiguous coverage across all chunks for head 0
    expected_start = 0
    for c in head0_chunks:
        assert c.metadata.start_pos == expected_start, f"Gap detected before {c.metadata.chunk_id}"
        expected_start = c.metadata.end_pos
    assert expected_start == seq_len, "Partition did not cover full sequence length"


def test_chunk_memory_bytes_accuracy():
    """Verify exact tensor memory bytes calculation against analytical formula."""
    batch_size = 1
    num_heads = 1
    seq_len = 32
    head_dim = 128
    keys = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32) # 4 bytes
    values = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)

    builder = ChunkBuilder(chunk_size=16, sink_tokens=0, recent_tokens=0)
    chunks = builder.partition_layer(keys, values, layer_idx=0)

    for c in chunks:
        expected_bytes = (c.key.numel() * 4) + (c.value.numel() * 4)
        assert c.memory_bytes == expected_bytes
        assert c.uncompressed_bytes == expected_bytes
        assert c.metadata.memory_bytes == expected_bytes


def test_importance_scorer_behavior_with_and_without_attention():
    """Verify importance scorer handles present and absent attention truthfully."""
    scorer = ImportanceScorer(w_attn=0.5, w_recency=0.3, w_norm=0.2)
    key = torch.randn(1, 1, 16, 64)
    value = torch.randn(1, 1, 16, 64)

    # 1. Without attention scores
    res_no_attn = scorer.calculate_score(
        key=key,
        value=value,
        start_pos=16,
        end_pos=32,
        total_seq_len=64,
        attention_slice=None,
    )
    assert res_no_attn["attention_score"] is None
    assert 0.0 <= res_no_attn["composite_score"] <= 1.0
    assert 0.0 <= res_no_attn["recency_score"] <= 1.0

    # 2. With attention scores
    attn = torch.full((1, 16), 0.75)
    res_with_attn = scorer.calculate_score(
        key=key,
        value=value,
        start_pos=16,
        end_pos=32,
        total_seq_len=64,
        attention_slice=attn,
    )
    assert res_with_attn["attention_score"] == pytest.approx(0.75, abs=1e-5)
    assert 0.0 <= res_with_attn["composite_score"] <= 1.0


def test_chunk_builder_short_sequence():
    """Verify behavior when sequence length is smaller than sink + recent tokens."""
    keys = torch.randn(1, 2, 10, 64)
    values = torch.randn(1, 2, 10, 64)

    builder = ChunkBuilder(chunk_size=16, sink_tokens=4, recent_tokens=20)
    chunks = builder.partition_layer(keys, values)

    # All tokens should be protected (sink: 4, recent: 6)
    head0_chunks = [c for c in chunks if c.metadata.head_idx == 0]
    assert all(c.metadata.status == ChunkStatus.PROTECTED for c in head0_chunks)
    total_tokens = sum(c.metadata.token_count for c in head0_chunks)
    assert total_tokens == 10
