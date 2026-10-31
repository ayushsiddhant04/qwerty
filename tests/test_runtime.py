"""
Unit Tests for Stage 5: FlashCache Runtime and Block Allocation Engine.
"""

import pytest
import torch

from flashcache.core.cache_state import BlockState, ChunkStatus
from flashcache.core.block_allocator import BlockAllocator
from flashcache.core.cache_manager import LayerCacheManager
from flashcache.chunking import ChunkMetadata, KVChunk
from flashcache.runtime import FlashCacheRuntime, MemoryMonitor


def test_block_allocator_safety_and_reclamation():
    """Verify that BlockAllocator strictly forbids premature reclamation."""
    allocator = BlockAllocator()

    key = torch.randn(1, 1, 16, 64)
    val = torch.randn(1, 1, 16, 64)
    meta = ChunkMetadata(
        chunk_id="test_alloc_1",
        layer_idx=0,
        head_idx=0,
        start_pos=0,
        end_pos=16,
        dtype=key.dtype,
        state=BlockState.ACTIVE_BF16,
    )
    chunk = KVChunk(key, val, meta)
    allocator.register_block(chunk)

    # 1. Attempt reclamation while ACTIVE_BF16 (must be rejected)
    assert allocator.reclaim_block("test_alloc_1") is False
    assert chunk.key is not None

    # 2. Attempt reclamation while COMPRESSION_PENDING (must be rejected)
    allocator.transition_state("test_alloc_1", BlockState.COMPRESSION_PENDING)
    assert allocator.reclaim_block("test_alloc_1") is False
    assert chunk.key is not None

    # 3. Transition to RECLAIMABLE -> reclamation succeeds
    allocator.transition_state("test_alloc_1", BlockState.RECLAIMABLE)
    assert allocator.reclaim_block("test_alloc_1") is True
    # Tensor buffers released
    assert chunk.key is None
    assert chunk.value is None


def test_flashcache_runtime_compaction_and_reconstruction():
    """Verify runtime coordinates all 5 stages and allows dequantized cache reads."""
    runtime = FlashCacheRuntime(
        chunk_size=16,
        sink_tokens=4,
        recent_tokens=16,
        total_budget=64,
        bits=4,
    )

    batch_size = 1
    num_heads = 2
    seq_len = 100
    head_dim = 64

    keys = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    values = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)

    # Compact layer 0
    ranking_res = runtime.compact_layer(keys=keys, values=values, layer_idx=0)
    assert ranking_res.total_retained_tokens <= 64
    assert len(ranking_res.protected_chunks) > 0

    # Query reconstructed past_key_values
    past_k, past_v = runtime.get_past_key_values(layer_idx=0, target_dtype=torch.bfloat16)
    assert past_k is not None
    assert past_v is not None
    assert past_k.shape[0] == batch_size
    assert past_k.shape[1] == num_heads
    assert past_k.shape[-1] == head_dim
    # Sequence length of assembled view equals total retained tokens per head
    retained_per_head = ranking_res.total_retained_tokens // num_heads
    assert past_k.shape[2] == retained_per_head

    # Verify memory statistics
    stats = runtime.get_memory_stats()
    assert stats["compression_ratio"] > 1.0
    assert stats["memory_reduction_pct"] > 0.0
    assert stats["total_blocks"] > 0

    runtime.shutdown()


def test_hardware_memory_monitor():
    """Verify hardware info inspection and telemetry calculations."""
    hw_info = MemoryMonitor.get_hardware_info()
    assert "device" in hw_info
    assert "cpu_count" in hw_info
    assert hw_info["cpu_count"] > 0

    stats = MemoryMonitor.get_current_memory_stats(
        original_kv_bytes=1000000,
        compacted_kv_bytes=250000,
    )
    assert stats["compression_ratio"] == pytest.approx(4.0, abs=1e-3)
    assert stats["memory_reduction_pct"] == pytest.approx(75.0, abs=1e-3)
