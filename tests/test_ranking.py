"""
Unit Tests for Stage 3: Budget-Aware Chunk Ranking.
"""

import pytest
import torch

from flashcache.core.cache_state import ChunkStatus
from flashcache.chunking import ChunkBuilder
from flashcache.allocator import AdaptiveBudgetAllocator, AllocationPolicy
from flashcache.ranking import ChunkRanker, RankingCriteria, RankingResult


@pytest.fixture
def ranked_pipeline_setup():
    seq_len = 120
    chunk_size = 16
    sink_tokens = 4
    recent_tokens = 20

    keys = torch.randn(1, 2, seq_len, 64)
    values = torch.randn(1, 2, seq_len, 64)

    builder = ChunkBuilder(
        chunk_size=chunk_size,
        sink_tokens=sink_tokens,
        recent_tokens=recent_tokens,
    )
    candidates = builder.partition_layer(keys, values, layer_idx=0)
    return candidates, seq_len, sink_tokens, recent_tokens


def test_ranking_preserves_sink_and_recent_tokens(ranked_pipeline_setup):
    """Verify sink and recent tokens are never evicted even under small budget."""
    candidates, seq_len, sink_tokens, recent_tokens = ranked_pipeline_setup

    # Allocate minimal budget (only enough for protected tokens)
    allocator = AdaptiveBudgetAllocator(policy=AllocationPolicy.UNIFORM)
    # Total budget per head = 24 tokens (exactly 4 sink + 20 recent)
    head_budgets = allocator.allocate(total_budget=48, candidates=candidates)

    ranker = ChunkRanker(budget_includes_protected=True)
    result = ranker.rank_and_select(candidates=candidates, head_budgets=head_budgets)

    # Protected tokens: 2 heads * (4 + 20) = 48 tokens
    assert result.total_protected_tokens == 48
    assert result.total_selected_tokens == 0 # No budget left for middle chunks
    assert result.total_evicted_tokens > 0
    assert len(result.protected_chunks) == 4 # 2 sink chunks + 2 recent chunks
    assert all(c.metadata.status == ChunkStatus.PROTECTED for c in result.protected_chunks)
    assert all(c.metadata.status == ChunkStatus.EVICTED for c in result.evicted_chunks)


def test_ranking_selects_top_chunks_within_budget(ranked_pipeline_setup):
    """Verify middle candidates with highest importance scores are selected first."""
    candidates, seq_len, sink_tokens, recent_tokens = ranked_pipeline_setup

    # Generous budget: enough for protected tokens + exactly 1 middle chunk per head (16 tokens)
    # 24 protected + 16 middle = 40 per head -> 80 total
    allocator = AdaptiveBudgetAllocator(policy=AllocationPolicy.UNIFORM)
    head_budgets = allocator.allocate(total_budget=80, candidates=candidates)

    ranker = ChunkRanker(budget_includes_protected=True)
    result = ranker.rank_and_select(candidates=candidates, head_budgets=head_budgets)

    assert result.total_protected_tokens == 48
    assert result.total_selected_tokens == 32 # 16 tokens * 2 heads
    assert len(result.selected_chunks) == 2
    assert all(c.metadata.status == ChunkStatus.SELECTED for c in result.selected_chunks)


def test_memory_accounting_conservation(ranked_pipeline_setup):
    """Verify exact conservation: original_bytes == retained_bytes + evicted_bytes."""
    candidates, _, _, _ = ranked_pipeline_setup

    allocator = AdaptiveBudgetAllocator(policy=AllocationPolicy.UNIFORM)
    head_budgets = allocator.allocate(total_budget=100, candidates=candidates)

    ranker = ChunkRanker()
    result = ranker.rank_and_select(candidates=candidates, head_budgets=head_budgets)

    assert result.original_bytes > 0
    assert result.original_bytes == result.retained_bytes + result.evicted_bytes
    assert 0.0 <= result.memory_reduction_pct <= 100.0

