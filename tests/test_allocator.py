"""
Unit Tests for Stage 2: Adaptive KV Budget Allocator.
"""

import pytest
import torch

from flashcache.chunking import ChunkBuilder
from flashcache.allocator import AdaptiveBudgetAllocator, AllocationPolicy, HeadBudgets


@pytest.fixture
def sample_candidates():
    keys = torch.randn(1, 4, 128, 64)
    values = torch.randn(1, 4, 128, 64)
    builder = ChunkBuilder(chunk_size=16, sink_tokens=4, recent_tokens=16)
    return builder.partition_layer(keys, values, layer_idx=0)


def test_uniform_allocator_strict_budget_enforcement(sample_candidates):
    """Verify uniform allocator allocates exactly total_budget tokens without overflow."""
    total_budget = 100
    allocator = AdaptiveBudgetAllocator(policy=AllocationPolicy.UNIFORM)
    budgets = allocator.allocate(total_budget=total_budget, candidates=sample_candidates)

    assert budgets.allocated_tokens == total_budget
    assert budgets.num_heads == 4
    # 100 / 4 = 25 tokens per head
    for budget_val in budgets.budgets.values():
        assert budget_val == 25


def test_uniform_allocator_non_divisible_budget(sample_candidates):
    """Verify remainder distribution ensures exact match when total_budget % heads != 0."""
    total_budget = 103 # 103 % 4 = 3
    allocator = AdaptiveBudgetAllocator(policy=AllocationPolicy.UNIFORM)
    budgets = allocator.allocate(total_budget=total_budget, candidates=sample_candidates)

    assert budgets.allocated_tokens == total_budget
    values = list(budgets.budgets.values())
    assert values.count(26) == 3
    assert values.count(25) == 1


def test_importance_weighted_allocator(sample_candidates):
    """Verify importance-weighted policy gives more budget to higher scoring units."""
    # Artificially modify scores of head 0 candidates
    for c in sample_candidates:
        if c.metadata.head_idx == 0:
            c.metadata.importance_score = 10.0
        else:
            c.metadata.importance_score = 0.1

    total_budget = 120
    allocator = AdaptiveBudgetAllocator(
        policy=AllocationPolicy.IMPORTANCE_WEIGHTED,
        min_head_budget=10,
    )
    budgets = allocator.allocate(total_budget=total_budget, candidates=sample_candidates)

    assert budgets.allocated_tokens == total_budget
    assert budgets.get_budget(0, 0) > budgets.get_budget(0, 1)
    # Check floor guarantee
    for val in budgets.budgets.values():
        assert val >= 10


def test_allocator_zero_budget_and_empty():
    """Verify edge cases: 0 budget or empty candidates list."""
    allocator = AdaptiveBudgetAllocator()
    res_zero = allocator.allocate(total_budget=0, candidates=[])
    assert res_zero.allocated_tokens == 0

    res_empty_cands = allocator.allocate(total_budget=50, candidates=[])
    assert res_empty_cands.allocated_tokens == 0
