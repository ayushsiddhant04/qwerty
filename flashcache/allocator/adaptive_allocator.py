"""
Adaptive Budget Allocator Implementation (Stage 2).
"""

from typing import List, Optional, Tuple, Dict, Union
import math

from flashcache.chunking.chunk_metadata import KVChunk
from flashcache.allocator.budget_allocator import AllocationPolicy, HeadBudgets


class AdaptiveBudgetAllocator:
    """
    Deterministic budget allocator distributing token retention allowances
    across attention heads and layers according to configured policies.
    Guarantees no budget overflow.
    """

    def __init__(
        self,
        policy: Union[AllocationPolicy, str] = AllocationPolicy.IMPORTANCE_WEIGHTED,
        min_head_budget: int = 0,
    ):
        """
        Args:
            policy: Strategy ('uniform', 'importance_weighted', 'recency_aware')
            min_head_budget: Minimum token floor guaranteed per head to prevent starvation.
        """
        if isinstance(policy, str):
            policy = AllocationPolicy(policy.lower())
        self.policy = policy
        self.min_head_budget = max(0, int(min_head_budget))

    def allocate(
        self,
        total_budget: int,
        candidates: List[KVChunk],
        num_layers: Optional[int] = None,
        num_heads: Optional[int] = None,
    ) -> HeadBudgets:
        """
        Computes explicit per-head token budgets.

        Args:
            total_budget: Total allowed retained tokens across all units.
            candidates: List of candidate KV chunks from Stage 1.
            num_layers: Optional explicit layer count.
            num_heads: Optional explicit head count per layer.

        Returns:
            HeadBudgets mapping (layer_idx, head_idx) -> token allowance.
        """
        total_budget = max(0, int(total_budget))

        # Identify all distinct (layer_idx, head_idx) allocation units
        units: set[Tuple[int, int]] = set()
        for c in candidates:
            units.add((c.metadata.layer_idx, c.metadata.head_idx))

        # If explicit layer/head counts provided, ensure all are represented
        if num_layers is not None and num_heads is not None:
            for l in range(num_layers):
                for h in range(num_heads):
                    units.add((l, h))

        sorted_units = sorted(list(units))
        n_units = len(sorted_units)

        if n_units == 0 or total_budget == 0:
            return HeadBudgets(budgets={}, total_budget=total_budget, policy=self.policy)

        # Handle extreme case where total_budget < n_units * min_head_budget
        effective_min = self.min_head_budget
        if total_budget < n_units * effective_min:
            effective_min = total_budget // n_units

        # 1. Uniform Allocation Strategy
        if self.policy == AllocationPolicy.UNIFORM:
            base = total_budget // n_units
            remainder = total_budget % n_units
            budgets_dict = {}
            for i, u in enumerate(sorted_units):
                budgets_dict[u] = base + (1 if i < remainder else 0)
            return HeadBudgets(budgets=budgets_dict, total_budget=total_budget, policy=self.policy)

        # 2. Importance-Weighted Strategy
        elif self.policy == AllocationPolicy.IMPORTANCE_WEIGHTED:
            # Aggregate importance score per unit
            scores: Dict[Tuple[int, int], float] = {u: 0.0 for u in sorted_units}
            for c in candidates:
                u = (c.metadata.layer_idx, c.metadata.head_idx)
                # Protected sink/recent tokens have score inf; treat as baseline
                score_val = c.metadata.importance_score
                if math.isinf(score_val):
                    score_val = 1.0
                scores[u] += max(0.0, float(score_val))

            return self._proportional_allocation(
                total_budget=total_budget,
                sorted_units=sorted_units,
                weights=scores,
                min_budget=effective_min,
                policy=self.policy,
            )

        # 3. Recency-Aware Strategy
        elif self.policy == AllocationPolicy.RECENCY_AWARE:
            recency_scores: Dict[Tuple[int, int], float] = {u: 0.0 for u in sorted_units}
            for c in candidates:
                u = (c.metadata.layer_idx, c.metadata.head_idx)
                # Bias towards later layers (higher index) and recent tokens
                layer_weight = 1.0 + (c.metadata.layer_idx * 0.1)
                recency_scores[u] += c.metadata.recency_score * layer_weight

            return self._proportional_allocation(
                total_budget=total_budget,
                sorted_units=sorted_units,
                weights=recency_scores,
                min_budget=effective_min,
                policy=self.policy,
            )

        else:
            raise ValueError(f"Unsupported allocation policy: {self.policy}")

    def _proportional_allocation(
        self,
        total_budget: int,
        sorted_units: List[Tuple[int, int]],
        weights: Dict[Tuple[int, int], float],
        min_budget: int,
        policy: AllocationPolicy,
    ) -> HeadBudgets:
        """
        Distributes budget using the Largest Remainder (Hare-Niemeyer) method,
        guaranteeing exact integer allocation with zero overflow and floor protection.
        """
        n_units = len(sorted_units)
        reserved_total = min_budget * n_units
        remaining_pool = max(0, total_budget - reserved_total)

        total_weight = sum(weights.values())
        budgets_dict: Dict[Tuple[int, int], int] = {u: min_budget for u in sorted_units}

        if total_weight <= 1e-9 or remaining_pool == 0:
            # Fallback to uniform distribution of remaining tokens
            base = remaining_pool // n_units
            rem = remaining_pool % n_units
            for i, u in enumerate(sorted_units):
                budgets_dict[u] += base + (1 if i < rem else 0)
            return HeadBudgets(budgets=budgets_dict, total_budget=total_budget, policy=policy)

        # Compute exact quota and fractional parts
        fractional_parts: List[Tuple[float, Tuple[int, int]]] = []
        allocated_so_far = 0

        for u in sorted_units:
            share = (weights[u] / total_weight) * remaining_pool
            integer_part = int(math.floor(share))
            frac_part = share - integer_part
            budgets_dict[u] += integer_part
            allocated_so_far += integer_part
            fractional_parts.append((frac_part, u))

        # Distribute remaining tokens to units with largest fractional parts
        leftover = remaining_pool - allocated_so_far
        # Sort descending by fractional remainder
        fractional_parts.sort(key=lambda x: x[0], reverse=True)
        for i in range(leftover):
            u_target = fractional_parts[i % n_units][1]
            budgets_dict[u_target] += 1

        return HeadBudgets(budgets=budgets_dict, total_budget=total_budget, policy=policy)
