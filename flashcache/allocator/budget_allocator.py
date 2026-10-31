"""
Budget Allocator Data Structures and Base Definitions (Stage 2).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Tuple, List, Optional
import math


class AllocationPolicy(Enum):
    """
    Supported budget allocation strategies.
    """
    UNIFORM = "uniform"
    IMPORTANCE_WEIGHTED = "importance_weighted"
    RECENCY_AWARE = "recency_aware"


@dataclass
class HeadBudgets:
    """
    Mapping of (layer_idx, head_idx) to explicit token retention budgets.
    """
    budgets: Dict[Tuple[int, int], int] = field(default_factory=dict)
    total_budget: int = 0
    policy: AllocationPolicy = AllocationPolicy.UNIFORM

    def get_budget(self, layer_idx: int, head_idx: int) -> int:
        return self.budgets.get((layer_idx, head_idx), 0)

    @property
    def allocated_tokens(self) -> int:
        return sum(self.budgets.values())

    @property
    def num_heads(self) -> int:
        return len(self.budgets)

    def __repr__(self) -> str:
        return (
            f"HeadBudgets(policy={self.policy.value}, "
            f"allocated={self.allocated_tokens}/{self.total_budget} tokens, "
            f"units={len(self.budgets)})"
        )
