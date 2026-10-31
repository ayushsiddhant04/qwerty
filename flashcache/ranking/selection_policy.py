"""
Selection Policy and Ranking Result Data Structures (Stage 3).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any
from enum import Enum

from flashcache.chunking.chunk_metadata import KVChunk


class RankingCriteria(Enum):
    """Criteria used to rank middle candidate chunks."""
    SCORE = "score"                       # Pure composite importance score
    SCORE_DENSITY = "score_density"       # Score normalized by memory byte size


@dataclass
class RankingResult:
    """
    Detailed audit and outcome of Stage 3 budget-aware chunk ranking.
    """
    selected_chunks: List[KVChunk] = field(default_factory=list)
    evicted_chunks: List[KVChunk] = field(default_factory=list)
    protected_chunks: List[KVChunk] = field(default_factory=list)
    
    # Detailed tracking per head: (layer, head) -> {'budget': int, 'used': int}
    head_utilization: Dict[Tuple[int, int], Dict[str, int]] = field(default_factory=dict)
    
    total_selected_tokens: int = 0
    total_evicted_tokens: int = 0
    total_protected_tokens: int = 0
    
    # Memory statistics in physical bytes
    original_bytes: int = 0
    retained_bytes: int = 0
    evicted_bytes: int = 0

    @property
    def total_retained_tokens(self) -> int:
        return self.total_protected_tokens + self.total_selected_tokens

    @property
    def retention_ratio(self) -> float:
        total = self.total_retained_tokens + self.total_evicted_tokens
        return (self.total_retained_tokens / total) if total > 0 else 0.0

    @property
    def memory_reduction_pct(self) -> float:
        if self.original_bytes == 0:
            return 0.0
        return ((self.original_bytes - self.retained_bytes) / self.original_bytes) * 100.0

    def __repr__(self) -> str:
        return (
            f"RankingResult(retained_tokens={self.total_retained_tokens} "
            f"({self.total_protected_tokens} prot + {self.total_selected_tokens} sel), "
            f"evicted_tokens={self.total_evicted_tokens}, "
            f"memory_reduction={self.memory_reduction_pct:.2f}%)"
        )
