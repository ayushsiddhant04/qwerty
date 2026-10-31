"""
Budget-Aware Chunk Ranker Implementation (Stage 3).
"""

from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from flashcache.core.cache_state import ChunkStatus
from flashcache.chunking.chunk_metadata import KVChunk
from flashcache.allocator.budget_allocator import HeadBudgets
from flashcache.ranking.selection_policy import RankingCriteria, RankingResult


class ChunkRanker:
    """
    Ranks historical candidate chunks and enforces head-level token retention budgets.
    Preserves attention sinks and recent tokens while pruning low-priority blocks.
    """

    def __init__(
        self,
        criteria: RankingCriteria = RankingCriteria.SCORE,
        budget_includes_protected: bool = True,
    ):
        """
        Args:
            criteria: Ranking metric ('score' or 'score_density').
            budget_includes_protected: If True, protected sink/recent tokens
                consume budget from their respective head allowance.
        """
        self.criteria = criteria
        self.budget_includes_protected = budget_includes_protected

    def rank_and_select(
        self,
        candidates: List[KVChunk],
        head_budgets: HeadBudgets,
    ) -> RankingResult:
        """
        Executes budget-aware chunk selection.

        Args:
            candidates: List of all KVChunks (sinks, middle candidates, recent).
            head_budgets: Stage 2 HeadBudgets allocation mapping.

        Returns:
            RankingResult with separated protected, selected, and evicted chunks.
        """
        # Segregate chunks by (layer_idx, head_idx)
        protected_by_head: Dict[Tuple[int, int], List[KVChunk]] = defaultdict(list)
        middle_by_head: Dict[Tuple[int, int], List[KVChunk]] = defaultdict(list)

        for c in candidates:
            u = (c.metadata.layer_idx, c.metadata.head_idx)
            if c.metadata.is_sink or c.metadata.is_recent:
                c.metadata.status = ChunkStatus.PROTECTED
                protected_by_head[u].append(c)
            else:
                middle_by_head[u].append(c)

        selected_chunks: List[KVChunk] = []
        evicted_chunks: List[KVChunk] = []
        all_protected_chunks: List[KVChunk] = []
        head_utilization: Dict[Tuple[int, int], Dict[str, int]] = {}

        # Collect all active units
        all_units = set(protected_by_head.keys()).union(set(middle_by_head.keys()))
        if head_budgets.budgets:
            all_units = all_units.union(set(head_budgets.budgets.keys()))

        total_orig_bytes = sum(c.uncompressed_bytes for c in candidates)

        for u in sorted(list(all_units)):
            allocated_budget = head_budgets.get_budget(u[0], u[1])
            prots = protected_by_head.get(u, [])
            all_protected_chunks.extend(prots)

            # Calculate tokens consumed by protected blocks
            prot_tokens = sum(c.metadata.token_count for c in prots)

            # Determine available budget for middle candidates
            if self.budget_includes_protected:
                avail_tokens = max(0, allocated_budget - prot_tokens)
            else:
                avail_tokens = allocated_budget

            # Sort middle candidate chunks by criteria
            mids = middle_by_head.get(u, [])
            if self.criteria == RankingCriteria.SCORE_DENSITY:
                # Score divided by memory bytes (avoid division by 0)
                mids.sort(
                    key=lambda c: (c.metadata.importance_score / max(1, c.uncompressed_bytes)),
                    reverse=True
                )
            else:
                mids.sort(key=lambda c: c.metadata.importance_score, reverse=True)

            # Greedily select middle chunks under available budget
            used_candidate_tokens = 0
            for c in mids:
                c_tok = c.metadata.token_count
                if used_candidate_tokens + c_tok <= avail_tokens:
                    c.metadata.status = ChunkStatus.SELECTED
                    selected_chunks.append(c)
                    used_candidate_tokens += c_tok
                else:
                    c.metadata.status = ChunkStatus.EVICTED
                    evicted_chunks.append(c)

            total_used_for_head = prot_tokens + used_candidate_tokens
            head_utilization[u] = {
                "budget": allocated_budget,
                "used": total_used_for_head,
                "protected_tokens": prot_tokens,
                "selected_tokens": used_candidate_tokens,
            }

        # Calculate summary statistics
        sel_tok = sum(c.metadata.token_count for c in selected_chunks)
        evict_tok = sum(c.metadata.token_count for c in evicted_chunks)
        prot_tok = sum(c.metadata.token_count for c in all_protected_chunks)

        retained_bytes = (
            sum(c.uncompressed_bytes for c in all_protected_chunks) +
            sum(c.uncompressed_bytes for c in selected_chunks)
        )
        evicted_bytes = sum(c.uncompressed_bytes for c in evicted_chunks)

        return RankingResult(
            selected_chunks=selected_chunks,
            evicted_chunks=evicted_chunks,
            protected_chunks=all_protected_chunks,
            head_utilization=head_utilization,
            total_selected_tokens=sel_tok,
            total_evicted_tokens=evict_tok,
            total_protected_tokens=prot_tok,
            original_bytes=total_orig_bytes,
            retained_bytes=retained_bytes,
            evicted_bytes=evicted_bytes,
        )

