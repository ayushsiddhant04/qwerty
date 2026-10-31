"""
FlashCache Master Runtime Engine (Stage 5 Orchestrator).
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import time
import torch

from flashcache.core.cache_state import BlockState, ChunkStatus
from flashcache.core.block_allocator import BlockAllocator
from flashcache.core.cache_manager import LayerCacheManager
from flashcache.chunking import ChunkBuilder, ImportanceScorer, KVChunk
from flashcache.allocator import AdaptiveBudgetAllocator, AllocationPolicy, HeadBudgets
from flashcache.ranking import ChunkRanker, RankingCriteria, RankingResult
from flashcache.compression import CompressionManager
from flashcache.runtime.async_compressor import AsyncCompressor
from flashcache.runtime.memory_monitor import MemoryMonitor


class FlashCacheRuntime:
    """
    Master Runtime for FlashCache-KV: Intelligent Adaptive KV-Cache Management and Compression.
    Integrates all five stages into a cohesive, memory-safe, and benchmarkable engine.
    """

    def __init__(
        self,
        chunk_size: int = 16,
        sink_tokens: int = 4,
        recent_tokens: int = 32,
        total_budget: int = 256,
        bits: int = 4,
        allocation_policy: Union[AllocationPolicy, str] = AllocationPolicy.IMPORTANCE_WEIGHTED,
        ranking_criteria: RankingCriteria = RankingCriteria.SCORE,
        enable_async: bool = False,
        device: Optional[str] = None,
    ):
        """
        Args:
            chunk_size: Token count per partitioned chunk.
            sink_tokens: Count of initial tokens permanently protected.
            recent_tokens: Count of recent tokens protected uncompressed.
            total_budget: Maximum allowed retained tokens across all heads per layer.
            bits: Quantization bit-width (2 or 4).
            allocation_policy: Budget allocator strategy.
            ranking_criteria: Ranking metric.
            enable_async: Whether to execute quantization in background threads.
            device: 'cuda', 'cpu', or None (auto-detect).
        """
        self.chunk_size = chunk_size
        self.sink_tokens = sink_tokens
        self.recent_tokens = recent_tokens
        self.total_budget = total_budget
        self.bits = bits
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.enable_async = enable_async

        # Pipeline Stage 1: Chunk Builder
        self.scorer = ImportanceScorer()
        self.builder = ChunkBuilder(
            chunk_size=chunk_size,
            sink_tokens=sink_tokens,
            recent_tokens=recent_tokens,
            scorer=self.scorer,
        )

        # Pipeline Stage 2: Budget Allocator
        self.allocator = AdaptiveBudgetAllocator(policy=allocation_policy)

        # Pipeline Stage 3: Chunk Ranker
        self.ranker = ChunkRanker(criteria=ranking_criteria, budget_includes_protected=True)

        # Pipeline Stage 4: Compression Manager
        cuda_stream = torch.cuda.Stream() if (self.device == "cuda" and torch.cuda.is_available()) else None
        self.compression_manager = CompressionManager(bits=bits, cuda_stream=cuda_stream)

        # Pipeline Stage 5: Core Storage and Asynchronous Engine
        self.block_allocator = BlockAllocator()
        self.layer_managers: Dict[int, LayerCacheManager] = {}
        self.async_compressor = AsyncCompressor(self.compression_manager) if enable_async else None

        # Telemetry
        self.last_ranking_result: Optional[RankingResult] = None
        self.last_compaction_time_ms: float = 0.0
        self.layer_original_bytes: Dict[int, int] = {}

    def get_or_create_layer_manager(self, layer_idx: int) -> LayerCacheManager:
        if layer_idx not in self.layer_managers:
            self.layer_managers[layer_idx] = LayerCacheManager(layer_idx)
        return self.layer_managers[layer_idx]

    def compact_layer(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        layer_idx: int = 0,
        attention_scores: Optional[torch.Tensor] = None,
    ) -> RankingResult:
        """
        Executes the full 5-stage FlashCache-KV pipeline for a single transformer layer.
        
        STAGE 1: ChunkKV Partitioning & Scoring
        STAGE 2: Adaptive Budget Allocation
        STAGE 3: Budget-Aware Chunk Ranking & Eviction
        STAGE 4: KIVI-Inspired Asymmetric Quantization & Memory Safety Barrier
        STAGE 5: Cache Assembly & Safe Memory Reclamation
        """
        start_time = time.perf_counter()

        # STAGE 1: ChunkKV Candidate Construction
        candidates = self.builder.partition_layer(
            keys=keys,
            values=values,
            layer_idx=layer_idx,
            attention_scores=attention_scores,
        )

        # Register raw candidate blocks in allocator
        for c in candidates:
            self.block_allocator.register_block(c)

        # STAGE 2: Adaptive KV Budget Allocation
        num_heads = keys.shape[1]
        head_budgets = self.allocator.allocate(
            total_budget=self.total_budget,
            candidates=candidates,
            num_layers=1,
            num_heads=num_heads,
        )

        # STAGE 3: Budget-Aware Chunk Ranking
        ranking_result = self.ranker.rank_and_select(
            candidates=candidates,
            head_budgets=head_budgets,
        )
        self.last_ranking_result = ranking_result
        self.layer_original_bytes[layer_idx] = ranking_result.original_bytes

        # Immediately release evicted blocks from active memory
        for c in ranking_result.evicted_chunks:
            self.block_allocator.transition_state(c.metadata.chunk_id, BlockState.RECLAIMABLE)
            self.block_allocator.reclaim_block(c.metadata.chunk_id)

        # STAGE 4: KIVI-Inspired Asymmetric Quantization
        # Only SELECTED middle chunks are compressed; PROTECTED sink/recent stay in high precision
        chunks_to_compress = ranking_result.selected_chunks

        if self.async_compressor is not None and self.enable_async:
            # Asynchronous compression
            for c in chunks_to_compress:
                self.async_compressor.submit_compression(c)
            # Wait for completion to guarantee safe publishing before attention reads
            self.async_compressor.wait_all()
        else:
            # Synchronous deterministic compression
            for c in chunks_to_compress:
                self.compression_manager.compress_chunk_safe(c)

        # STAGE 5: FlashCache Runtime Update & Memory Reclamation
        # Transition successfully compressed blocks to RECLAIMABLE and free original BF16 source storage
        for c in chunks_to_compress:
            if c.metadata.state == BlockState.COMPRESSED:
                self.block_allocator.transition_state(c.metadata.chunk_id, BlockState.RECLAIMABLE)
                self.block_allocator.reclaim_block(c.metadata.chunk_id)

        # Update layer cache view
        layer_mgr = self.get_or_create_layer_manager(layer_idx)
        # Retained blocks = protected + selected
        retained = ranking_result.protected_chunks + ranking_result.selected_chunks

        # Group by head and populate layer cache
        heads_dict: Dict[int, List[KVChunk]] = {}
        for c in retained:
            h = c.metadata.head_idx
            heads_dict.setdefault(h, []).append(c)

        for h, h_chunks in heads_dict.items():
            layer_mgr.set_head_chunks(h, h_chunks)

        self.last_compaction_time_ms = (time.perf_counter() - start_time) * 1000.0
        return ranking_result

    def get_past_key_values(
        self,
        layer_idx: int,
        target_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Reconstructs dequantized key and value tensors for a layer,
        ready for the current decoding step's attention computation.
        """
        layer_mgr = self.layer_managers.get(layer_idx)
        if layer_mgr is None:
            return None, None
        return layer_mgr.assemble_layer_tensors(target_dtype=target_dtype)

    def get_memory_stats(self) -> Dict[str, Any]:
        """Provides full empirical memory metrics and compaction ratios."""
        alloc_stats = self.block_allocator.get_stats()
        
        # Calculate active cache physical bytes across retained blocks
        active_cache_bytes = 0
        for layer_mgr in self.layer_managers.values():
            for h in layer_mgr._head_chunks:
                for c in layer_mgr._head_chunks[h]:
                    active_cache_bytes += c.memory_bytes
        
        # Total uncompressed original bytes across all active layers
        total_orig_bytes = sum(self.layer_original_bytes.values())
        if total_orig_bytes == 0:
            total_orig_bytes = active_cache_bytes
        
        mem_telemetry = MemoryMonitor.get_current_memory_stats(
            original_kv_bytes=total_orig_bytes,
            compacted_kv_bytes=active_cache_bytes,
        )
        return {
            **mem_telemetry,
            **alloc_stats,
            "last_compaction_time_ms": self.last_compaction_time_ms,
        }

    def shutdown(self):
        if self.async_compressor is not None:
            self.async_compressor.shutdown()
