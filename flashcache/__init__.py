"""
FlashCache-KV: Intelligent Adaptive KV-Cache Management and Compression for LLM Inference.
"""

from flashcache.core.cache_state import BlockState, ChunkStatus
from flashcache.core.block_allocator import BlockAllocator
from flashcache.core.cache_manager import LayerCacheManager
from flashcache.chunking import ChunkBuilder, ImportanceScorer, KVChunk, ChunkMetadata
from flashcache.allocator import AdaptiveBudgetAllocator, AllocationPolicy, HeadBudgets
from flashcache.ranking import ChunkRanker, RankingCriteria, RankingResult
from flashcache.compression import (
    KIVIQuantizer,
    CompressedTensor,
    KIVIDequantizer,
    CompressionManager,
)
from flashcache.runtime import FlashCacheRuntime, MemoryMonitor, AsyncCompressor

__version__ = "0.2.0"

__all__ = [
    "BlockState",
    "ChunkStatus",
    "BlockAllocator",
    "LayerCacheManager",
    "ChunkBuilder",
    "ImportanceScorer",
    "KVChunk",
    "ChunkMetadata",
    "AdaptiveBudgetAllocator",
    "AllocationPolicy",
    "HeadBudgets",
    "ChunkRanker",
    "RankingCriteria",
    "RankingResult",
    "KIVIQuantizer",
    "CompressedTensor",
    "KIVIDequantizer",
    "CompressionManager",
    "FlashCacheRuntime",
    "MemoryMonitor",
    "AsyncCompressor",
]

