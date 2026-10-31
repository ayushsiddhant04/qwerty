"""
FlashCache Core module.
"""

from flashcache.core.cache_state import BlockState, ChunkStatus
from flashcache.core.block_allocator import BlockAllocator
from flashcache.core.cache_manager import LayerCacheManager

__all__ = [
    "BlockState",
    "ChunkStatus",
    "BlockAllocator",
    "LayerCacheManager",
]

