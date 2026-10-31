"""
Block Allocator and Physical Memory Tracking (Stage 5 Core).
"""

from typing import Dict, Optional, Set
import threading
import torch

from flashcache.core.cache_state import BlockState
from flashcache.chunking.chunk_metadata import KVChunk


class BlockAllocator:
    """
    Manages the physical registration, state transitions, and memory reclamation
    for all active and historical KV cache blocks with thread-safe locking.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._blocks: Dict[str, KVChunk] = {}
        self._total_allocated_bytes: int = 0
        self._total_reclaimed_bytes: int = 0

    def register_block(self, chunk: KVChunk):
        """Registers a newly partitioned or allocated KV block."""
        with self._lock:
            self._blocks[chunk.metadata.chunk_id] = chunk
            self._total_allocated_bytes += chunk.uncompressed_bytes

    def get_block(self, chunk_id: str) -> Optional[KVChunk]:
        with self._lock:
            return self._blocks.get(chunk_id)

    def transition_state(self, chunk_id: str, new_state: BlockState):
        """Thread-safe lifecycle state transition."""
        with self._lock:
            chunk = self._blocks.get(chunk_id)
            if chunk is not None:
                chunk.metadata.state = new_state

    def reclaim_block(self, chunk_id: str) -> bool:
        """
        Reclaims source memory if and only if the block is in RECLAIMABLE state.
        Never reclaims active BF16 or pending blocks prematurely.
        """
        with self._lock:
            chunk = self._blocks.get(chunk_id)
            if chunk is None:
                return False

            if chunk.metadata.state != BlockState.RECLAIMABLE:
                return False

            # Safely release uncompressed source tensors
            freed = chunk.uncompressed_bytes
            self._total_reclaimed_bytes += freed
            # Drop uncompressed tensor buffers while retaining metadata & compressed representations
            chunk.key = None
            chunk.value = None
            return True

    def get_stats(self) -> Dict[str, int]:
        """Calculates exact memory breakdown across all registered blocks."""
        with self._lock:
            active_bf16_bytes = 0
            compressed_bytes = 0
            pending_bytes = 0
            reclaimable_bytes = 0

            state_counts = {state: 0 for state in BlockState}

            for chunk in self._blocks.values():
                st = chunk.metadata.state
                state_counts[st] += 1
                if st == BlockState.ACTIVE_BF16:
                    active_bf16_bytes += chunk.uncompressed_bytes
                elif st == BlockState.COMPRESSED:
                    compressed_bytes += chunk.memory_bytes
                elif st == BlockState.COMPRESSION_PENDING:
                    pending_bytes += chunk.uncompressed_bytes
                elif st == BlockState.RECLAIMABLE:
                    reclaimable_bytes += chunk.uncompressed_bytes

            return {
                "total_blocks": len(self._blocks),
                "active_bf16_blocks": state_counts[BlockState.ACTIVE_BF16],
                "compressed_blocks": state_counts[BlockState.COMPRESSED],
                "pending_blocks": state_counts[BlockState.COMPRESSION_PENDING],
                "failed_blocks": state_counts[BlockState.COMPRESSION_FAILED],
                "reclaimable_blocks": state_counts[BlockState.RECLAIMABLE],
                "active_bf16_bytes": active_bf16_bytes,
                "compressed_bytes": compressed_bytes,
                "pending_bytes": pending_bytes,
                "reclaimable_bytes": reclaimable_bytes,
                "total_reclaimed_bytes": self._total_reclaimed_bytes,
            }
