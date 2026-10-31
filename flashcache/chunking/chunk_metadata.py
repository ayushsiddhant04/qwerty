"""
Chunk Metadata and KVChunk data structures.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import torch

from flashcache.core.cache_state import BlockState, ChunkStatus


@dataclass
class ChunkMetadata:
    """
    Metadata describing the token span, origin, importance metrics,
    and lifecycle status of a KV chunk.
    """
    chunk_id: str
    layer_idx: int
    head_idx: int
    start_pos: int
    end_pos: int
    dtype: torch.dtype
    is_sink: bool = False
    is_recent: bool = False
    importance_score: float = 0.0
    recency_score: float = 0.0
    attention_score: Optional[float] = None
    norm_score: float = 0.0
    state: BlockState = BlockState.ACTIVE_BF16
    status: ChunkStatus = ChunkStatus.CANDIDATE
    memory_bytes: int = 0
    error_log: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def token_count(self) -> int:
        return max(0, self.end_pos - self.start_pos)


class KVChunk:
    """
    Self-contained container binding historical Key-Value tensor slices
    with their associated ChunkMetadata and lifecycle management references.
    """

    def __init__(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        metadata: ChunkMetadata,
    ):
        """
        key shape: [batch_size, 1, token_count, head_dim] or [token_count, head_dim]
        value shape: [batch_size, 1, token_count, head_dim] or [token_count, head_dim]
        """
        self.key = key
        self.value = value
        self.metadata = metadata
        
        # Compressed representations (populated when Stage 4 completes)
        self.compressed_key: Optional[Any] = None
        self.compressed_value: Optional[Any] = None

        # Calculate exact memory bytes
        self._update_memory_bytes()

    def _update_memory_bytes(self):
        k_bytes = self.key.numel() * self.key.element_size() if isinstance(self.key, torch.Tensor) else 0
        v_bytes = self.value.numel() * self.value.element_size() if isinstance(self.value, torch.Tensor) else 0
        self.metadata.memory_bytes = k_bytes + v_bytes

    @property
    def memory_bytes(self) -> int:
        if self.metadata.state == BlockState.COMPRESSED and self.compressed_key is not None and self.compressed_value is not None:
            comp_k = getattr(self.compressed_key, "total_bytes", 0)
            comp_v = getattr(self.compressed_value, "total_bytes", 0)
            return comp_k + comp_v
        return self.metadata.memory_bytes

    @property
    def uncompressed_bytes(self) -> int:
        return self.metadata.memory_bytes

    def __repr__(self) -> str:
        return (
            f"KVChunk(id='{self.metadata.chunk_id}', "
            f"layer={self.metadata.layer_idx}, head={self.metadata.head_idx}, "
            f"tokens=[{self.metadata.start_pos}:{self.metadata.end_pos}], "
            f"score={self.metadata.importance_score:.4f}, "
            f"state={self.metadata.state.value}, status={self.metadata.status.value})"
        )
