"""
Chunk Builder for ChunkKV Candidate Construction.
"""

from typing import List, Optional, Tuple, Dict
import torch

from flashcache.core.cache_state import BlockState, ChunkStatus
from flashcache.chunking.chunk_metadata import ChunkMetadata, KVChunk
from flashcache.chunking.importance_scorer import ImportanceScorer


class ChunkBuilder:
    """
    Partitions raw historical Key and Value tensors into discrete candidate chunks,
    preserves attention sink tokens and recent sliding windows, and assigns
    multi-signal importance scores.
    """

    def __init__(
        self,
        chunk_size: int = 16,
        sink_tokens: int = 4,
        recent_tokens: int = 32,
        scorer: Optional[ImportanceScorer] = None,
    ):
        """
        Args:
            chunk_size: Number of tokens per chunk.
            sink_tokens: Number of initial tokens permanently protected as attention sinks.
            recent_tokens: Number of recent tail tokens protected uncompressed.
            scorer: Custom or default ImportanceScorer instance.
        """
        self.chunk_size = max(1, int(chunk_size))
        self.sink_tokens = max(0, int(sink_tokens))
        self.recent_tokens = max(0, int(recent_tokens))
        self.scorer = scorer or ImportanceScorer()

    def partition_layer(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        layer_idx: int = 0,
        attention_scores: Optional[torch.Tensor] = None,
    ) -> List[KVChunk]:
        """
        Partitions a single layer's keys and values into KVChunks.

        Args:
            keys: Tensor of shape [batch_size, num_heads, seq_len, head_dim]
            values: Tensor of shape [batch_size, num_heads, seq_len, head_dim]
            layer_idx: Transformer layer index
            attention_scores: Optional tensor of shape [batch_size, num_heads, seq_len]
                              or [num_heads, seq_len]

        Returns:
            List of constructed KVChunk objects.
        """
        if keys.dim() != 4 or values.dim() != 4:
            raise ValueError(
                f"Expected keys and values of shape [batch, num_heads, seq_len, head_dim], "
                f"got keys shape {keys.shape} and values shape {values.shape}"
            )

        batch_size, num_heads, seq_len, head_dim = keys.shape
        chunks: List[KVChunk] = []

        for head_idx in range(num_heads):
            head_k = keys[:, head_idx:head_idx + 1, :, :]      # [batch, 1, seq_len, head_dim]
            head_v = values[:, head_idx:head_idx + 1, :, :]    # [batch, 1, seq_len, head_dim]
            
            head_attn = None
            if attention_scores is not None:
                if attention_scores.dim() == 3:
                    head_attn = attention_scores[:, head_idx, :] # [batch, seq_len]
                elif attention_scores.dim() == 2:
                    head_attn = attention_scores[head_idx, :]    # [seq_len]

            head_chunks = self._partition_head(
                key=head_k,
                value=head_v,
                layer_idx=layer_idx,
                head_idx=head_idx,
                seq_len=seq_len,
                attention_slice=head_attn,
            )
            chunks.extend(head_chunks)

        return chunks

    def _partition_head(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
        head_idx: int,
        seq_len: int,
        attention_slice: Optional[torch.Tensor],
    ) -> List[KVChunk]:
        """Partitions tensors for a single attention head."""
        chunks: List[KVChunk] = []

        if seq_len == 0:
            return chunks

        # Boundary calculations
        effective_sink = min(self.sink_tokens, seq_len)
        effective_recent = min(self.recent_tokens, max(0, seq_len - effective_sink))
        middle_start = effective_sink
        middle_end = max(middle_start, seq_len - effective_recent)

        # 1. Attention Sink Chunk(s)
        if effective_sink > 0:
            sink_k = key[:, :, :effective_sink, :]
            sink_v = value[:, :, :effective_sink, :]
            sink_meta = ChunkMetadata(
                chunk_id=f"L{layer_idx}_H{head_idx}_sink_0_{effective_sink}",
                layer_idx=layer_idx,
                head_idx=head_idx,
                start_pos=0,
                end_pos=effective_sink,
                dtype=key.dtype,
                is_sink=True,
                is_recent=False,
                importance_score=float("inf"), # Permanent preservation
                recency_score=0.0,
                attention_score=1.0,
                norm_score=1.0,
                state=BlockState.ACTIVE_BF16,
                status=ChunkStatus.PROTECTED,
            )
            chunks.append(KVChunk(sink_k, sink_v, sink_meta))

        # 2. Middle Historical Candidate Chunks
        current_pos = middle_start
        chunk_idx = 0
        while current_pos < middle_end:
            chunk_end = min(current_pos + self.chunk_size, middle_end)
            c_k = key[:, :, current_pos:chunk_end, :]
            c_v = value[:, :, current_pos:chunk_end, :]

            # Compute attention sub-slice if available
            c_attn = None
            if attention_slice is not None:
                c_attn = attention_slice[..., current_pos:chunk_end]

            scores = self.scorer.calculate_score(
                key=c_k,
                value=c_v,
                start_pos=current_pos,
                end_pos=chunk_end,
                total_seq_len=seq_len,
                attention_slice=c_attn,
            )

            meta = ChunkMetadata(
                chunk_id=f"L{layer_idx}_H{head_idx}_c{chunk_idx}_{current_pos}_{chunk_end}",
                layer_idx=layer_idx,
                head_idx=head_idx,
                start_pos=current_pos,
                end_pos=chunk_end,
                dtype=key.dtype,
                is_sink=False,
                is_recent=False,
                importance_score=scores["composite_score"],
                recency_score=scores["recency_score"],
                attention_score=scores["attention_score"],
                norm_score=scores["norm_score"],
                state=BlockState.ACTIVE_BF16,
                status=ChunkStatus.CANDIDATE,
            )
            chunks.append(KVChunk(c_k, c_v, meta))
            current_pos = chunk_end
            chunk_idx += 1

        # 3. Recent Local Window Chunk(s)
        if effective_recent > 0:
            rec_start = seq_len - effective_recent
            rec_k = key[:, :, rec_start:seq_len, :]
            rec_v = value[:, :, rec_start:seq_len, :]
            rec_meta = ChunkMetadata(
                chunk_id=f"L{layer_idx}_H{head_idx}_recent_{rec_start}_{seq_len}",
                layer_idx=layer_idx,
                head_idx=head_idx,
                start_pos=rec_start,
                end_pos=seq_len,
                dtype=key.dtype,
                is_sink=False,
                is_recent=True,
                importance_score=float("inf"), # High priority retention
                recency_score=1.0,
                attention_score=None,
                norm_score=1.0,
                state=BlockState.ACTIVE_BF16,
                status=ChunkStatus.PROTECTED,
            )
            chunks.append(KVChunk(rec_k, rec_v, rec_meta))

        return chunks
