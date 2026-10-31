"""
Cache Manager per Layer and Attention Heads (Stage 5 Core).
"""

from typing import Dict, List, Optional, Tuple
import threading
import torch

from flashcache.core.cache_state import BlockState
from flashcache.chunking.chunk_metadata import KVChunk
from flashcache.compression.dequantizer import KIVIDequantizer


class LayerCacheManager:
    """
    Maintains the ordered collection of active and compressed KV chunks
    for a single transformer layer, and assembles continuous cache views for forward attention.
    """

    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx
        self._lock = threading.Lock()
        # Mapping head_idx -> List[KVChunk] ordered by start_pos
        self._head_chunks: Dict[int, List[KVChunk]] = {}

    def set_head_chunks(self, head_idx: int, chunks: List[KVChunk]):
        with self._lock:
            # Sort chronologically by token start_pos
            self._head_chunks[head_idx] = sorted(chunks, key=lambda c: c.metadata.start_pos)

    def get_head_chunks(self, head_idx: int) -> List[KVChunk]:
        with self._lock:
            return list(self._head_chunks.get(head_idx, []))

    def assemble_layer_tensors(
        self,
        target_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Dequantizes compressed historical blocks on-the-fly and concatenates them
        with high-precision recent blocks to reconstruct full [batch, num_heads, seq_len, head_dim]
        tensors for attention computation.
        """
        with self._lock:
            if not self._head_chunks:
                return None, None

            head_indices = sorted(list(self._head_chunks.keys()))
            num_heads = len(head_indices)

            head_k_list = []
            head_v_list = []

            for h in head_indices:
                chunks = self._head_chunks[h]
                if not chunks:
                    continue

                chunk_k_list = []
                chunk_v_list = []

                for c in chunks:
                    if c.metadata.state == BlockState.COMPRESSED and c.compressed_key is not None:
                        # Dequantize compressed key & value
                        recon_k = KIVIDequantizer.dequantize(c.compressed_key, target_dtype=target_dtype)
                        recon_v = KIVIDequantizer.dequantize(c.compressed_value, target_dtype=target_dtype)
                        chunk_k_list.append(recon_k)
                        chunk_v_list.append(recon_v)
                    elif c.key is not None and c.value is not None:
                        # Direct high-precision BF16/FP16 view
                        chunk_k_list.append(c.key.to(target_dtype) if target_dtype else c.key)
                        chunk_v_list.append(c.value.to(target_dtype) if target_dtype else c.value)

                if chunk_k_list and chunk_v_list:
                    # Concatenate along sequence axis (dim=-2)
                    head_k = torch.cat(chunk_k_list, dim=-2)
                    head_v = torch.cat(chunk_v_list, dim=-2)
                    head_k_list.append(head_k)
                    head_v_list.append(head_v)

            if not head_k_list or not head_v_list:
                return None, None

            # Stack/Concatenate across heads (dim=1)
            # Each head tensor is [batch, 1, seq_len, head_dim]
            layer_keys = torch.cat(head_k_list, dim=1)
            layer_values = torch.cat(head_v_list, dim=1)
            return layer_keys, layer_values
