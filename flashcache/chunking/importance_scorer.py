"""
Multi-signal Importance Scorer for KV Chunks.
"""

import math
from typing import Optional, Dict
import torch


class ImportanceScorer:
    """
    Computes composite importance scores for candidate KV chunks using
    empirical attention signals, recency decay, and tensor magnitude heuristics.
    Never invents or fabricates attention signals when absent.
    """

    def __init__(
        self,
        w_attn: float = 0.5,
        w_recency: float = 0.3,
        w_norm: float = 0.2,
        recency_decay: float = 2.0,
    ):
        """
        Args:
            w_attn: Weight allocated to empirical attention mass.
            w_recency: Weight allocated to temporal recency.
            w_norm: Weight allocated to tensor Frobenius norm / magnitude.
            recency_decay: Exponential decay coefficient.
        """
        self.w_attn = float(w_attn)
        self.w_recency = float(w_recency)
        self.w_norm = float(w_norm)
        self.recency_decay = float(recency_decay)

    def calculate_score(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        start_pos: int,
        end_pos: int,
        total_seq_len: int,
        attention_slice: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[float]]:
        """
        Calculates normalized composite importance score.

        Returns:
            Dictionary containing:
                - 'composite_score': float
                - 'recency_score': float
                - 'attention_score': Optional[float]
                - 'norm_score': float
        """
        total_seq_len = max(1, total_seq_len)
        span = max(1, end_pos - start_pos)

        # 1. Recency Score (Exponential temporal decay towards prompt start)
        # Closer to total_seq_len -> higher score -> approaching 1.0
        distance_from_tail = max(0, total_seq_len - end_pos)
        norm_dist = distance_from_tail / total_seq_len
        recency_score = float(math.exp(-self.recency_decay * norm_dist))

        # 2. Frobenius Norm Score (Historical activation magnitude)
        with torch.no_grad():
            k_norm = torch.linalg.norm(key.float()).item() / (math.sqrt(key.numel()) + 1e-6)
            v_norm = torch.linalg.norm(value.float()).item() / (math.sqrt(value.numel()) + 1e-6)
            # Soft-clamp magnitude into [0, 1] range via sigmoid/tanh-style mapping
            raw_mag = (k_norm + v_norm) / 2.0
            norm_score = float(math.tanh(raw_mag))

        # 3. Attention Score (if truthfully provided)
        attention_score: Optional[float] = None
        if attention_slice is not None and isinstance(attention_slice, torch.Tensor) and attention_slice.numel() > 0:
            with torch.no_grad():
                # Mean attention allocated to this token span
                attn_val = attention_slice.float().mean().item()
                attention_score = max(0.0, float(attn_val))

        # 4. Composite Score Formulation with Dynamic Weight Normalization
        if attention_score is not None:
            total_weight = self.w_attn + self.w_recency + self.w_norm
            comp = (
                self.w_attn * attention_score +
                self.w_recency * recency_score +
                self.w_norm * norm_score
            ) / max(1e-6, total_weight)
        else:
            # Rebalance between recency and norm when attention is unavailable
            total_weight = self.w_recency + self.w_norm
            comp = (
                self.w_recency * recency_score +
                self.w_norm * norm_score
            ) / max(1e-6, total_weight)

        return {
            "composite_score": max(0.0, comp),
            "recency_score": recency_score,
            "attention_score": attention_score,
            "norm_score": norm_score,
        }
