"""MVPt model implementation for video-music contrastive learning.

This module implements the MVPt (Music Video Pre-training) architecture for learning joint
embeddings of video and music audio through contrastive learning.
"""

import math
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import UnimodalEncoder

__all__ = [
    "MaDE",
    "UnimodalEncoder",
]


class MaDE(nn.Module):
    """MaDE model for video-music contrastive learning.

    The MaDE model learns joint representations of video and music by encoding
    both modalities into a shared embedding space using separate encoder towers
    and training with contrastive loss.

    Args:
        video_encoder (nn.Module): Encoder network for video inputs.
            Typically a vision model (e.g., CLIP ViT) wrapped in ModalTowerWrapper.
        music_encoder (nn.Module): Encoder network for music/audio inputs.
            Typically an Audio Spectrogram Transformer wrapped in ModalTowerWrapper.
        loss_fn (nn.Module, optional): Contrastive loss function for training.
            If provided, forward pass returns embeddings and loss. If None, only
            returns embeddings. Default: None.

    Examples:
        >>> from sv2m.models.made import MaDE, ModalTowerWrapper
        >>> from sv2m.criterion import CrossModalInfoNCELoss
        >>> video_encoder = UnimodalEncoder(projector=video_projector, temporal_backbone=video_backbone, activation=True)
        >>> music_encoder = UnimodalEncoder(projector=music_projector, temporal_backbone=music_backbone, activation=True)
        >>> loss_fn = CrossModalInfoNCELoss(temperature=0.1)
        >>> model = MaDE(video_encoder, music_encoder, loss_fn)
        >>> video_embedding, music_embedding, loss = model(video_input, music_input, apply_normalization=True)

    """  # noqa: E501

    def __init__(
        self,
        video_encoder: nn.Module,
        music_encoder: nn.Module,
        loss_fn: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        self.video_encoder = video_encoder
        self.music_encoder = music_encoder
        self.loss_fn = loss_fn

    def forward(
        self,
        video_feats: torch.Tensor,
        music_feats: torch.Tensor,
        video_masks: Optional[torch.Tensor] = None,
        music_masks: Optional[torch.Tensor] = None,
        music_span_masks: Optional[torch.Tensor] = None,
        spans_target: Optional[torch.Tensor] = None,
        music_ids: Optional[list[str]] = None,
        apply_normalization: Optional[Union[bool, Callable]] = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of MVPt.

        Args:
            video_feats (torch.Tensor): Video input tensor.
            music_feats (torch.Tensor): Music input tensor.
            video_masks (Optional[torch.Tensor]): Video mask tensor.
            music_masks (Optional[torch.Tensor]): Music mask tensor.
            music_span_masks (Optional[torch.Tensor]): Music span mask tensor.
            music_ids (Optional[list[str]]): Music ids.
            apply_normalization (Optional[Union[bool, Callable]]): Normalization to apply to embeddings.
                If None, applies L2 normalization by default.
                If bool True, applies L2 normalization.
                If bool False, no normalization is applied.
                If Callable, applies the custom normalization function.

        Returns:
            Union[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
                If loss_fn is None: Returns tuple of (video_embeddings, music_embeddings)
                If loss_fn is provided: Returns tuple of (video_embeddings, music_embeddings, loss)
        """  # noqa: E501
        video_embeddings, video_masks = self.video_encoder(video_feats, video_masks)
        music_embeddings, music_masks = self.music_encoder(music_feats, music_masks)

        if apply_normalization is None:
            pass
        elif isinstance(apply_normalization, bool):
            if apply_normalization:
                video_embeddings = F.normalize(video_embeddings, p=2, dim=-1)
                music_embeddings = F.normalize(music_embeddings, p=2, dim=-1)
            else:
                # If False, no normalization is applied
                pass
        elif callable(apply_normalization):
            video_embeddings = apply_normalization(video_embeddings)
            music_embeddings = apply_normalization(music_embeddings)
        else:
            raise ValueError(f"Invalid apply_normalization type: {type(apply_normalization)}")

        if self.loss_fn is None:
            return video_embeddings, video_masks, music_embeddings, music_masks

        loss, attention_weights = self.loss_fn(
            video_features=video_embeddings,
            video_masks=video_masks,
            music_features=music_embeddings,
            music_masks=music_masks,
            music_span_masks=music_span_masks,
            spans_target=spans_target,
            music_ids=music_ids,
        )

        if attention_weights is None:
            predict_spans = torch.zeros_like(spans_target)
        else:
            predict_spans = self._calculate_spans_from_attention(attention_weights, spans_target)

        return video_embeddings, video_masks, music_embeddings, music_masks, loss, predict_spans
    
    def _calculate_spans_from_attention(
        self,
        attention_weights: torch.Tensor,
        spans_target: torch.Tensor,
        max_music_duration: int = 240,
        stride: float = 2.5,
    ) -> torch.Tensor:
        """Estimate span from attention weights.

        Args:
            attention_weights: (V, M, S)
            spans_target: (V, 1, 2) in normalized [0, 1] format
            max_music_duration: float - maximum duration of music in seconds
            stride: float - stride of music segments in seconds

        Returns:
            predict_spans: (V, 1, 2) in normalized [0, 1] format
        """
        V, _, S = attention_weights.shape
        diag_idx = torch.arange(V, device=attention_weights.device) 
        attention_weights = attention_weights[diag_idx, diag_idx] # (V, S)

        width = (spans_target[:, 0, 1] * max_music_duration / stride).round().long()
        width = width.clamp(min=1, max=S) # (V,)

        # prefix sum を使って、各 start 位置ごとの window sum を O(V*S) で計算
        cumsum = torch.cat(
            [
                torch.zeros((V, 1), device=attention_weights.device, dtype=attention_weights.dtype),
                attention_weights.cumsum(dim=-1),
            ],
            dim=-1,
        )  # (V, S + 1)
        start_positions = torch.arange(S, device=attention_weights.device)
        start_positions = start_positions.unsqueeze(0).expand(V, S)  # (V, S)
        end_positions = start_positions + width.unsqueeze(-1)  # (V, S) 
        valid = end_positions <= S
        end_positions = end_positions.clamp(max=S)

        window_sums = cumsum.gather(dim=-1, index=end_positions) - cumsum.gather(dim=-1, index=start_positions)
        window_sums = window_sums.masked_fill(~valid, float("-inf"))

        best_start = window_sums.argmax(dim=-1)  # (V,)
        best_end = best_start + width - 1  # (V,)

        # start/end index を V, 2 で保持
        self.last_index_span = torch.zeros((V, 2), device=attention_weights.device, dtype=torch.long)
        self.last_index_span[:, 0] = best_start
        self.last_index_span[:, 1] = best_end

        # index span を normalized [0, 1] の center/width に戻す
        start_norm = (self.last_index_span[:, 0].to(dtype=attention_weights.dtype) + 0.5) / S
        end_norm = (self.last_index_span[:, 1].to(dtype=attention_weights.dtype) + 0.5) / S

        predict_spans = torch.zeros((V, 1, 2), device=attention_weights.device, dtype=attention_weights.dtype)
        predict_spans[:, 0, 0] = (start_norm + end_norm) / 2.0
        predict_spans[:, 0, 1] = (end_norm - start_norm).clamp(min=1.0 / S)

        return predict_spans


