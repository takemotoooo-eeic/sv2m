"""MVPt model implementation for video-music contrastive learning.

This module implements the MVPt (Music Video Pre-training) architecture for learning joint
embeddings of video and music audio through contrastive learning.
"""

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
        video_input: torch.Tensor,
        music_input: torch.Tensor,
        video_mask: Optional[torch.Tensor] = None,
        music_mask: Optional[torch.Tensor] = None,
        music_ids: Optional[list[str]] = None,
        apply_normalization: Optional[Union[bool, Callable]] = True,
    ) -> Union[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Forward pass of MVPt.

        Args:
            video_input (torch.Tensor): Video input tensor.
            music_input (torch.Tensor): Music input tensor.
            video_mask (Optional[torch.Tensor]): Video mask tensor.
            music_mask (Optional[torch.Tensor]): Music mask tensor.
            video_ids (Optional[list[str]]): Video ids.
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
        video_embeddings, video_mask = self.video_encoder(video_input, video_mask)
        music_embeddings, music_mask = self.music_encoder(music_input, music_mask)

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
            return video_embeddings, video_mask, music_embeddings, music_mask

        loss = self.loss_fn(
            video_features=video_embeddings,
            video_masks=video_mask,
            music_features=music_embeddings,
            music_masks=music_mask,
            music_ids=music_ids,
        )

        return video_embeddings, video_mask, music_embeddings, music_mask, loss
