"""MVPt model implementation for video-music contrastive learning.

This module implements the MVPt (Music Video Pre-training) architecture for learning joint
embeddings of video and music audio through contrastive learning.
"""

from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tower import ModalTowerWrapper
from .video import CLIPVideoEncoder

__all__ = [
    "MVPt",
    "ModalTowerWrapper",
    "CLIPVideoEncoder",
]


class MVPt(nn.Module):
    """MVPt model for video-music contrastive learning.

    The MVPt model learns joint representations of video and music by encoding
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
        >>> from musreel.models.mvpt import MVPt, ModalTowerWrapper
        >>> from musreel.criterion import CrossModalInfoNCELoss
        >>> video_encoder = ModalTowerWrapper(video_backbone, out_channels=128)
        >>> music_encoder = ModalTowerWrapper(audio_backbone, out_channels=128)
        >>> loss_fn = CrossModalInfoNCELoss(temperature=0.1)
        >>> model = MVPt(video_encoder, music_encoder, loss_fn)
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
        apply_normalization: Optional[Union[bool, Callable]] = True,
    ) -> Union[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Forward pass of MVPt.

        Args:
            video_input (torch.Tensor): Video input tensor.
            music_input (torch.Tensor): Music input tensor.
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
        video_embeddings = self.video_encoder(video_input)
        music_embeddings = self.music_encoder(music_input)

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
            return video_embeddings, music_embeddings

        loss = self.loss_fn(video_embeddings, music_embeddings)

        return video_embeddings, music_embeddings, loss