"""Modal tower wrapper for MVPt encoders.

This module provides a wrapper class for encoder backbones (video or audio) used in
the MVPt model, adding a projection layer to map encoder outputs to a shared
embedding space.
"""

from typing import Optional

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from transformers import BertModel
from transformers.tokenization_utils_base import BatchEncoding

from ..mvpt.video import CLIPVideoEncoder
from ..mvpt.ast import ModifiedAudioSpectrogramTransformer

__all__ = [
    "ModalTowerWrapper",
]


class ModalTowerWrapper(nn.Module):
    """Wrapper for encoder backbones in MVPt architecture.

    This wrapper adds a linear projection layer on top of encoder backbones
    (video or audio) to map their outputs to a shared embedding space of
    specified dimensionality. Supports automatic detection of backbone output
    dimensions for common architectures.

    Args:
        backbone (nn.Module): The encoder backbone network. Supported backbones include:
            - CLIPVideoEncoder
            - ModifiedAudioSpectrogramTransformer
            - SentenceTransformer
        out_channels (int): Output embedding dimension for the shared space.
        hidden_channels (int, optional): Dimension of backbone output. If None,
            automatically inferred from backbone architecture. Default: None.
        freeze_backbone (bool): If True, freezes backbone parameters during training.
            Useful for fine-tuning only the projection layer. Default: False.

    Raises:
        NotImplementedError: If backbone type is not supported for automatic
            hidden_channels inference.

    Examples:
        >>> from musreel.models.ast import ModifiedAudioSpectrogramTransformer
        >>> from musreel.models.mvpt import ModalTowerWrapper
        >>> audio_backbone = ModifiedAudioSpectrogramTransformer.from_pretrained(
        ...     "ast-base-stride10"
        ... )
        >>> audio_encoder = ModalTowerWrapper(
        ...     audio_backbone, out_channels=128, freeze_backbone=False
        ... )

    """

    def __init__(
        self,
        backbone: nn.Module,
        out_channels: int,
        hidden_channels: Optional[int] = None,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        self.backbone = backbone

        if hidden_channels is None:
            if isinstance(
                backbone,
                (
                    ModifiedAudioSpectrogramTransformer,
                    CLIPVideoEncoder,
                ),
            ):
                backbone: ModifiedAudioSpectrogramTransformer
                hidden_channels = backbone.embedding_dim
            elif isinstance(backbone, SentenceTransformer):
                backbone: SentenceTransformer
                hidden_channels = backbone[-1].word_embedding_dimension
            else:
                raise NotImplementedError(
                    f"{type(backbone)} is not supported as backbone network."
                )

        self.linear = nn.Linear(hidden_channels, out_channels)

        self.freeze_backbone = freeze_backbone

        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass of tower wrapper.

        .. note::

            You need to apply L2 normalization after forward function.

        Args:
            args (tuple): Positional arguments given to backbone.
            kwargs (dict): Keyword arguments given to backbone.

        Returns:
            torch.Tensor: Feature of shape (*, out_channels).

        """
        backbone = self.backbone
        embedding = backbone(*args, **kwargs)

        if isinstance(backbone, SentenceTransformer):
            assert isinstance(embedding, BatchEncoding), (
                f"Invalid type {type(embedding)} is detected."
            )

            embedding = embedding["sentence_embedding"]
        elif isinstance(
            backbone,
            (
                ModifiedAudioSpectrogramTransformer,
                CLIPVideoEncoder,
            ),
        ):
            pass
        else:
            raise ValueError(f"{type(backbone)} is not supported as backbone.")

        output = self.linear(embedding)

        return output

    def load_state_dict(self, state_dict, **kwargs) -> nn.modules.module._IncompatibleKeys:
        if isinstance(self.backbone, BertModel):
            # for backward compatibility of BertModel in transformers
            self_state_dict = self.state_dict()
            is_state_dict_pos_ids_available = (
                "backbone.embeddings.position_ids" in state_dict.keys()
            )
            is_model_pos_ids_available = (
                "backbone.embeddings.position_ids" in self_state_dict.keys()
            )

            if is_state_dict_pos_ids_available and not is_model_pos_ids_available:
                # remove "backbone.embeddings.position_ids" from given state_dict
                position_ids = state_dict.pop("backbone.embeddings.position_ids")
                self_position_ids = self.backbone.embeddings.position_ids.to(position_ids.device)
                assert torch.allclose(position_ids, self_position_ids)
            elif not is_state_dict_pos_ids_available and is_model_pos_ids_available:
                # define "backbone.embeddings.position_ids" in state_dict
                state_dict["backbone.embeddings.position_ids"] = self_state_dict[
                    "backbone.embeddings.position_ids"
                ]

        return super().load_state_dict(state_dict=state_dict, **kwargs)