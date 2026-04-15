"""CLIP-based video encoder for MVPt."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from sv2m.modules.aggregater import Aggregator
from sv2m.modules.head import Head


class CLIPVideoEncoder(nn.Module):
    """Video encoder using CLIP vision model as backbone with temporal modeling.

    This encoder processes video frames by:
    1. Encoding each frame independently with CLIP extractor
    2. Adding temporal positional embeddings
    3. Processing through a Transformer encoder for temporal modeling
    4. Optionally aggregating and projecting the output

    Args:
        extractor (nn.Module): CLIP vision model for frame-level feature extraction.
        temporal_embedding (nn.Module): Positional embedding module for temporal modeling.
        temporal_backbone (nn.TransformerEncoder): Transformer encoder for temporal processing.
        aggregator (Aggregator, optional): Aggregator for pooling temporal features.
        head (Head, optional): Output projection head.
        freeze_extractor (bool): Whether to freeze the extractor parameters. Default: True.

    Examples:
        >>> encoder = CLIPVideoEncoder.from_pretrained("openai/clip-vit-base-patch32")
        >>> video = torch.randn(2, 8, 3, 224, 224)  # (batch, frames, channels, height, width)
        >>> embedding = encoder(video)  # (2, hidden_size)
    """

    def __init__(
        self,
        temporal_embedding: nn.Module,
        temporal_backbone: nn.TransformerEncoder,
        aggregator: Aggregator = None,
        head: Head = None,
    ) -> None:
        super().__init__()

        self.temporal_embedding = temporal_embedding
        self.temporal_backbone = temporal_backbone
        self.aggregator = aggregator
        self.head = head

        self._reset_parameters()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Encode video frames into a single embedding.

        Args:
            input: Video tensor of shape (batch_size, num_frames, num_channels, height, width)

        Returns:
            Video embedding of shape (batch_size, hidden_size)
        """
        x = self.temporal_embedding_forward(x)
        output = self.temporal_backbone_forward(x)

        if self.aggregator is not None:
            output = self.aggregator(output)

        if self.head is not None:
            output = self.head(output)

        return output

    def temporal_embedding_forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.temporal_embedding(input)

    def temporal_backbone_forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.temporal_backbone(input)

    def prepend_head_tokens(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.temporal_embedding.prepend_head_tokens(sequence)

    def prepend_tokens(
        self, sequence: torch.Tensor, tokens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.temporal_embedding.prepend_tokens(sequence, tokens=tokens)

    def split_sequence(self, sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split sequence to head tokens and content tokens.

        Args:
            sequence (torch.Tensor): Sequence containing head tokens, i.e. class and distillation
                tokens or corresponding mask. If the tokens are given, the shape should be
                (batch_size, length, embedding_dim). Otherwise (mask is given), the shape should be
                (batch_size, length).

        Returns:
            tuple: Tuple of tensors containing

                - torch.Tensor: Head tokens of shape (batch_size, num_head_tokens, embedding_dim).
                - torch.Tensor: Sequence of shape
                    (batch_size, length - num_head_tokens, embedding_dim).

        .. note::

            This method is applicable even when sequence does not contain head tokens. In that
            case, an empty sequence is returned as the first item of returned tensors.

        """
        return self.temporal_embedding.split_sequence(sequence)

    def _reset_parameters(self) -> None:
        for p in self.temporal_backbone.parameters():
            if p.dim() > 1 and p.requires_grad:
                nn.init.xavier_uniform_(p)