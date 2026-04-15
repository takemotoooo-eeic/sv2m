# ported from https://github.com/tky823/TorchOnlyAST/blob/772b9f928f092101215f252696cd931d2eff121a/torch_only_ast/models/ast.py

import copy
import warnings
from abc import abstractmethod
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_2_t

from ...modules.patch_embedding import PositionalPatchEmbedding
from ...modules.head import Head, MLPHead
from ...modules.aggregater import Aggregator, AverageAggregator, HeadTokensAggregator
from ...utils import download_pretrained_model_from_vos

__all__ = [
    "ModifiedAudioSpectrogramTransformer",
    "Aggregator",
    "AverageAggregator",
    "HeadTokensAggregator",
    "Head",
    "MLPHead",
]


class _AudioSpectrogramTransformer(nn.Module):
    """Base class of audio spectrogram transformer."""

    def __init__(
        self,
        embedding: PositionalPatchEmbedding,
        backbone: nn.TransformerEncoder,
    ) -> None:
        super().__init__()

        self.embedding = embedding
        self.backbone = backbone

    def pad_by_length(
        self, input: torch.Tensor, length: Optional[torch.LongTensor] = None
    ) -> torch.Tensor:
        """Pad feature by length.

        Args:
            input (torch.Tensor): Spectrogram-like feature of shape (batch_size, n_bins, n_frames).
            length (torch.LongTensor, optional): Length of each sample in batch of
                shape (batch_size,).

        Returns:
            torch.Tensor: Padded feature of shape (batch_size, n_bins, n_frames).

        """
        if length is None:
            output = input
        else:
            factory_kwargs = {
                "device": input.device,
                "dtype": torch.long,
            }
            max_length = input.size(-1)
            padding_mask = torch.arange(max_length, **factory_kwargs) >= length.unsqueeze(dim=-1)
            output = input.masked_fill(padding_mask.unsqueeze(dim=-2), 0)

        return output

    def compute_patch_embedding(self, input: torch.Tensor) -> torch.Tensor:
        """Compute output shape from shape of spectrogram."""
        output = self.embedding.compute_patch_embedding(input)

        return output

    def apply_positional_embedding(
        self,
        input: torch.Tensor,
        n_bins: int,
        n_frames: int,
    ) -> torch.Tensor:
        """Apply positional embedding.

        Args:
            input (torch.Tensor): Patches of shape (batch_size, embedding_dim, height, width).
            n_bins (int): Number of bins, not height.
            n_frames (int): Number of frames, not width.

        Returns:
            torch.Tensor: Resampled positional embedding of shape (embedding_dim, height', width').

        """
        positional_embedding = self.embedding.positional_embedding

        output = input + self.embedding.resample_positional_embedding(
            positional_embedding,
            n_bins,
            n_frames,
        )

        return output

    def dropout_embedding(self, input: torch.Tensor) -> torch.Tensor:
        output = self.embedding.dropout(input)

        return output

    def compute_padding_mask(
        self,
        input: torch.Tensor,
        length: Optional[torch.LongTensor] = None,
    ) -> Optional[torch.BoolTensor]:
        """Compute padding mask.

        Args:
            input (torch.Tensor): Input feature of shape (batch_size, n_bins, max_frames).
            length (torch.LongTensor, optional): Length of input of shape (batch_size,).

        Returns:
            torch.BoolTensor: Padding mask of shape
                (batch_size, height * max_width + num_head_tokens).

        """
        if length is None:
            padding_mask = None
        else:
            factory_kwargs = {
                "dtype": torch.long,
                "device": length.device,
            }
            _, n_bins, max_frames = input.size()
            width = []

            for _length in length:
                n_frames = _length.item()
                _, _width = self.embedding.compute_output_shape(n_bins, n_frames)
                width.append(_width)

            width = torch.tensor(width, **factory_kwargs)
            max_height, max_width = self.embedding.compute_output_shape(n_bins, max_frames)
            padding_mask = torch.arange(max_width, **factory_kwargs) >= width.unsqueeze(dim=-1)
            padding_mask = padding_mask.unsqueeze(dim=-2)
            padding_mask = padding_mask.repeat((1, max_height, 1))
            padding_mask = self.patches_to_sequence(padding_mask)

            num_head_tokens = 0

            if self.embedding.insert_cls_token:
                num_head_tokens += 1

            if self.embedding.insert_dist_token:
                num_head_tokens += 1

            padding_mask = F.pad(padding_mask, (num_head_tokens, 0), value=False)

        return padding_mask

    def patch_transformer_forward(
        self,
        input: torch.Tensor,
        padding_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """Transformer with patch inputs.

        Args:
            input (torch.Tensor): Patch feature of shape
                (batch_size, embedding_dim, height, width).
            padding_mask (torch.BoolTensor): Padding mask of shape (batch_size, height, width).

        Returns:
            torch.Tensor: Estimated patches of shape (batch_size, embedding_dim, height, width).

        """
        _, _, height, width = input.size()

        x = self.patches_to_sequence(input)

        if padding_mask is not None:
            padding_mask = self.patches_to_sequence(padding_mask)

        x = self.transformer_forward(x, padding_mask=padding_mask)
        output = self.sequence_to_patches(x, height=height, width=width)

        return output

    def transformer_forward(
        self,
        input: torch.Tensor,
        padding_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """Run forward pass of backbone.

        Args:
            input (torch.Tensor): Sequence of shape (batch_size, length, embedding_dim).
            padding_mask (torch.BoolTensor, optional): Padding mask of shape (batch_size, length).

        Returns:
            torch.Tensor: Estimated sequence of shape (batch_size, length, embedding_dim).

        """
        if padding_mask is None:
            kwargs = {}
        else:
            if isinstance(self.backbone, nn.TransformerEncoder):
                kwargs = {
                    "src_key_padding_mask": padding_mask,
                }
            else:
                kwargs = {
                    "padding_mask": padding_mask,
                }

        output = self.backbone(input, **kwargs)

        return output

    def spectrogram_to_patches(self, input: torch.Tensor) -> torch.Tensor:
        """Convert spectrogram to patches.

        Actual implementation depends on ``self.embedding.spectrogram_to_patches``.

        """
        return self.embedding.spectrogram_to_patches(input)

    def patches_to_sequence(self, input: Union[torch.Tensor, torch.BoolTensor]) -> torch.Tensor:
        r"""Convert 3D (batch_size, height, width) or 4D (batch_size, embedding_dim, height, width)
        tensor to shape (batch_size, length, \*) for input of Transformer.

        Args:
            input (torch.Tensor): Patches of shape (batch_size, height, width) or
                (batch_size, embedding_dim, height, width).

        Returns:
            torch.Tensor: Sequence of shape (batch_size, length) or
                (batch_size, length, embedding_dim).

        """
        n_dims = input.dim()

        if n_dims == 3:
            batch_size, height, width = input.size()
            output = input.view(batch_size, height * width)
        elif n_dims == 4:
            batch_size, embedding_dim, height, width = input.size()
            x = input.view(batch_size, embedding_dim, height * width)
            output = x.permute(0, 2, 1).contiguous()
        else:
            raise ValueError("Only 3D and 4D tensors are supported.")

        return output

    def sequence_to_patches(
        self, input: Union[torch.Tensor, torch.BoolTensor], height: int, width: int
    ) -> torch.Tensor:
        r"""Convert (batch_size, max_length, \*) tensor to 3D (batch_size, height, width)
        or 4D (batch_size, embedding_dim, height, width) one.
        This method corresponds to inversion of ``patches_to_sequence``.
        """
        n_dims = input.dim()

        if n_dims == 2:
            batch_size, _ = input.size()
            output = input.view(batch_size, height, width)
        elif n_dims == 3:
            batch_size, _, embedding_dim = input.size()
            x = input.view(batch_size, height, width, embedding_dim)
            output = x.permute(0, 3, 1, 2).contiguous()
        else:
            raise ValueError("Only 2D and 3D tensors are supported.")

        return output

    def split_sequence(self, sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split sequence to head tokens and content tokens.

        Args:
            sequence (torch.Tensor): Sequence containing head tokens, i.e. class and distillation
                tokens or corresponding mask. If the tokens are given, the shape should be
                (batch_size, length, embedding_dim). Otherwise (mask is given), the shape should be
                (batch_size, length).

        Returns:
            tuple: Tuple of tensors containing

                - torch.Tensor: Head tokens of shape (batch_size, num_head_tokens, embedding_dim)
                    or (batch_size, num_head_tokens).
                - torch.Tensor: Sequence of shape
                    (batch_size, length - num_head_tokens, embedding_dim) or
                    (batch_size, length - num_head_tokens).

        .. note::

            This method is applicable even when sequence does not contain head tokens. In that
            case, an empty sequnce is returned as the first item of returned tensors.

        """
        n_dims = sequence.dim()

        if n_dims == 2:
            sequence = sequence.unsqueeze(dim=-1)

        head_tokens, sequence = self.embedding.split_sequence(sequence)

        if n_dims == 2:
            sequence = sequence.squeeze(dim=-1)

        return head_tokens, sequence

    def prepend_head_tokens(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.embedding.prepend_head_tokens(sequence)

    def prepend_tokens(
        self, sequence: torch.Tensor, tokens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Prepaned tokens to sequence.

        This method is inversion of ``split_sequence``.

        Args:
            sequence (torch.Tensor): Sequence of shape (batch_size, length, embedding_dim)
                or (batch_size, length).
            tokens (torch.Tensor, optional): Tokens of shape
                (batch_size, num_tokens, embedding_dim) or (batch_size, num_tokens).

        Returns:
            torch.Tensor: Concatenated sequence of shape
                (batch_size, length + num_tokens, embedding_dim)
                or (batch_size, length + num_tokens).

        """
        if tokens is None:
            return sequence
        else:
            if sequence.dim() == 2:
                # assume (batch_size, length) and (batch_size, num_tokens)
                return torch.cat([tokens, sequence], dim=-1)
            else:
                return torch.cat([tokens, sequence], dim=-2)

    @property
    def embedding_dim(self) -> int:
        return self.embedding.embedding_dim


class ModifiedAudioSpectrogramTransformer(_AudioSpectrogramTransformer):
    """Audio spectrogram transformer.

    Args:
        embedding (mulan2025.modules.vit.ModifiedPositionalPatchEmbedding): Patch embedding
            followed by positional embedding.
        backbone (nn.TransformerEncoder): Transformer (encoder).

    """

    def __init__(
        self,
        embedding: PositionalPatchEmbedding,
        backbone: nn.TransformerEncoder,
        aggregator: Optional["Aggregator"] = None,
        head: Optional["Head"] = None,
    ) -> None:
        super().__init__(embedding=embedding, backbone=backbone)

        self.aggregator = aggregator
        self.head = head

        if self.aggregator is None and self.head is not None:
            warnings.warn(
                "Head is given, but aggregator is not given, "
                "which may lead to unexpected behavior.",
                UserWarning,
                stacklevel=2,
            )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        stride: Optional[_size_2_t] = None,
        n_bins: Optional[int] = None,
        n_frames: Optional[int] = None,
        aggregator: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
        force_download: bool = False,
    ) -> "ModifiedAudioSpectrogramTransformer":
        """Build pretrained AudioSpectrogramTransformer.

        Args:
            pretrained_model_name_or_path (str): Path to pretrained model or name of pretrained model.
            aggregator (nn.Module, optional): Aggregator module.
            head (nn.Module, optional): Head module.

        Examples:

            >>> from mulan2025.models.ast import ModifiedAudioSpectrogramTransformer
            >>> model = ModifiedAudioSpectrogramTransformer.from_pretrained("ast-base-stride10")

        .. note::

            Supported pretrained model names are
                - ast-base-stride10

        """  # noqa: E501

        if pretrained_model_name_or_path == "ast-base-stride10":
            expected_sha256 = "f30b1b777a1f849d644daff15895b85000150e9d5eaf2ede71f8beddac0ae2d1"
            path = download_pretrained_model_from_vos(
                "mulan2025/pretrained_models/public/modified-audio-spectrogram-transformer/ast-base-stride10.pth",
                force_download=force_download,
                sha256=expected_sha256,
            )

            d_model = 768
            nhead = 12
            dim_feedforward = 3072
            out_channels = 527

            kernel_size = 16
            _stride = 10
            _n_bins = 128
            _n_frames = 1024

            batch_first = True
            norm_first = True
            layer_norm_eps = 1e-6
            num_layers = 12
            activation = nn.GELU()
            norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

            insert_cls_token = True
            insert_dist_token = True

            embedding = PositionalPatchEmbedding(
                d_model,
                kernel_size=kernel_size,
                stride=_stride,
                insert_cls_token=insert_cls_token,
                insert_dist_token=insert_dist_token,
                n_bins=_n_bins,
                n_frames=_n_frames,
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model,
                nhead,
                dim_feedforward,
                activation=activation,
                batch_first=batch_first,
                norm_first=norm_first,
                layer_norm_eps=layer_norm_eps,
            )
            backbone = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=norm)
            _aggregator = HeadTokensAggregator(
                insert_cls_token=insert_cls_token,
                insert_dist_token=insert_dist_token,
            )
            _head = MLPHead(
                d_model,
                out_channels,
            )

            model = ModifiedAudioSpectrogramTransformer(
                embedding,
                backbone,
                aggregator=_aggregator,
                head=_head,
            )

            state_dict = torch.load(
                path,
                map_location=lambda storage, loc: storage,
            )
            model.load_state_dict(state_dict)

            # override aggregator and head if necessary
            if aggregator is not None:
                model.aggregator = aggregator

            if head is not None:
                model.head = head

            # update patch embedding if necessary
            model.embedding = _align_patch_embedding(
                model.embedding, stride=stride, n_bins=n_bins, n_frames=n_frames
            )
        else:
            raise ValueError(
                "Only ast-base-stride10 is supported as pretrained model, "
                f"but {pretrained_model_name_or_path} is given."
            )

        return model

    def forward(
        self,
        input: torch.Tensor,
        length: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """Forward pass of ModifiedAudioSpectrogramTransformer.

        Args:
            input (torch.Tensor): Spectrogram of shape (batch_size, n_bins, n_frames).
            length (torch.LongTensor, optional): Length of input of shape (batch_size,).

        Returns:
            torch.Tensor: Estimated patches. The shape is one of
                - (batch_size, height * width + num_head_tokens, embedding_dim).
                - (batch_size, height * width + num_head_tokens, out_channels).
                - (batch_size, embedding_dim).
                - (batch_size, out_channels).

        """
        input = self.pad_by_length(input, length=length)
        x = self.embedding(input)
        padding_mask = self.compute_padding_mask(input, length=length)
        output = self.transformer_forward(x, padding_mask=padding_mask)

        if self.aggregator is not None:
            output = self.aggregator(output, padding_mask=padding_mask)

        if self.head is not None:
            output = self.head(output)

        return output
    
def _align_patch_embedding(
    orig_patch_embedding: PositionalPatchEmbedding,
    stride: Optional[_size_2_t] = None,
    n_bins: Optional[int] = None,
    n_frames: Optional[int] = None,
) -> PositionalPatchEmbedding:
    pretrained_embedding_dim = orig_patch_embedding.embedding_dim
    pretrained_kernel_size = orig_patch_embedding.kernel_size
    pretrained_stride = orig_patch_embedding.stride
    pretrained_insert_cls_token = orig_patch_embedding.insert_cls_token
    pretrained_insert_dist_token = orig_patch_embedding.insert_dist_token
    pretrained_n_bins = orig_patch_embedding.n_bins
    pretrained_n_frames = orig_patch_embedding.n_frames
    pretrained_conv2d = orig_patch_embedding.conv2d
    pretrained_positional_embedding = orig_patch_embedding.positional_embedding
    pretrained_cls_token = orig_patch_embedding.cls_token
    pretrained_dist_token = orig_patch_embedding.dist_token

    if stride is None:
        stride = pretrained_stride

    if n_bins is None:
        n_bins = pretrained_n_bins

    if n_frames is None:
        n_frames = pretrained_n_frames

    new_patch_embedding = PositionalPatchEmbedding(
        pretrained_embedding_dim,
        kernel_size=pretrained_kernel_size,
        stride=stride,
        insert_cls_token=pretrained_insert_cls_token,
        insert_dist_token=pretrained_insert_dist_token,
        n_bins=n_bins,
        n_frames=n_frames,
    )

    conv2d_state_dict = copy.deepcopy(pretrained_conv2d.state_dict())
    new_patch_embedding.conv2d.load_state_dict(conv2d_state_dict)

    pretrained_positional_embedding = new_patch_embedding.resample_positional_embedding(
        pretrained_positional_embedding, n_bins, n_frames
    )
    new_patch_embedding.positional_embedding.data.copy_(pretrained_positional_embedding)

    if pretrained_insert_cls_token:
        new_patch_embedding.cls_token.data.copy_(pretrained_cls_token)

    if pretrained_insert_dist_token:
        new_patch_embedding.dist_token.data.copy_(pretrained_dist_token)

    return new_patch_embedding