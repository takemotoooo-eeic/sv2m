import math
from abc import abstractmethod
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Aggregator(nn.Module):
    """Base class of module to aggregate features."""

    @abstractmethod
    def forward(
        self,
        input: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """Forward pass of Aggregator.

        Args:
            input (torch.Tensor): Sequence of shape (batch_size, length, embedding_dim).
            mask (torch.BoolTensor, optional): Validity mask of shape (batch_size, length).
                True indicates valid positions.

        Returns:
            torch.Tensor: Aggregated feature of shape (batch_size, embedding_dim).

        """
        pass


class AverageAggregator(Aggregator):
    """Module of aggregation by average operation.

    Args:
        insert_cls_token (bool): Given sequence is assumed to contain [CLS] token.
        insert_dist_token (bool): Given sequence is assumed to contain [DIST] token.

    """

    def __init__(
        self,
        insert_cls_token: bool = True,
        insert_dist_token: bool = True,
    ) -> None:
        super().__init__()

        self.insert_cls_token = insert_cls_token
        self.insert_dist_token = insert_dist_token

    def forward(
        self,
        input: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """Forward pass of AverageAggregator.

        Args:
            input (torch.Tensor): Sequence of shape (batch_size, length, embedding_dim).
            mask (torch.BoolTensor, optional): Mask of shape (batch_size, length).

        Returns:
            torch.Tensor: Aggregated feature of shape (batch_size, embedding_dim).

        """
        num_head_tokens = 0

        if self.insert_cls_token:
            num_head_tokens += 1

        if self.insert_dist_token:
            num_head_tokens += 1

        _, x = torch.split(input, [num_head_tokens, input.size(-2) - num_head_tokens], dim=-2)

        if mask is None:
            batch_size, length, _ = x.size()
            mask = torch.full(
                (batch_size, length),
                fill_value=True,
                dtype=torch.bool,
                device=x.device,
            )
        else:
            _, mask = torch.split(mask, [num_head_tokens, mask.size(-1) - num_head_tokens], dim=-1)
            mask = mask.to(torch.bool)

        x = x.masked_fill(~mask.unsqueeze(dim=-1), 0)
        valid_count = mask.to(torch.long).sum(dim=-1, keepdim=True).clamp_min(1)
        output = x.sum(dim=-2) / valid_count

        return output


class HeadTokensAggregator(Aggregator):
    """Module of aggregation by extraction of head tokens.

    Args:
        insert_cls_token (bool): Given sequence is assumed to contain [CLS] token.
        insert_dist_token (bool): Given sequence is assumed to contain [DIST] token.

    """

    def __init__(
        self,
        insert_cls_token: bool = True,
        insert_dist_token: bool = True,
    ) -> None:
        super().__init__()

        if not insert_cls_token and not insert_dist_token:
            raise ValueError("At least one of insert_cls_token and insert_dist_token should be True.")

        self.insert_cls_token = insert_cls_token
        self.insert_dist_token = insert_dist_token

    def forward(
        self,
        input: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """Forward pass of HeadTokensAggregator.

        Args:
            input (torch.Tensor): Sequence of shape (batch_size, length, embedding_dim).
            padding_mask (torch.BoolTensor, optional): Padding mask of shape (batch_size, length).

        Returns:
            torch.Tensor: Aggregated feature of shape (batch_size, embedding_dim).

        .. note::

            mask is ignored.

        """
        num_head_tokens = 0

        if self.insert_cls_token:
            num_head_tokens += 1

        if self.insert_dist_token:
            num_head_tokens += 1

        head_tokens, _ = torch.split(input, [num_head_tokens, input.size(-2) - num_head_tokens], dim=-2)
        output = torch.mean(head_tokens, dim=-2)

        return output


class CrossAttention(nn.Module):
    def __init__(self, dim_input: int, num_heads: int = 1):
        super(CrossAttention, self).__init__()
        self.dim_input = dim_input
        self.num_heads = num_heads
        assert self.dim_input % self.num_heads == 0
        self.head_dim = self.dim_input // self.num_heads

        self.q_proj = nn.Linear(self.dim_input, self.dim_input)
        self.k_proj = nn.Linear(self.dim_input, self.dim_input)
        self.v_proj = nn.Linear(self.dim_input, self.dim_input)
        self.out_proj = nn.Linear(self.dim_input, self.dim_input)

    def forward(
        self,
        video_embeds: torch.Tensor,
        music_features: torch.Tensor,
        music_mask: Optional[torch.Tensor] = None,
    ):
        """
        Input
            video_embeds: (batch_size, embed_dim)
            music_features: (batch_size, seq_len, embed_dim)
            music_mask: (batch_size, seq_len)
        Output
            o: (video_batch_size, music_batch_size, embed_dim)
        """
        # video_batch_size x embed_dim
        q = self.q_proj(video_embeds)
        q = rearrange(q, "v (h d) -> h v d", h=self.num_heads, d=self.head_dim)  # (num_heads, video_batch_size, head_dim)

        k = self.k_proj(music_features)  # (music_batch_size, seq_len, embed_dim)
        k = rearrange(k, "m f (h d) -> m h f d", h=self.num_heads, d=self.head_dim)  # (music_batch_size, num_heads, seq_len, head_dim)

        v = self.v_proj(music_features)  # (music_batch_size, seq_len, embed_dim)
        v = rearrange(v, "m f (h d) -> m h f d", h=self.num_heads, d=self.head_dim)  # (music_batch_size, num_heads, seq_len, head_dim)

        # The dot product attention gives relevancy weights from a video to each segment.
        attention_logits = torch.einsum("hvd,mhfd->vmhf", q, k)  # (video_batch_size, music_batch_size, num_heads, seq_len)
        attention_logits = attention_logits / math.sqrt(self.head_dim)

        # mask the attention_logits
        if music_mask is not None:
            music_mask = music_mask[None, :, None, :]  # (1, music_batch_size, 1, seq_len)
            attention_logits = attention_logits.masked_fill(music_mask == 0, float("-inf"))
        attention_weights = F.softmax(attention_logits, dim=-1)  # (video_batch_size, music_batch_size, num_heads, seq_len)

        attention = torch.einsum("vmhf,mhfd->vmhd", attention_weights, v)  # (video_batch_size, music_batch_size, num_heads, head_dim)
        attention = rearrange(attention, "v m h d -> v m (h d)")  # (video_batch_size, music_batch_size, num_heads*head_dim)
        o = self.out_proj(attention)  # (video_batch_size, music_batch_size, embed_dim)
        return o


class XPoolAggregator(nn.Module):
    def __init__(
        self,
        dim_input: int,
        num_heads: int = 1,
        dropout: float = 0.3,
        insert_cls_token: bool = True,
        insert_dist_token: bool = True,
    ):
        super(XPoolAggregator, self).__init__()

        self.cross_attn = CrossAttention(dim_input, num_heads)
        self.insert_cls_token = insert_cls_token
        self.insert_dist_token = insert_dist_token

        self.linear_proj = nn.Linear(dim_input, dim_input)

        self.layer_norm1 = nn.LayerNorm(dim_input)
        self.layer_norm2 = nn.LayerNorm(dim_input)
        self.layer_norm3 = nn.LayerNorm(dim_input)
        self.dropout = nn.Dropout(dropout)
        self._init_parameters()

    def _init_parameters(self):
        for name, param in self.named_parameters():
            if "linear" in name or "proj" in name:
                if "weight" in name:
                    nn.init.eye_(param)
                elif "bias" in name:
                    param.data.fill_(0.0)

    def forward(
        self,
        video_embeds: torch.Tensor,
        music_features: torch.Tensor,
        music_masks: Optional[torch.Tensor] = None,
    ):
        """
        Input
            video_embeds: (batch_size, embed_dim)
            music_features: (batch_size, seq_len, embed_dim)
            music_masks: (batch_size, seq_len)
        Output
            out: (video_batch_size, music_batch_size, embed_dim)
        """
        num_head_tokens = 0

        if self.insert_cls_token:
            num_head_tokens += 1

        if self.insert_dist_token:
            num_head_tokens += 1

        _, music_features = torch.split(music_features, [num_head_tokens, music_features.size(-2) - num_head_tokens], dim=-2)
        video_embeds = self.layer_norm1(video_embeds)  # [video_batch_size, embed_dim]
        music_features = self.layer_norm1(music_features)  # [music_batch_size, seq_len, embed_dim]

        # video_batch_size x music_batch_size x embed_dim
        attn_out = self.cross_attn(video_embeds, music_features, music_masks)
        attn_out = self.layer_norm2(attn_out)

        linear_out = self.linear_proj(attn_out)
        out = attn_out + self.dropout(linear_out)
        out = self.layer_norm3(out)

        return out
