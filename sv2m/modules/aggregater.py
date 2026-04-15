from abc import abstractmethod
from typing import Optional
import torch
import torch.nn as nn



class Aggregator(nn.Module):
    """Base class of module to aggregate features."""

    @abstractmethod
    def forward(
        self,
        input: torch.Tensor,
        padding_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """Forward pass of Aggregator.

        Args:
            input (torch.Tensor): Sequence of shape (batch_size, length, embedding_dim).
            padding_mask (torch.BoolTensor, optional): Padding mask of shape (batch_size, length).

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
        padding_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """Forward pass of AverageAggregator.

        Args:
            input (torch.Tensor): Sequence of shape (batch_size, length, embedding_dim).
            padding_mask (torch.BoolTensor, optional): Padding mask of shape (batch_size, length).

        Returns:
            torch.Tensor: Aggregated feature of shape (batch_size, embedding_dim).

        """
        num_head_tokens = 0

        if self.insert_cls_token:
            num_head_tokens += 1

        if self.insert_dist_token:
            num_head_tokens += 1

        _, x = torch.split(input, [num_head_tokens, input.size(-2) - num_head_tokens], dim=-2)

        if padding_mask is None:
            batch_size, length, _ = x.size()
            padding_mask = torch.full(
                (batch_size, length),
                fill_value=False,
                dtype=torch.bool,
                device=x.device,
            )
        else:
            _, padding_mask = torch.split(
                padding_mask, [num_head_tokens, padding_mask.size(-1) - num_head_tokens], dim=-1
            )

        x = x.masked_fill(padding_mask.unsqueeze(dim=-1), 0)
        non_padding_mask = torch.logical_not(padding_mask)
        non_padding_mask = non_padding_mask.to(torch.long)
        output = x.sum(dim=-2) / non_padding_mask.sum(dim=-1, keepdim=True)

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
            raise ValueError(
                "At least one of insert_cls_token and insert_dist_token should be True."
            )

        self.insert_cls_token = insert_cls_token
        self.insert_dist_token = insert_dist_token

    def forward(
        self,
        input: torch.Tensor,
        padding_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """Forward pass of HeadTokensAggregator.

        Args:
            input (torch.Tensor): Sequence of shape (batch_size, length, embedding_dim).
            padding_mask (torch.BoolTensor, optional): Padding mask of shape (batch_size, length).

        Returns:
            torch.Tensor: Aggregated feature of shape (batch_size, embedding_dim).

        .. note::

            padding_mask is ignored.

        """
        num_head_tokens = 0

        if self.insert_cls_token:
            num_head_tokens += 1

        if self.insert_dist_token:
            num_head_tokens += 1

        head_tokens, _ = torch.split(
            input, [num_head_tokens, input.size(-2) - num_head_tokens], dim=-2
        )
        output = torch.mean(head_tokens, dim=-2)

        return output