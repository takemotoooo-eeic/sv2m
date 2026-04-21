"""
Contrastive loss functions for cross-modal learning.

This module implements various contrastive loss functions used in cross-modal learning,
particularly for music-text contrastive learning in MuLan.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from ..distributed import is_distributed_mode
from ..modules.aggregater import XPoolAggregator
from .distributed import SyncFunction


class _CrossModalContrastiveLoss(nn.Module, ABC):
    """Base class for cross-modal contrastive loss functions.

    This abstract base class provides common functionality for contrastive learning losses:
    - Temperature parameter management (learnable log_temperature)
    - DDP gradient synchronization for log_temperature
    - Input validation (batch size checking)
    - DDP-aware embedding gathering

    Args:
        temperature (float): Initial temperature parameter for scaling the logits.
            This becomes a learnable parameter stored as log_temperature.
        reduction (str): Specifies the reduction to apply to the output.
    """

    def __init__(
        self,
        video_aggregators: list[nn.Module],
        music_aggregators: list[nn.Module],
        distribution_loss: Optional[nn.Module] = None,
        temperature: float = 0.1,
        min_temperature: float = 0.01,
        delete_duplicate: bool = False,
        reduction: str = "mean",
    ) -> None:
        super().__init__()

        self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
        self.min_temperature = min_temperature
        self.video_aggregators = video_aggregators
        self.music_aggregators = music_aggregators
        self.distribution_loss = distribution_loss
        self.reduction = reduction
        self.delete_duplicate = delete_duplicate

        # Register hook to synchronize log_temperature gradient in DDP mode
        # This hook works correctly whether the criterion is wrapped with DDP or not:
        # - If criterion is NOT wrapped: hook manually synchronizes the gradient
        # - If criterion IS wrapped with DDP: DDP synchronizes first, then hook
        #   re-synchronizes (redundant but harmless, as averaging twice gives same result)
        def _sync_log_temperature_grad(grad):
            """Synchronize log_temperature gradient across all ranks in DDP mode.

            In DDP training, each rank computes gradients on different data batches.
            This hook ensures log_temperature gradients are averaged across all ranks.

            Note: If the criterion is wrapped with DDP, DDP will also synchronize
            this gradient. The double synchronization is harmless as averaging an
            already-averaged gradient produces the same result.
            """
            if is_distributed_mode() and grad is not None:
                # Average gradients across all ranks
                dist.all_reduce(grad, op=dist.ReduceOp.SUM)
                grad.div_(dist.get_world_size())

            return grad

        self.log_temperature.register_hook(_sync_log_temperature_grad)

    @property
    def temperature(self) -> torch.Tensor:
        """Get temperature from log_temperature parameter."""
        return torch.clamp(torch.exp(self.log_temperature), min=self.min_temperature)

    def _validate_and_gather_inputs(
        self,
        music_features: torch.Tensor,
        music_masks: torch.Tensor,
        music_span_masks: torch.Tensor,
        video_features: torch.Tensor,
        video_masks: torch.Tensor,
        music_ids: list[str],
        spans_target: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[list[str]],
        int,
        bool,
        torch.device,
    ]:
        """Validate inputs and gather features/masks/ids across ranks if in DDP mode.

        Args:
            music_features: Music features (local_batch_size, ...)
            music_masks: Music masks (local_batch_size, ...)
            video_features: Video features (local_batch_size, ...)
            video_masks: Video masks (local_batch_size, ...)

        Returns:
            Tuple of (global_music_features, global_music_masks, global_video_features,
                      global_video_masks, global_music_ids, local_batch_size, is_distributed, device)
        """
        batch_size_music = music_features.size(0)
        batch_size_video = video_features.size(0)

        if batch_size_music != batch_size_video:
            raise RuntimeError(f"Batch sizes must match: music={batch_size_music}, video={batch_size_video}")
        if music_masks.size(0) != batch_size_music:
            raise RuntimeError(f"Music feature/mask batch sizes must match: features={batch_size_music}, masks={music_masks.size(0)}")
        if video_masks.size(0) != batch_size_video:
            raise RuntimeError(f"Video feature/mask batch sizes must match: features={batch_size_video}, masks={video_masks.size(0)}")

        local_batch_size = batch_size_music
        is_distributed = is_distributed_mode()

        if local_batch_size < 2 and not is_distributed:
            raise ValueError(f"Batch size must be at least 2 for contrastive learning, got {local_batch_size}")

        device = music_features.device

        if is_distributed:
            global_music_features = SyncFunction.apply(music_features, True)
            global_music_masks = SyncFunction.apply(music_masks, True)
            global_video_features = SyncFunction.apply(video_features, True)
            global_video_masks = SyncFunction.apply(video_masks, True)
            global_music_span_masks = SyncFunction.apply(music_span_masks, True)
            global_spans_target = SyncFunction.apply(spans_target, True)
            
            gathered_music_ids: list[Optional[list[str]]] = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered_music_ids, list(music_ids))
            global_music_ids = [music_id for ids in gathered_music_ids if ids is not None for music_id in ids]
        else:
            global_music_features = music_features
            global_music_masks = music_masks
            global_video_features = video_features
            global_video_masks = video_masks
            global_music_ids = music_ids
            global_music_span_masks = music_span_masks
            global_spans_target = spans_target

        return (
            global_music_features,
            global_music_masks,
            global_music_span_masks,
            global_video_features,
            global_video_masks,
            global_music_ids,
            global_spans_target,
            local_batch_size,
            is_distributed,
            device,
        )

    def _build_duplicate_mask(
        self,
        candidate_ids: list[str],
        row_start: int,
        row_end: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Build mask to exclude negatives sharing the same source id.

        The positive pair on the diagonal is kept unmasked.
        """
        if not self.delete_duplicate:
            return None
        if candidate_ids is None:
            return None

        mask = torch.zeros((row_end - row_start, len(candidate_ids)), dtype=torch.bool, device=device)
        has_duplicate = False
        for local_row, global_row in enumerate(range(row_start, row_end)):
            row_id = candidate_ids[global_row]
            for col, col_id in enumerate(candidate_ids):
                if col != global_row and col_id == row_id:
                    mask[local_row, col] = True
                    has_duplicate = True
        return mask if has_duplicate else None

    @abstractmethod
    def forward(
        self,
        music_features: torch.Tensor,
        music_masks: torch.Tensor,
        music_ids: Optional[list[str]],
        video_features: torch.Tensor,
        video_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Compute contrastive loss between music and video features.

        Args:
            music_features: L2-normalized music features (batch_size, embedding_dim)
            video_features: L2-normalized video features (batch_size, embedding_dim)
            music_masks: Music masks (batch_size, num_frames)
            video_masks: Video masks (batch_size, num_frames)
            music_ids: Music ids (batch_size)

        Returns:
            Contrastive loss value
        """
        pass


class CrossModalInfoNCELoss(_CrossModalContrastiveLoss):
    """Cross-modal InfoNCE loss for contrastive learning.

    InfoNCE (Information Noise Contrastive Estimation) loss is based on mutual information
    estimation. It explicitly separates positive and negative samples.

    This loss function supports Distributed Data Parallel (DDP) training. When DDP is enabled,
    embeddings from all processes are gathered to compute the similarity matrix, increasing
    the effective batch size and number of negative samples.

    Args:
        temperature (float): Initial temperature parameter for scaling the logits.
            This becomes a learnable parameter stored as log_temperature.
        video_aggregators: list of video aggregators
        music_aggregators: list of music aggregators
        min_temperature: minimum temperature
        delete_duplicate: whether to delete duplicate samples
        reduction (str): Specifies the reduction to apply to the output.

    DDP Usage:
        This loss function automatically supports both common DDP usage patterns:

        Pattern 1: Wrap only the model with DDP (recommended)
            Apply `nn.parallel.DistributedDataParallel` only to the model.
            The loss function will automatically gather embeddings across ranks.

            Example:
                >>> import torch
                >>> import torch.distributed as dist
                >>> from torch.nn.parallel import DistributedDataParallel as DDP
                >>>
                >>> # Initialize process group
                >>> dist.init_process_group(...)
                >>>
                >>> # Wrap only the model with DDP
                >>> model = YourModel().cuda()
                >>> model = DDP(model, device_ids=[local_rank])
                >>>
                >>> # Create loss function (no DDP wrapper needed)
                >>> criterion = CrossModalInfoNCELoss(temperature=0.1).cuda()
                >>>
                >>> # Forward pass
                >>> music_embedding, text_embedding = model(music_input, text_input)
                >>> loss = criterion(music_embedding, text_embedding)
                >>>
                >>> # Backward pass
                >>> loss.backward()

        Pattern 2: Wrap both model and criterion together with DDP (also supported)
            Wrap both model and criterion together in a single module with DDP.

            Example:
                >>> import torch
                >>> import torch.distributed as dist
                >>> from torch.nn.parallel import DistributedDataParallel as DDP
                >>>
                >>> # Initialize process group
                >>> dist.init_process_group(...)
                >>>
                >>> # Create a module containing both model and criterion
                >>> class ModelWithLoss(nn.Module):
                >>>     def __init__(self):
                >>>         super().__init__()
                >>>         self.model = YourModel()
                >>>         self.criterion = CrossModalInfoNCELoss(temperature=0.1)
                >>>     def forward(self, music_input, text_input):
                >>>         music_embedding, text_embedding = self.model(music_input, text_input)
                >>>         return self.criterion(music_embedding, text_embedding)
                >>>
                >>> # Wrap the combined module with DDP
                >>> model_with_loss = ModelWithLoss().cuda()
                >>> model_with_loss = DDP(model_with_loss, device_ids=[local_rank])
                >>>
                >>> # Forward pass
                >>> loss = model_with_loss(music_input, text_input)
                >>>
                >>> # Backward pass
                >>> loss.backward()

    Note:
        - Both usage patterns produce identical results
        - Pattern 1 (model-only) is simpler and recommended for most use cases
        - In DDP mode, the effective batch size becomes local_batch_size * world_size
        - All gradients (model parameters and log_temperature) are automatically synchronized
    """

    def __init__(
        self,
        video_aggregators: list[nn.Module],
        music_aggregators: list[nn.Module],
        temperature: float = 0.1,
        min_temperature: float = 0.01,
        delete_duplicate: bool = False,
        distribution_loss: Optional[nn.Module] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__(
            video_aggregators=video_aggregators,
            music_aggregators=music_aggregators,
            temperature=temperature,
            min_temperature=min_temperature,
            delete_duplicate=delete_duplicate,
            reduction=reduction,
        )
        self.distribution_loss = distribution_loss
        self.last_contrastive_loss: float = 0.0
        self.last_distribution_loss: float = 0.0
        self.last_total_loss: float = 0.0

    def forward(
        self,
        music_features: torch.Tensor,
        music_masks: torch.Tensor,
        music_span_masks: Optional[torch.Tensor],
        spans_target: Optional[torch.Tensor],
        music_ids: Optional[list[str]],
        video_features: torch.Tensor,
        video_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute InfoNCE loss between music and video features.

        Args:
            music_features (torch.Tensor): L2-normalized music features of
                shape (batch_size, embedding_dim).

            video_features (torch.Tensor): L2-normalized video features of
                shape (batch_size, embedding_dim).

        Returns:
            torch.Tensor: InfoNCE loss.
        """
        if len(self.video_aggregators) != len(self.music_aggregators):
            raise ValueError("video_aggregators and music_aggregators must have same length")
        if len(self.video_aggregators) == 0:
            raise ValueError("at least one aggregator is required")

        (
            global_music_features,
            global_music_masks,
            global_music_span_masks,
            global_video_features,
            global_video_masks,
            global_music_ids,
            global_spans_target,
            local_batch_size,
            is_distributed,
            device,
        ) = self._validate_and_gather_inputs(music_features, music_masks, music_span_masks, video_features, video_masks, music_ids, spans_target)

        similarity_matrix_sum = None
        attention_weights = None

        for video_aggregator, music_aggregator in zip(self.video_aggregators, self.music_aggregators):
            if isinstance(video_aggregator, XPoolAggregator):
                raise ValueError("video_aggregator cannot be XPoolAggregator")
            video_embeddings = video_aggregator(global_video_features, global_video_masks)  # [batch_size, embed_dim]

            if isinstance(music_aggregator, XPoolAggregator):
                music_embeddings, attention_weights = music_aggregator(video_embeddings, global_music_features, global_music_masks, global_music_span_masks)  # [video_batch_size, music_batch_size, embed_dim]
            else:
                music_embeddings = music_aggregator(global_music_features, global_music_masks, global_music_span_masks)  # [batch_size, embed_dim]
            
            if isinstance(music_aggregator, XPoolAggregator):
                similarity_matrix = torch.einsum("vmd,vd->vm", music_embeddings, video_embeddings)  # [video_batch_size, music_batch_size]
            else:
                similarity_matrix = torch.matmul(video_embeddings, music_embeddings.T)  # [video_batch_size, music_batch_size]
            
            if similarity_matrix_sum is None:
                similarity_matrix_sum = similarity_matrix
            else:
                similarity_matrix_sum = similarity_matrix_sum + similarity_matrix
        similarity_matrix = similarity_matrix_sum / self.temperature
        

        if is_distributed:
            # In DDP mode, each rank computes loss only on its local batch portion
            rank = dist.get_rank()
            labels = torch.arange(rank * local_batch_size, (rank + 1) * local_batch_size, device=device)
            start_index = rank * local_batch_size
            end_index = (rank + 1) * local_batch_size
            local_similarity_v2m = similarity_matrix[start_index:end_index]
            local_similarity_m2v = similarity_matrix.T[start_index:end_index]
            duplicate_mask = self._build_duplicate_mask(global_music_ids, start_index, end_index, device)
        else:
            labels = torch.arange(local_batch_size, device=device)
            local_similarity_v2m = similarity_matrix
            local_similarity_m2v = similarity_matrix.T
            duplicate_mask = self._build_duplicate_mask(global_music_ids, 0, local_batch_size, device)

        if duplicate_mask is not None:
            local_similarity_v2m = local_similarity_v2m.masked_fill(duplicate_mask, float("-inf"))
            local_similarity_m2v = local_similarity_m2v.masked_fill(duplicate_mask.T, float("-inf"))

        # Compute bidirectional cross-entropy loss
        loss_v2m = F.cross_entropy(local_similarity_v2m, labels, reduction=self.reduction)
        loss_m2v = F.cross_entropy(local_similarity_m2v, labels, reduction=self.reduction)
        contrastive_loss = (loss_v2m + loss_m2v) / 2

        if self.distribution_loss is not None:
            if attention_weights is None:
                raise ValueError("distribution_loss requires attention_weights(XPoolAggregator) from music_aggregator")
            if is_distributed:
                spans_target_for_loss = global_spans_target[start_index:end_index]
                row_offset = start_index
            else:
                spans_target_for_loss = global_spans_target
                row_offset = 0

            if is_distributed:
                attention_weights_for_loss = attention_weights[start_index:end_index]
            else:
                attention_weights_for_loss = attention_weights

            distribution_loss = self.distribution_loss(
                attention_weights=attention_weights_for_loss,
                music_masks=global_music_masks,
                span_target=spans_target_for_loss,
                positive_col_offset=row_offset,
            )
        else:
            distribution_loss = torch.tensor(0.0, device=device)

        loss = contrastive_loss + distribution_loss

        self.last_contrastive_loss = float(contrastive_loss.detach().item())
        self.last_distribution_loss = float(distribution_loss.detach().item())
        self.last_total_loss = float(loss.detach().item())
        return loss, attention_weights