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
from ..modules.aggregater import LateInteractionAggregator, XPoolAggregator
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
        video_aggregators: Optional[list[nn.Module]],
        music_aggregators: Optional[list[nn.Module]],
        distribution_loss: Optional[nn.Module] = None,
        temperature: float = 0.1,
        min_temperature: float = 0.01,
        delete_duplicate: bool = False,
        reduction: str = "mean",
    ) -> None:
        super().__init__()

        self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
        self.min_temperature = min_temperature
        self.video_aggregators = nn.ModuleList(video_aggregators or [])
        self.music_aggregators = nn.ModuleList(music_aggregators or [])
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

    def build_duplicate_mask(
        self,
        candidate_ids: list[str],
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Build global square mask to exclude negatives sharing the same source id.

        The returned mask has shape ``[global_batch, global_batch]`` and can be
        sliced/transposed in the same way as the similarity matrix.
        The positive pair on the diagonal is kept unmasked.
        """
        if not self.delete_duplicate:
            return None
        if candidate_ids is None:
            return None

        global_batch_size = len(candidate_ids)
        mask = torch.zeros((global_batch_size, global_batch_size), dtype=torch.bool, device=device)
        has_duplicate = False
        for row in range(global_batch_size):
            row_id = candidate_ids[row]
            for col, col_id in enumerate(candidate_ids):
                if col != row and col_id == row_id:
                    mask[row, col] = True
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
        self._validate_aggregators(video_aggregators, music_aggregators, distribution_loss)

    def _validate_aggregators(self, video_aggregators: list[nn.Module], music_aggregators: list[nn.Module], distribution_loss: Optional[nn.Module]) -> None:
        """Validate video and music aggregators for compatibility."""
        if len(video_aggregators) != len(music_aggregators):
            raise ValueError("video_aggregators and music_aggregators must have same length")
        if len(video_aggregators) == 0 or len(music_aggregators) == 0:
            raise ValueError("at least one aggregator is required")
        for video_aggregator, music_aggregator in zip(video_aggregators, music_aggregators):
            if isinstance(video_aggregator, XPoolAggregator):
                raise ValueError("video_aggregator cannot be XPoolAggregator")
            if isinstance(music_aggregator, LateInteractionAggregator) and not isinstance(video_aggregator, LateInteractionAggregator):
                raise ValueError("LateInteractionAggregator must be used for both video_aggregator and music_aggregator")
            if not isinstance(music_aggregator, LateInteractionAggregator) and isinstance(video_aggregator, LateInteractionAggregator):
                raise ValueError("LateInteractionAggregator must be used for both video_aggregator and music_aggregator")
            
        if distribution_loss is not None:
            has_xpool = any(isinstance(agg, XPoolAggregator) for agg in music_aggregators)
            if not has_xpool:
                raise ValueError("distribution_loss requires at least one XPoolAggregator in music_aggregators")

    def compute_similarity_matrixs(
        self,
        video_features: torch.Tensor,
        music_features: torch.Tensor,
        video_masks: torch.Tensor,
        music_masks: torch.Tensor,
        music_span_masks: Optional[torch.Tensor],
        chunk_size: Optional[int] = None,
    ) -> Tuple[list[torch.Tensor], Optional[torch.Tensor]]:
        """Compute per-aggregator similarity matrices from features and masks.

        Args:
            video_features: Video features with shape [Bv, ...].
            music_features: Music features with shape [Bm, ...].
            video_masks: Video masks with shape [Bv, ...].
            music_masks: Music masks with shape [Bm, ...].
            music_span_masks: Optional music span masks with shape [Bm, ...].
            chunk_size: Optional chunk size used for late-interaction aggregators.

        Returns:
            Tuple of (similarity_matrixs, attention_weights). ``attention_weights``
            is returned when an ``XPoolAggregator`` is used.
        """
        if len(self.video_aggregators) != len(self.music_aggregators):
            raise ValueError("video_aggregators and music_aggregators must have same length")
        if len(self.video_aggregators) == 0:
            raise ValueError("at least one aggregator is required")

        similarity_matrixs: list[torch.Tensor] = []
        attention_weights = None

        for video_aggregator, music_aggregator in zip(self.video_aggregators, self.music_aggregators):
            if isinstance(video_aggregator, LateInteractionAggregator) and isinstance(music_aggregator, LateInteractionAggregator):
                if chunk_size is None:
                    similarity_matrix = self.compute_late_interaction_similarity_matrix(
                        aggregator=video_aggregator,
                        video_features=video_features,
                        music_features=music_features,
                        video_masks=video_masks,
                        music_masks=music_masks,
                        music_span_masks=music_span_masks,
                    )
                else:
                    chunked_similarity_matrixs = []
                    for start in range(0, video_features.size(0), chunk_size):
                        end = min(start + chunk_size, video_features.size(0))
                        chunked_similarity_matrixs.append(
                            self.compute_late_interaction_similarity_matrix(
                                aggregator=video_aggregator,
                                video_features=video_features[start:end],
                                music_features=music_features,
                                video_masks=video_masks[start:end],
                                music_masks=music_masks,
                                music_span_masks=music_span_masks,
                            )
                        )
                    similarity_matrix = torch.cat(chunked_similarity_matrixs, dim=0)

                similarity_matrixs.append(similarity_matrix)
                continue

            if isinstance(music_aggregator, XPoolAggregator):
                # XPoolAggregator requires video_embeddings, so compute them first
                if chunk_size is None:
                    video_embeddings = video_aggregator(video_features, video_masks)
                    music_embeddings, attention_weights = music_aggregator(
                        video_embeddings,
                        music_features,
                        music_masks,
                        music_span_masks,
                    )
                    similarity_matrix = torch.einsum("vmd,vd->vm", music_embeddings, video_embeddings)
                else:
                    chunked_similarity_matrixs = []
                    chunked_attention_weights = []
                    for start in range(0, video_features.size(0), chunk_size):
                        end = min(start + chunk_size, video_features.size(0))
                        chunk_video_embeddings = video_aggregator(video_features[start:end], video_masks[start:end])
                        chunk_music_embeddings, chunk_attention_weights = music_aggregator(
                            chunk_video_embeddings,
                            music_features,
                            music_masks,
                            music_span_masks,
                        )
                        chunk_similarity_matrix = torch.einsum("vmd,vd->vm", chunk_music_embeddings, chunk_video_embeddings)
                        chunked_similarity_matrixs.append(chunk_similarity_matrix)
                        chunked_attention_weights.append(chunk_attention_weights)
                    similarity_matrix = torch.cat(chunked_similarity_matrixs, dim=0)
                    attention_weights = torch.cat(chunked_attention_weights, dim=0)
            else:
                video_embeddings = video_aggregator(video_features, video_masks)
                music_embeddings = music_aggregator(music_features, music_masks, music_span_masks)
                similarity_matrix = torch.matmul(video_embeddings, music_embeddings.T)

            similarity_matrixs.append(similarity_matrix)

        return similarity_matrixs, attention_weights

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

        similarity_matrixs, attention_weights = self.compute_similarity_matrixs(
            video_features=global_video_features,
            music_features=global_music_features,
            video_masks=global_video_masks,
            music_masks=global_music_masks,
            music_span_masks=global_music_span_masks,
        )
        similarity_matrix = torch.mean(torch.stack(similarity_matrixs), dim=0) / self.temperature
        

        if is_distributed:
            # In DDP mode, each rank computes loss only on its local batch portion
            rank = dist.get_rank()
            labels = torch.arange(rank * local_batch_size, (rank + 1) * local_batch_size, device=device)
            start_index = rank * local_batch_size
            end_index = (rank + 1) * local_batch_size
            local_similarity_v2m = similarity_matrix[start_index:end_index]
            local_similarity_m2v = similarity_matrix.T[start_index:end_index]
            duplicate_mask = self.build_duplicate_mask(global_music_ids, device)
        else:
            labels = torch.arange(local_batch_size, device=device)
            local_similarity_v2m = similarity_matrix
            local_similarity_m2v = similarity_matrix.T
            duplicate_mask = self.build_duplicate_mask(global_music_ids, device)

        if duplicate_mask is not None:
            if is_distributed:
                local_duplicate_mask_v2m = duplicate_mask[start_index:end_index]
                local_duplicate_mask_m2v = duplicate_mask.T[start_index:end_index]
            else:
                local_duplicate_mask_v2m = duplicate_mask
                local_duplicate_mask_m2v = duplicate_mask.T

            local_similarity_v2m = local_similarity_v2m.masked_fill(local_duplicate_mask_v2m, float("-inf"))
            local_similarity_m2v = local_similarity_m2v.masked_fill(local_duplicate_mask_m2v, float("-inf"))

        if attention_weights is not None and is_distributed:
            attention_weights = attention_weights[start_index:end_index]

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


            distribution_loss = self.distribution_loss(
                attention_weights=attention_weights,
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

    def compute_late_interaction_similarity_matrix(
        self,
        aggregator: LateInteractionAggregator,
        video_features: torch.Tensor,
        music_features: torch.Tensor,
        video_masks: Optional[torch.Tensor] = None,
        music_masks: Optional[torch.Tensor] = None,
        music_span_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute late-interaction similarity matrix.

        The score between a video v and music m is:

            s(v, m) = sum_i max_j sim(v_i, m_j)

        where i indexes video tokens and j indexes music tokens.

        Args:
            video_features (torch.Tensor):
                Video features with shape [B_v, T_v, D] (token-level) or [B_v, D] (pooled).
            music_features (torch.Tensor):
                Music features with shape [B_m, T_m, D] (token-level) or [B_m, D] (pooled).
            video_masks (Optional[torch.Tensor]):
                Valid-token mask for videos with shape [B_v, T_v]. True means valid.
            music_masks (Optional[torch.Tensor]):
                Valid-token mask for music with shape [B_m, T_m]. True means valid.
            

        Returns:
            torch.Tensor:
                Late-interaction similarity matrix with shape [B_v, B_m].
        """
        video_features = F.normalize(video_features, p=2, dim=-1)
        music_features = F.normalize(music_features, p=2, dim=-1)

        # Pairwise token similarity: [B_v, B_m, T_v, T_m]
        similarity = torch.einsum("avd,bmd->abvm", video_features, music_features)  # [B_v, B_m, T_v, T_m]

        music_masks = music_masks.to(dtype=torch.bool)
        if aggregator.use_span_mask and music_span_masks is not None:
            similarity = similarity.masked_fill(~music_span_masks[None, :, None, :], float("-inf"))
        else:
            similarity = similarity.masked_fill(~music_masks[None, :, None, :], float("-inf"))

        # For each video token, keep the best matching music token: [B_v, B_m, T_v]
        if aggregator.aggregation == "max":
            token_scores = similarity.max(dim=-1).values
        elif aggregator.aggregation == "log_sum":
            token_scores = torch.logsumexp(similarity / aggregator.aggregation_temperature, dim=-1) * aggregator.aggregation_temperature
        elif aggregator.aggregation == "top_k":
            top_k = min(aggregator.top_k, similarity.size(-1))
            token_scores = similarity.topk(k=top_k, dim=-1).values.mean(dim=-1)
        else:
            raise ValueError(f"Invalid aggregation method: {aggregator.aggregation}")

        video_masks = video_masks.to(dtype=torch.bool)
        token_scores = token_scores.masked_fill(~video_masks[:, None, :], 0.0)
        valid_video_counts = video_masks.sum(dim=-1, keepdim=True).clamp_min(1).to(token_scores.dtype)  # [B_v, 1]
        return token_scores.sum(dim=-1) / valid_video_counts  # [B_v, B_m]