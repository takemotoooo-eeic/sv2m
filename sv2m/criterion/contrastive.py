"""
Contrastive loss functions for cross-modal learning.

This module implements various contrastive loss functions used in cross-modal learning,
particularly for music-text contrastive learning in MuLan.
"""

from abc import ABC, abstractmethod

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from ..distributed import is_distributed_mode
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

    def __init__(self, temperature: float = 0.1, reduction: str = "mean") -> None:
        super().__init__()

        self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
        self.reduction = reduction

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
        return torch.exp(self.log_temperature)

    def _validate_and_gather_embeddings(
        self,
        music_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int, bool, torch.device]:
        """Validate inputs and gather embeddings across ranks if in DDP mode.

        Args:
            music_embeddings: L2-normalized music embeddings (local_batch_size, embedding_dim)
            text_embeddings: L2-normalized text embeddings (local_batch_size, embedding_dim)

        Returns:
            Tuple of (global_music_embeddings, global_text_embeddings, local_batch_size,
                     is_distributed, device)
        """
        batch_size_music = music_embeddings.size(0)
        batch_size_text = text_embeddings.size(0)

        if batch_size_music != batch_size_text:
            raise RuntimeError(
                f"Batch sizes must match: music={batch_size_music}, text={batch_size_text}"
            )

        local_batch_size = batch_size_music
        is_distributed = is_distributed_mode()

        if local_batch_size < 2 and not is_distributed:
            raise ValueError(
                f"Batch size must be at least 2 for contrastive learning, got {local_batch_size}"
            )

        device = music_embeddings.device

        if is_distributed:
            # Gather embeddings from all ranks to compute cross-rank similarities
            # for effective contrastive learning.
            #
            # We use sync_grad=True because in contrastive learning, each embedding
            # participates in loss computation across all ranks, so gradients must
            # be accumulated via all_reduce.
            #
            # SyncFunction handles embedding gradient synchronization, while the
            # autograd hook registered in __init__ handles log_temperature synchronization.
            global_music_embeddings = SyncFunction.apply(music_embeddings, True)
            global_text_embeddings = SyncFunction.apply(text_embeddings, True)
        else:
            global_music_embeddings = music_embeddings
            global_text_embeddings = text_embeddings

        return (
            global_music_embeddings,
            global_text_embeddings,
            local_batch_size,
            is_distributed,
            device,
        )

    @abstractmethod
    def forward(
        self,
        music_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute contrastive loss between music and text embeddings.

        Args:
            music_embeddings: L2-normalized music embeddings (batch_size, embedding_dim)
            text_embeddings: L2-normalized text embeddings (batch_size, embedding_dim)

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

    def forward(
        self,
        music_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute InfoNCE loss between music and text embeddings.

        Args:
            music_embeddings (torch.Tensor): L2-normalized music embeddings of
                shape (batch_size, embedding_dim).
            text_embeddings (torch.Tensor): L2-normalized text embeddings of
                shape (batch_size, embedding_dim).

        Returns:
            torch.Tensor: InfoNCE loss.
        """
        # Validate inputs and gather embeddings across ranks
        (
            global_music_embeddings,
            global_text_embeddings,
            local_batch_size,
            is_distributed,
            device,
        ) = self._validate_and_gather_embeddings(music_embeddings, text_embeddings)

        # Compute similarity matrix using all gathered embeddings
        similarity_matrix = (
            torch.matmul(global_music_embeddings, global_text_embeddings.T) / self.temperature
        )

        if is_distributed:
            # In DDP mode, each rank computes loss only on its local batch portion
            rank = dist.get_rank()
            labels = torch.arange(
                rank * local_batch_size, (rank + 1) * local_batch_size, device=device
            )
            start_index = rank * local_batch_size
            end_index = (rank + 1) * local_batch_size
            local_similarity_m2t = similarity_matrix[start_index:end_index]
            local_similarity_t2m = similarity_matrix.T[start_index:end_index]
        else:
            labels = torch.arange(local_batch_size, device=device)
            local_similarity_m2t = similarity_matrix
            local_similarity_t2m = similarity_matrix.T

        # Compute bidirectional cross-entropy loss
        loss_m2t = F.cross_entropy(local_similarity_m2t, labels, reduction=self.reduction)
        loss_t2m = F.cross_entropy(local_similarity_t2m, labels, reduction=self.reduction)
        loss = (loss_m2t + loss_t2m) / 2

        return loss


class CrossModalNTXentLoss(_CrossModalContrastiveLoss):
    """Cross-modal NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.

    NT-Xent loss is commonly used in cross-modal contrastive learning (e.g., CLIP, MuLan).
    It applies cross-entropy directly to the similarity matrix with temperature scaling.

    This loss function supports Distributed Data Parallel (DDP) training. When DDP is enabled,
    embeddings from all processes are gathered to compute the similarity matrix, increasing
    the effective batch size and number of negative samples.

    Args:
        temperature (float): Initial temperature parameter for scaling the logits.
            This becomes a learnable parameter stored as log_temperature.
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
                >>> criterion = CrossModalNTXentLoss(temperature=0.1).cuda()
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
                >>>         self.criterion = CrossModalNTXentLoss(temperature=0.1)
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

    def forward(
        self,
        music_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute NT-Xent loss between music and text embeddings.

        Args:
            music_embeddings (torch.Tensor): L2-normalized music embeddings of
                shape (batch_size, embedding_dim).
            text_embeddings (torch.Tensor): L2-normalized text embeddings of
                shape (batch_size, embedding_dim).

        Returns:
            torch.Tensor: NT-Xent loss.
        """
        # Validate inputs and gather embeddings across ranks
        (
            global_music_embeddings,
            global_text_embeddings,
            local_batch_size,
            is_distributed,
            device,
        ) = self._validate_and_gather_embeddings(music_embeddings, text_embeddings)

        if is_distributed:
            world_size = dist.get_world_size()
            global_batch_size = local_batch_size * world_size
            rank = dist.get_rank()
        else:
            global_batch_size = local_batch_size

        # Concatenate music and text embeddings
        all_embeddings = torch.cat([global_music_embeddings, global_text_embeddings], dim=0)
        similarity_matrix = torch.matmul(all_embeddings, all_embeddings.T) / self.temperature

        # Create labels: for each music embedding, the positive is the corresponding text embedding
        # and vice versa
        music_indices = torch.arange(global_batch_size, 2 * global_batch_size, device=device)
        text_indices = torch.arange(0, global_batch_size, device=device)
        labels = torch.cat([music_indices, text_indices])

        # Mask out self-similarities
        mask = torch.eye(2 * global_batch_size, device=device, dtype=torch.bool)
        similarity_matrix = similarity_matrix.masked_fill(mask, float("-inf"))

        if is_distributed:
            start_index = rank * local_batch_size
            end_index = (rank + 1) * local_batch_size

            # Extract local portions for both music-to-text and text-to-music
            local_similarity = torch.cat(
                [
                    similarity_matrix[start_index:end_index],  # music embeddings
                    similarity_matrix[
                        global_batch_size + start_index : global_batch_size + end_index
                    ],  # text embeddings
                ],
                dim=0,
            )
            local_labels = torch.cat(
                [
                    labels[start_index:end_index],
                    labels[global_batch_size + start_index : global_batch_size + end_index],
                ],
                dim=0,
            )
            loss = F.cross_entropy(local_similarity, local_labels, reduction=self.reduction)
        else:
            # Non-distributed mode
            loss = F.cross_entropy(similarity_matrix, labels, reduction=self.reduction)

        return loss