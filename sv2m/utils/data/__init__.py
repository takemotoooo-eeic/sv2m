from typing import Any, Iterator

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

__all__ = [
    "MGSVECLoader",
]


class MGSVECLoader:
    """Distributed-aware DataLoader wrapper for MGSVEC datasets.

    The wrapper creates ``DistributedSampler`` internally when distributed training
    is initialized, and delegates iteration/length/attributes to an internal
    ``torch.utils.data.DataLoader`` instance.
    """

    def __init__(
        self,
        dataset: Any,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        drop_last: bool = False,
        pin_memory: bool = False,
        **kwargs,
    ) -> None:
        """Initialize DataLoader with internal distributed sampler.

        Args:
            dataset: Dataset instance.
            batch_size: Batch size.
            shuffle: Shuffle flag. In distributed mode, sampler controls shuffle.
            num_workers: Number of worker processes.
            drop_last: Whether to drop the last incomplete batch.
            pin_memory: Whether to pin memory in DataLoader.
            **kwargs: Additional arguments passed to ``torch.utils.data.DataLoader``.
        """
        if shuffle is None:
            shuffle = True

        self._dataset = dataset

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.sampler = DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
            )
            dataloader_shuffle = False
        else:
            self.sampler = None
            dataloader_shuffle = shuffle

        self._dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=dataloader_shuffle,
            sampler=self.sampler,
            drop_last=drop_last,
            pin_memory=pin_memory,
            **kwargs,
        )

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for internal sampler/dataset when available."""
        if self.sampler is not None and hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)

        if hasattr(self._dataset, "set_epoch"):
            self._dataset.set_epoch(epoch)

    def __iter__(self) -> Iterator[Any]:
        """Iterate over batches from the wrapped dataloader."""
        yield from self._dataloader

    def __len__(self) -> int:
        """Return the number of batches."""
        return len(self._dataloader)

    def __getattr__(self, name):
        """Delegate attribute access to the wrapped dataloader."""
        return getattr(self._dataloader, name)
