import os

import torch
import torch.distributed as dist
import torch.nn as nn

__all__ = [
    "is_distributed_mode",
    "unwrap",
    "init_distributed_training_if_necessary",
]


def is_distributed_mode() -> bool:
    return dist.is_available() and dist.is_initialized()


def unwrap(module: nn.Module) -> nn.Module:
    """Unwrap a module from DistributedDataParallel wrapper recursively.

    Args:
        module (nn.Module): Module to unwrap.

    Returns:
        nn.Module: Unwrapped module.

    """
    if isinstance(module, nn.parallel.DistributedDataParallel):
        return unwrap(module.module)
    else:
        return module


def init_distributed_training_if_necessary() -> None:
    """Initialize distributed training if environment variables are set.

    Sets up distributed training using NCCL backend for GPU or Gloo backend for CPU.
    This function supports both single-node multi-GPU and multi-node multi-GPU training.

    Environment Variables:
        RANK: Global rank of the current process (0 to WORLD_SIZE-1)
        WORLD_SIZE: Total number of processes across all nodes
        LOCAL_RANK: Local rank of the current process within its node
        MASTER_ADDR: Address of the master node (required for multi-node)
        MASTER_PORT: Port on the master node (required for multi-node)

    .. note::

        This function should be called before model initialization in distributed training.
        Environment variables must be set by the launcher (e.g., ``torchrun``,
        ``torch.distributed.launch``, or manual setup with SLURM).

    Examples:
        Single-node, multi-GPU (4 GPUs):
            torchrun --nproc_per_node=4 train.py

        Multi-node, multi-GPU (2 nodes, 4 GPUs each):
            Node 0: torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \\
                             --master_addr="192.168.1.1" --master_port=29500 train.py
            Node 1: torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \\
                             --master_addr="192.168.1.1" --master_port=29500 train.py
    """
    # Check if distributed training is requested via environment variables
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        global_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # Determine backend based on available hardware
        if torch.cuda.is_available():
            backend = "nccl"
            torch.cuda.set_device(local_rank)
        else:
            backend = "gloo"

        # For multi-node training, ensure MASTER_ADDR and MASTER_PORT are set
        if world_size > 1:
            if world_size > torch.cuda.device_count():
                # Multi-node case
                if "MASTER_ADDR" not in os.environ or "MASTER_PORT" not in os.environ:
                    raise RuntimeError(
                        "Multi-node training requires MASTER_ADDR and MASTER_PORT "
                        "environment variables to be set."
                    )

            dist.init_process_group(
                backend=backend,
                init_method="env://",
                rank=global_rank,
                world_size=world_size,
            )
