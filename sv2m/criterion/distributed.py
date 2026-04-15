from typing import Any

import torch
import torch.distributed as dist


class SyncFunction(torch.autograd.Function):
    """Synchronize tensors across distributed processes.

    This function gathers tensors from all ranks during forward pass and
    distributes gradients back to each rank during backward pass.

    Args:
        tensor: Input tensor to be gathered across all ranks.
        sync_grad: If True, gradients are synchronized (summed) across all ranks
                   using all_reduce. If False, each rank receives only its portion
                   of the gradient without synchronization. Default: False.

    Note:
        - sync_grad=True: Use for contrastive learning where each rank's embeddings
          participate in all ranks' loss computations. Gradients must be accumulated
          across ranks using all_reduce to get the correct total gradient.
        - sync_grad=False: Use when each rank's data is processed independently and
          gradients don't need to be accumulated (rare in contrastive learning).
    """

    @staticmethod
    def forward(ctx: Any, tensor: Any, sync_grad: bool = False) -> Any:
        ctx.batch_size_per_device = tensor.size(0)
        ctx.rank = dist.get_rank()
        ctx.sync_grad = sync_grad

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx: Any, gathered_grad_output: Any) -> Any:
        batch_size_per_device = ctx.batch_size_per_device
        rank = ctx.rank
        sync_grad = ctx.sync_grad

        start_index = rank * batch_size_per_device
        end_index = (rank + 1) * batch_size_per_device

        if sync_grad:
            # Synchronize gradients across all ranks (e.g., for model parameters)
            gathered_grad_input = gathered_grad_output.clone()
            dist.all_reduce(gathered_grad_input, op=dist.ReduceOp.SUM)

            return gathered_grad_input[start_index:end_index], None
        else:
            # Each rank gets its own portion of gradient without synchronization
            # (e.g., for embeddings in contrastive learning)
            return gathered_grad_output[start_index:end_index], None