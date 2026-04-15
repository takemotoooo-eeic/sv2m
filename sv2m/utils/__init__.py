import os
import random
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from ._torch import convert_dtype
from .cache import get_cache_dir

__all__ = [
    "cache_dir",
    "convert_dtype",
    "set_seed",
    "set_device",
    "select_device",
]

cache_dir = get_cache_dir()


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def set_device(
    module: nn.Module,
    accelerator: str,
    is_distributed: bool = False,
    ddp_kwargs: Optional[Dict[str, Any]] = None,
) -> Union[nn.Module, nn.parallel.DistributedDataParallel]:
    device = select_device(accelerator, is_distributed=is_distributed)
    module = module.to(device)

    is_trainable = any(p.requires_grad for p in module.parameters())

    if ddp_kwargs is None:
        ddp_kwargs = {}

    if is_distributed and is_trainable:
        module = nn.parallel.DistributedDataParallel(module, device_ids=[device], **ddp_kwargs)

    return module


def select_device(accelerator: Optional[str], is_distributed: bool = False) -> str:
    if accelerator is None:
        accelerator = "cuda" if torch.cuda.is_available() else "cpu"

    if accelerator in ["cuda", "gpu"] and is_distributed:
        device = int(os.environ["LOCAL_RANK"])
    elif accelerator in ["cpu", "cuda", "mps"]:
        device = accelerator
    elif accelerator == "gpu":
        device = "cuda"
    else:
        raise ValueError(f"Unknown accelerator {accelerator} is specified.")

    return device
