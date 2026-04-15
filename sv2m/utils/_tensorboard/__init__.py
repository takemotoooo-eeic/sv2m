import os
from typing import Optional

from torch.utils.tensorboard import SummaryWriter

__all__ = [
    "get_writer",
]


def get_writer(log_dir: Optional[str] = None, is_distributed: bool = False) -> SummaryWriter:
    """Get logger by name.

    When ``is_distributed=True`` and ``int(os.environ["RANK"])>0``, a dummy writer is returned.

    Args:
        log_dir (str, optional): Directory to save logs.

    Returns:
        SummaryWriter: Writer to record training.

    """
    if is_distributed and int(os.environ["RANK"]) > 0:
        writer = DummySummaryWriter(log_dir)
    else:
        writer = SummaryWriter(log_dir)

    return writer


class DummySummaryWriter(SummaryWriter):
    def __init__(self, *args, **kwargs) -> None:
        pass

    def __getattribute__(self, name: str) -> None:
        if hasattr(super(), name):

            def _no_ops(*args, **kwargs) -> None:
                pass

            return _no_ops
        else:
            raise AttributeError(f"SummaryWriter has no attribute {name}.")
