import math

import torch
from torch.optim.lr_scheduler import LambdaLR


class ConstantLRSchedule(LambdaLR):
    """Constant learning rate schedule."""

    def __init__(self, optimizer: torch.optim.Optimizer, last_epoch: int = -1):
        super(ConstantLRSchedule, self).__init__(optimizer, lambda _: 1.0, last_epoch=last_epoch)


class WarmupConstantSchedule(LambdaLR):
    """Linear warmup and then constant.
    Linearly increases learning rate schedule from 0 to 1 over `warmup_steps` training steps.
    Keeps learning rate schedule equal to 1. after warmup_steps.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_rate: float,
        total_steps: int,
        last_epoch: int = -1,
    ):
        self.warmup_steps = int(total_steps * warmup_rate)
        super(WarmupConstantSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step: int) -> float:
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        return 1.0


class WarmupLinearSchedule(LambdaLR):
    """Linear warmup and then linear decay.
    Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
    Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_rate: float,
        total_steps: int,
        last_epoch: int = -1,
    ):
        self.warmup_steps = int(total_steps * warmup_rate)
        self.total_steps = total_steps
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step: int) -> float:
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(
            0.0,
            float(self.total_steps - step) / float(max(1.0, self.total_steps - self.warmup_steps)),
        )


class WarmupCosineSchedule(LambdaLR):
    """Linear warmup and then cosine decay.
    Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
    Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
    If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_rate: float,
        total_steps: int,
        cycles: float = 0.5,
        last_epoch: int = -1,
    ):
        self.warmup_steps = int(total_steps * warmup_rate)
        self.total_steps = total_steps
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step: int) -> float:
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))
