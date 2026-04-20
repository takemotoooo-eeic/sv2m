import torch
import torch.nn as nn


class QuickGELU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform quick gelu."""
        return x * torch.sigmoid(1.702 * x)
