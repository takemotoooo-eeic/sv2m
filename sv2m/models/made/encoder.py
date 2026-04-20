from typing import Optional

import torch
import torch.nn as nn

from sv2m.modules.gelu import QuickGELU


class UnimodalEncoder(nn.Module):
    def __init__(
        self,
        projector: nn.Module,
        temporal_backbone: nn.Module,
        activation: bool = False,
    ) -> None:
        super().__init__()
        self.projector = projector
        self.temporal_backbone = temporal_backbone
        self.activation = activation
        if self.activation:
            self.act = QuickGELU()
        else:
            self.act = nn.Identity()

    def forward(self, feat: torch.Tensor, mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.projector(feat)
        if self.activation:
            feat = self.act(feat)
        feat, mask = self.temporal_backbone(feat, mask)
        return feat, mask
