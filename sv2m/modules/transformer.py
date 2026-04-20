from typing import Optional

import torch
import torch.nn as nn


class SelfAttentionTransformer(nn.Module):
    def __init__(
        self,
        positional_encoding: nn.Module,
        dim_in: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dim_out: int,
        dropout: float = 0.0,
        use_cls_token: bool = False,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([])
        self.use_cls_token = use_cls_token
        for _ in range(depth):
            attn = nn.MultiheadAttention(dim_in, heads, dropout=dropout)
            ff = nn.Sequential(
                nn.Linear(dim_in, mlp_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_dim, dim_in),
                nn.Dropout(dropout),
            )
            norm1 = nn.LayerNorm(dim_in)
            norm2 = nn.LayerNorm(dim_in)
            self.layers.append(nn.ModuleList([norm1, attn, norm2, ff]))
        self.final_linear = nn.Linear(dim_in, dim_out)

        self.positional_encoding = positional_encoding

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim_in))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        # add cls token
        if self.use_cls_token:
            cls_token = self.cls_token.repeat(x.shape[0], 1, 1)
            x = torch.cat([cls_token, x], dim=1)
            mask = torch.cat([torch.ones(x.shape[0], 1).to(x.device), mask], dim=1)

        # add positional encoding
        x = x + self.positional_encoding(x.shape[1]).repeat(x.shape[0], 1, 1)

        x = x.permute(1, 0, 2)  # [seq_len, bs, dim]
        mask = mask if mask is not None else None
        for norm1, attn, norm2, ff in self.layers:
            x = norm1(x)
            x = attn(x, x, x, key_padding_mask=~(mask.bool()), need_weights=False)[0] + x
            x = norm2(x)
            x = ff(x) + x
        x = x.permute(1, 0, 2)  # [bs, seq_len, dim]
        x = self.final_linear(x)
        x = x.masked_fill(mask.unsqueeze(-1) == 0, 0)
        return x, mask
