import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, seq_len: int, dim_model: int):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(seq_len, dim_model, requires_grad=False)  # [seq_len, dim_model]
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # [seq_len, 1]
        div_term = torch.exp(torch.arange(0, dim_model, 2).float() * -(math.log(10000.0) / dim_model))  # [dim_model/2]
        pe[:, 0::2] = torch.sin(position * div_term)  # [seq_len, dim_model/2]
        pe[:, 1::2] = torch.cos(position * div_term)  # [seq_len, dim_model/2]
        pe = pe.unsqueeze(0)  # [1, seq_len, dim_model]
        self.register_buffer("pe", pe)

    def forward(self, length: int) -> torch.Tensor:
        return self.pe[:, :length]  # [1, length, dim_model]
