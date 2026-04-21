import torch
import torch.nn as nn
import torch.nn.functional as F




class KLDivLoss(nn.Module):
    def __init__(
        self, 
        music_max_duration: int, 
        stride: float, 
        window_shape: str = "uniform", 
        apply_negative_sample: bool = False,
        weight: float = 1.0
    ) -> None:
        super().__init__()
        self.music_max_duration = music_max_duration
        self.stride = stride
        self.window_shape = window_shape
        self.apply_negative_sample = apply_negative_sample

        seq_len = int(self.music_max_duration // self.stride)
        self.register_buffer("start_time", torch.arange(seq_len, dtype=torch.float32) * self.stride)
        self.register_buffer("end_time", self.start_time + self.stride)

        self.weight = weight

    def _create_span_window(self, spans: torch.Tensor, music_masks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spans: (batch_size, 1, 2)
            music_masks: (batch_size, seq_len)
        Returns:
            span_window: (batch_size, seq_len)
        """
        gt_c = spans[:, 0, 0] * self.music_max_duration # [batch_size]
        gt_w = spans[:, 0, 1] * self.music_max_duration # [batch_size]
        
        gt_start = gt_c.unsqueeze(-1) - gt_w.unsqueeze(-1) / 2.0
        gt_end = gt_c.unsqueeze(-1) + gt_w.unsqueeze(-1) / 2.0
        
        overlap = (self.start_time.unsqueeze(0) < gt_end) & (self.end_time.unsqueeze(0) > gt_start)
        gt_mask = overlap.float() * music_masks

        if self.window_shape == "uniform":
            q_sum = gt_mask.sum(dim=1, keepdim=True).clamp(min=1e-12)
            q_gt = gt_mask / q_sum
        else:
            seg_center = (self.start_time + self.end_time) * 0.5
            diff = seg_center.unsqueeze(0) - gt_c.unsqueeze(-1)
            gt_w_half = (gt_w / 2.0).clamp(min=self.stride * 0.5).unsqueeze(-1)
            if self.window_shape == "gaussian":
                q_gt = torch.exp(-0.5 * (diff / gt_w_half) ** 2)
            elif self.window_shape == "triangle":
                q_gt = (1.0 - (diff.abs() / gt_w_half)).clamp(min=0.0)
            else:
                raise ValueError(f"Unknown window_shape: {self.window_shape}")
            q_gt = q_gt * music_masks
            q_sum = q_gt.sum(dim=1, keepdim=True).clamp(min=1e-12)
            q_gt = q_gt / q_sum
        return q_gt
    
    def _create_uniform_window(self, music_masks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            music_masks: (batch_size, seq_len)
        Returns:
            uniform_window: (batch_size, seq_len)
        """
        q_uniform = music_masks.float()
        q_sum = q_uniform.sum(dim=1, keepdim=True).clamp(min=1e-12)
        q_uniform = q_uniform / q_sum
        return q_uniform

    def forward(
            self, 
            attention_weights: torch.Tensor,
            music_masks: torch.Tensor,
            span_target: torch.Tensor,
            positive_col_offset: int = 0,
        ) -> torch.Tensor:
        """
        Args:
            attention_weights: (V, M, S)
            music_masks: (M, S)
            span_target: (V, 1, 2)
        """
        V, M, S = attention_weights.shape
        attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True).clamp(min=1e-12)

        q_uniform = self._create_uniform_window(music_masks) # (M, S)
        q_span = self._create_span_window(span_target, music_masks) # (V, S)

        target = q_uniform.unsqueeze(0).expand(V, M, S).clone() # (V, M, S)

        diag_row_idx = torch.arange(V, device=attention_weights.device)
        diag_col_idx = diag_row_idx + positive_col_offset
        valid = diag_col_idx < M
        target[diag_row_idx[valid], diag_col_idx[valid]] = q_span[diag_row_idx[valid]]

        kl_matrix = F.kl_div(
            input=torch.log(attention_weights.clamp(min=1e-12)),
            target=target,
            reduction="none",
        ).sum(dim=-1)  # (V, M)

        if self.apply_negative_sample:
            loss = kl_matrix.mean()
        else:
            if valid.any():
                loss = kl_matrix[diag_row_idx[valid], diag_col_idx[valid]].mean()
            else:
                loss = torch.tensor(0.0, device=kl_matrix.device, dtype=kl_matrix.dtype)

        return loss * self.weight