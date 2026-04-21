import torch

def calculate_miou(predict_spans: torch.Tensor, gt_spans: torch.Tensor, max_music_duration: float = 1.0) -> torch.Tensor:
        """
        Args:
            predict_spans: (batch_size, 1, 2) - predicted spans in normalized format [0, 1]
            gt_spans: (batch_size, 1, 2) - ground truth spans in normalized format [0, 1]
            max_music_duration: float - maximum duration of music in seconds
        Returns:            
            miou: (1,) - mean Intersection over Union between predicted spans and ground truth spans
        """
        gt_c =  gt_spans[:, 0, 0] * max_music_duration
        gt_w = gt_spans[:, 0, 1] * max_music_duration
        gt_start = gt_c.unsqueeze(-1) - gt_w.unsqueeze(-1) / 2.0
        gt_end = gt_c.unsqueeze(-1) + gt_w.unsqueeze(-1) / 2.0

        pred_c = predict_spans[:, 0, 0] * max_music_duration
        pred_w = predict_spans[:, 0, 1] * max_music_duration
        pred_start = pred_c.unsqueeze(-1) - pred_w.unsqueeze(-1) / 2.0
        pred_end = pred_c.unsqueeze(-1) + pred_w.unsqueeze(-1) / 2.0

        intersection = torch.clamp(torch.min(pred_end, gt_end) - torch.max(pred_start, gt_start), min=0.0)
        union = torch.clamp(torch.max(pred_end, gt_end) - torch.min(pred_start, gt_start), min=1e-12)
        iou = (intersection / union).squeeze(-1)
        miou = iou.mean()

        return miou
