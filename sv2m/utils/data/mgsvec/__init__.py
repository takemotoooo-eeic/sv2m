import os
import warnings
from typing import List, Optional, Union

import pandas as pd
import torch
from torch.utils.data import Dataset


class BaseMGSVECDataset(Dataset):
    """Base class for MGSVEC datasets containing shared methods.

    This is an abstract base class. Use TrainingMGSVECDataset or
    InferenceMGSVECDataset instead.
    """

    def __init__(
        self,
        video_feat_dir: str,
        music_feat_dir: str,
        csv_root: str,
        music_max_duration: int,
        max_video_frames: int,
        stride: float,
        subset: Optional[Union[str, List[str]]] = None,
        length: Optional[int] = None,
        crop_music_feat: bool = False,
    ) -> None:
        """Initialize the MGSVEC dataset.

        Args:
            video_feat_dir: Directory containing video feature files. The loader expects
                "vit_feature" and "vit_mask" subdirectories under this path.
            music_feat_dir: Directory containing music feature files. The loader expects
                "ast_feature" and "ast_mask" subdirectories under this path.
            csv_root: Directory containing annotation CSV files such as
                "training_data.csv", "validation_data.csv", and "evaluation_data.csv".
            music_max_duration: Maximum duration used to normalize the target music span.
            max_video_frames: Maximum number of video frames to keep in metadata.
            stride: Temporal stride between music feature frames.
            subset: Subset name(s) to load. If None, all available subsets are loaded.
                Can be a string or a list of strings.
            length: Total number of samples. If None, the length is inferred from the
                concatenated CSV rows.
            crop_music_feat: If True, music features outside the ground-truth span are masked.
        """
        super().__init__()

        if subset is None:
            subset = ["training", "validation", "evaluation"]

        if isinstance(subset, str):
            subsets = [subset]
        else:
            subsets = subset

        csv_paths = []

        for subset_name in subsets:
            _csv_path = f"{csv_root}/{subset_name}_data.csv"
            csv_paths.append(_csv_path)

        self.csv_paths: List[str] = csv_paths
        self.video_feat_dir = video_feat_dir
        self.music_feat_dir = music_feat_dir

        csv_data = [pd.read_csv(csv_path) for csv_path in self.csv_paths]
        self.csv_data = pd.concat(csv_data, ignore_index=True)

        self.max_music_duration = music_max_duration
        self.max_video_frames = max_video_frames
        self.stride = stride

        self.crop_music_feat = crop_music_feat

        if length is None:
            warnings.warn(
                "Dataset length not specified. Using the number of rows in the loaded CSV files. "
                "Consider providing the 'length' parameter to avoid this overhead.",
                UserWarning,
                stacklevel=2,
            )
            self.length = len(self.csv_data)
        else:
            self.length = length

    def __len__(self) -> int:
        return self.length

    def _get_span_propotion(self, gt_spans: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
            gt_spans: [1, 2]
            max_m_duration: float
        Outputs:
            propotion: [1, 2] (center_propotion, width_propotion)
        """
        gt_spans[:, 1] = torch.clamp(gt_spans[:, 1], max=self.max_music_duration)
        center_propotion = (gt_spans[:, 0] + gt_spans[:, 1]) / 2.0 / self.max_music_duration  # [1]
        width_propotion = (gt_spans[:, 1] - gt_spans[:, 0]) / self.max_music_duration  # [1]
        return torch.stack([center_propotion, width_propotion], dim=-1)  # [1, 2]

    def _crop_feats_by_span(
        self,
        music_feats: torch.Tensor,
        music_mask: torch.Tensor,
        music_start_time: float,
        music_end_time: float,
    ):
        """
        Crop music segments to the GT music span only by masking out segments outside the span.
        Keeps tensor shapes unchanged.
        """
        if music_end_time < music_start_time:
            music_start_time, music_end_time = music_end_time, music_start_time

        centers = torch.arange(music_feats.shape[0], dtype=torch.float32) * self.stride
        keep = (centers >= music_start_time) & (centers <= music_end_time)

        music_mask = music_mask * keep.to(music_mask.dtype)
        music_feats = music_feats[keep]
        return music_feats, music_mask

    def __getitem__(self, idx):
        row = self.csv_data.iloc[idx]

        video_id = row["video_id"]
        music_id = row["music_id"]
        music_duration = float(row["music_total_duration"])
        video_start_time = float(row["video_start"])
        video_end_time = float(row["video_end"])
        music_start_time = float(row["music_start"])
        music_end_time = float(row["music_end"])

        gt_windows_list = [(music_start_time, music_end_time)]
        gt_windows = torch.Tensor(gt_windows_list)  # [1, 2]

        # target spans
        spans_target = self._get_span_propotion(gt_windows)  # [1, 2]

        # extract features
        video_feature_path = os.path.join(self.video_feat_dir, "vit_feature", f"{video_id}.pt")
        video_mask_path = os.path.join(self.video_feat_dir, "vit_mask", f"{video_id}.pt")
        video_feats = torch.load(video_feature_path, map_location="cpu")
        video_masks = torch.load(video_mask_path, map_location="cpu")
        video_feats = video_feats.masked_fill(video_masks.unsqueeze(-1) == 0, 0)  # [bs, max_frame_num, 512]

        music_feature_path = os.path.join(self.music_feat_dir, "ast_feature", f"{music_id}.pt")
        music_mask_path = os.path.join(self.music_feat_dir, "ast_mask", f"{music_id}.pt")
        music_feats = torch.load(music_feature_path, map_location="cpu")
        music_mask = torch.load(music_mask_path, map_location="cpu")
        music_feats = music_feats.masked_fill(music_mask.unsqueeze(-1) == 0, 0)  # [bs, max_snippet_num, 768]

        if self.crop_music_feat:
            music_feats, music_mask = self._crop_feats_by_span(
                music_feats,
                music_mask,
                music_start_time,
                music_end_time,
            )

        output = {
            "video_id": str(video_id),
            "music_id": str(music_id),
            "video_duration": torch.tensor(video_end_time - video_start_time),
            "music_duration": torch.tensor(music_duration),
            "gt_moment": gt_windows,  # [1, 2]
            "spans_target": spans_target,  # [1, 2]
            "video_feats": video_feats,
            "video_masks": video_masks,
            "music_feats": music_feats,
            "music_masks": music_mask,
        }
        return output


class TrainingMGSVECDataset(BaseMGSVECDataset):
    """MGSVEC dataset for training.

    This is a thin wrapper around :class:`BaseMGSVECDataset`.
    """

    def __init__(
        self,
        video_feat_dir: str,
        music_feat_dir: str,
        csv_root: str,
        music_max_duration: int,
        max_video_frames: int,
        stride: float,
        subset: Optional[Union[str, List[str]]] = None,
        length: Optional[int] = None,
        crop_music_feat: bool = False,
    ) -> None:
        super().__init__(
            video_feat_dir=video_feat_dir,
            music_feat_dir=music_feat_dir,
            csv_root=csv_root,
            music_max_duration=music_max_duration,
            max_video_frames=max_video_frames,
            stride=stride,
            subset=subset,
            length=length,
            crop_music_feat=crop_music_feat,
        )


class InferenceMGSVECDataset(BaseMGSVECDataset):
    """MGSVEC dataset for inference.

    This is a thin wrapper around :class:`BaseMGSVECDataset`.
    """

    def __init__(
        self,
        video_feat_dir: str,
        music_feat_dir: str,
        csv_root: str,
        music_max_duration: int,
        max_video_frames: int,
        stride: float,
        subset: Optional[Union[str, List[str]]] = None,
        length: Optional[int] = None,
        crop_music_feat: bool = False,
    ) -> None:
        super().__init__(
            video_feat_dir=video_feat_dir,
            music_feat_dir=music_feat_dir,
            csv_root=csv_root,
            music_max_duration=music_max_duration,
            max_video_frames=max_video_frames,
            stride=stride,
            subset=subset,
            length=length,
            crop_music_feat=crop_music_feat,
        )
