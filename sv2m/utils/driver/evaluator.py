"""
Evaluation utilities for MVPt model.

This module provides evaluation loop and utilities for MVPt video-music retrieval.
"""

import warnings
import os
import json
from typing import Any, Dict, Optional
import json
import numpy as np

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from sv2m.models.made import MaDE
from sv2m.criterion import retrieval_metrics, calculate_miou
from sv2m.modules.aggregater import LateInteractionAggregator, XPoolAggregator

from ...distributed import unwrap
from ...models.mvpt import MVPt

from ...amp import should_enable_amp
from ...distributed import (
    init_distributed_training_if_necessary,
    is_distributed_mode,
    unwrap,
)

from .. import convert_dtype, set_device, set_seed
from .._omegaconf import replace_missing_with_none
from .._tensorboard import get_writer
from ..logging import get_logger
from .base import Driver


class MaDEEvaluator(Driver):
    """Evaluator for MaDE contrastive learning model.

    Args:
        evaluation_dataloader (DataLoader): Data loader for evaluation set.
        model (MaDE): The MaDE model to evaluate.
        config (DictConfig): Configuration for evaluation.
        device (torch.device, optional): Device to run evaluation on.

    .. note::

        This class does not support criterion because model should contain it.

    """

    def __init__(
        self,
        *,
        evaluation_dataloader: DataLoader = None,
        model: MVPt = None,
        config: DictConfig = None,
        device: Optional[torch.device] = None,
    ) -> None:
        unwrapped_model = unwrap(model)

        assert isinstance(unwrapped_model, MaDE), "Only MaDE is supported as model."

        self.evaluation_dataloader = evaluation_dataloader
        self.model = model
        self.config = config

        self._reset(config, device=device)

    def _reset(
        self,
        config: DictConfig,
        device: Optional[torch.device] = None,
    ) -> None:
        training_config = config.train

        if device is None:
            device = next(self.model.parameters()).device

        dtype = convert_dtype(training_config.torch_dtype)
        enable_amp = should_enable_amp(dtype)

        tensorboard_dir = os.path.join(training_config.output.tensorboard_dir)

        if is_distributed_mode():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            logger = get_logger(is_distributed=True)
            writer = get_writer(log_dir=tensorboard_dir, is_distributed=True)
        else:
            rank = 0
            world_size = 1
            logger = get_logger(is_distributed=False)
            writer = get_writer(log_dir=tensorboard_dir, is_distributed=False)

        self.set_commit_hash()

        self.device = device
        self.dtype = dtype
        self.enable_amp = enable_amp

        self.rank = rank
        self.world_size = world_size
        self.logger = logger
        self.writer = writer

        unwrapped_model = unwrap(self.model)
        self.logger.info(unwrapped_model)

    def evaluate(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        """Evaluate with full-dataset similarity matrix (like trainer validation)."""
        self.model.eval()

        total_evaluation_loss = 0.0
        num_evaluation_batches = 0

        all_video_features = []
        all_video_masks = []
        all_music_features = []
        all_music_masks = []
        all_music_span_masks = []
        all_spans_target = []
        all_predicted_spans = []
        all_music_ids: list[str] = []

        pbar = tqdm(dataloader, desc="Evaluation", disable=self.rank != 0)

        with torch.no_grad():
            for batch in pbar:
                video_feats = batch["video_feats"].to(self.device)
                music_feats = batch["music_feats"].to(self.device)
                video_masks = batch["video_masks"].to(self.device)
                music_masks = batch["music_masks"].to(self.device)
                music_span_masks = batch["music_span_masks"].to(self.device)
                spans_target = batch["spans_target"].to(self.device)
                music_ids = batch["music_id"]

                video_embeddings, video_masks, music_embeddings, music_masks, loss, predict_spans = self.model(
                    video_feats=video_feats,
                    music_feats=music_feats,
                    video_masks=video_masks,
                    music_masks=music_masks,
                    music_span_masks=music_span_masks,
                    spans_target=spans_target,
                    music_ids=music_ids,
                    apply_normalization=True,
                )

                total_evaluation_loss += loss.item()
                num_evaluation_batches += 1

                all_video_features.append(video_embeddings.detach().cpu())
                all_video_masks.append(video_masks.detach().cpu())
                all_music_features.append(music_embeddings.detach().cpu())
                all_music_masks.append(music_masks.detach().cpu())
                all_music_span_masks.append(music_span_masks.detach().cpu())
                all_predicted_spans.append(predict_spans.detach().cpu())
                all_spans_target.append(spans_target.detach().cpu())

                if isinstance(music_ids, (list, tuple)):
                    all_music_ids.extend([str(x) for x in music_ids])
                elif isinstance(music_ids, torch.Tensor):
                    all_music_ids.extend([str(x.item()) for x in music_ids])
                else:
                    all_music_ids.append(str(music_ids))

                pbar.set_postfix(
                    {
                        "loss": loss.item(),
                        "average_loss": total_evaluation_loss / num_evaluation_batches,
                    }
                )

        average_loss = total_evaluation_loss / max(num_evaluation_batches, 1)

        local_video_features = torch.cat(all_video_features, dim=0)
        local_video_masks = torch.cat(all_video_masks, dim=0)
        local_music_features = torch.cat(all_music_features, dim=0)
        local_music_masks = torch.cat(all_music_masks, dim=0)
        local_music_span_masks = torch.cat(all_music_span_masks, dim=0)
        local_predicted_spans = torch.cat(all_predicted_spans, dim=0)
        local_spans_target = torch.cat(all_spans_target, dim=0)

        if is_distributed_mode():
            world_size = dist.get_world_size()

            gathered_video_features = [None] * world_size
            gathered_video_masks = [None] * world_size
            gathered_music_features = [None] * world_size
            gathered_music_masks = [None] * world_size
            gathered_music_span_masks = [None] * world_size
            gathered_music_ids = [None] * world_size
            gathered_predicted_spans = [None] * world_size
            gathered_spans_target = [None] * world_size

            dist.all_gather_object(gathered_video_features, local_video_features)
            dist.all_gather_object(gathered_video_masks, local_video_masks)
            dist.all_gather_object(gathered_music_features, local_music_features)
            dist.all_gather_object(gathered_music_masks, local_music_masks)
            dist.all_gather_object(gathered_music_span_masks, local_music_span_masks)
            dist.all_gather_object(gathered_music_ids, all_music_ids)
            dist.all_gather_object(gathered_predicted_spans, local_predicted_spans)
            dist.all_gather_object(gathered_spans_target, local_spans_target)

            global_video_features = torch.cat(gathered_video_features, dim=0).to(self.device)
            global_video_masks = torch.cat(gathered_video_masks, dim=0).to(self.device)
            global_music_features = torch.cat(gathered_music_features, dim=0).to(self.device)
            global_music_masks = torch.cat(gathered_music_masks, dim=0).to(self.device)
            global_music_span_masks = torch.cat(gathered_music_span_masks, dim=0).to(self.device)
            global_music_ids = [music_id for rank_ids in gathered_music_ids for music_id in rank_ids]
            global_predicted_spans = torch.cat(gathered_predicted_spans, dim=0).to(self.device)
            global_spans_target = torch.cat(gathered_spans_target, dim=0).to(self.device)
        else:
            global_video_features = local_video_features.to(self.device)
            global_video_masks = local_video_masks.to(self.device)
            global_music_features = local_music_features.to(self.device)
            global_music_masks = local_music_masks.to(self.device)
            global_music_span_masks = local_music_span_masks.to(self.device)
            global_music_ids = all_music_ids
            global_predicted_spans = local_predicted_spans.to(self.device)
            global_spans_target = local_spans_target.to(self.device)

        unwrapped_model = unwrap(self.model)
        loss_fn = unwrapped_model.loss_fn

        def _compute_late_interaction_sim_chunked(
            video_features: torch.Tensor,
            music_features: torch.Tensor,
            video_masks: torch.Tensor,
            music_masks: torch.Tensor,
            music_span_masks: torch.Tensor,
            chunk_size: int,
        ) -> torch.Tensor:
            sims = []
            for start in range(0, video_features.size(0), chunk_size):
                end = min(start + chunk_size, video_features.size(0))
                chunk_sim = loss_fn.compute_late_interaction_similarity_matrix(
                    video_features[start:end],
                    music_features,
                    video_masks[start:end],
                    music_masks,
                    music_span_masks,
                )
                sims.append(chunk_sim)
            return torch.cat(sims, dim=0)

        if loss_fn is not None and len(loss_fn.video_aggregators) > 0:
            similarity_matrixs: list[torch.Tensor] = []
            for video_aggregator, music_aggregator in zip(loss_fn.video_aggregators, loss_fn.music_aggregators):
                if isinstance(video_aggregator, LateInteractionAggregator) and isinstance(music_aggregator, LateInteractionAggregator):
                    late_interaction_chunk_size = self.config.dataloader.evaluate.batch_size

                    if is_distributed_mode():
                        local_video_features_on_device = local_video_features.to(self.device)
                        local_video_masks_on_device = local_video_masks.to(self.device)

                        local_sim = _compute_late_interaction_sim_chunked(
                            local_video_features_on_device,
                            global_music_features,
                            local_video_masks_on_device,
                            global_music_masks,
                            global_music_span_masks,
                            late_interaction_chunk_size,
                        ) / loss_fn.temperature

                        gathered_local_sims = [None] * dist.get_world_size()
                        dist.all_gather_object(gathered_local_sims, local_sim.detach().cpu())
                        sim = torch.cat(gathered_local_sims, dim=0).to(self.device)
                    else:
                        sim = _compute_late_interaction_sim_chunked(
                            global_video_features,
                            global_music_features,
                            global_video_masks,
                            global_music_masks,
                            global_music_span_masks,
                            late_interaction_chunk_size,
                        ) / loss_fn.temperature

                    similarity_matrixs.append(sim)
                    continue

                video_emb = video_aggregator(global_video_features, global_video_masks)  # [batch_size, embed_dim]

                if isinstance(music_aggregator, XPoolAggregator):
                    music_emb, _ = music_aggregator(video_emb, global_music_features, global_music_masks, global_music_span_masks)
                    sim = torch.einsum("vmd,vd->vm", music_emb, video_emb) / loss_fn.temperature
                else:
                    music_emb = music_aggregator(global_music_features, global_music_masks, global_music_span_masks)
                    sim = torch.matmul(video_emb, music_emb.T) / loss_fn.temperature

                similarity_matrixs.append(sim)

            sim_matrix: np.ndarray = torch.stack(similarity_matrixs).sum(dim=0).detach().cpu().numpy()
            sim_matrixs: list[np.ndarray] = [sim.detach().cpu().numpy() for sim in similarity_matrixs]
        else:
            raise ValueError("Unsupported loss function for retrieval metrics calculation.")

        retrieval, _, _ = retrieval_metrics(sim_matrix, all_music_ids_list=global_music_ids)
        retrievals = []
        for sim in sim_matrixs:
            ret, _, _ = retrieval_metrics(sim, all_music_ids_list=global_music_ids)
            retrievals.append(ret)
        miou = calculate_miou(global_predicted_spans, global_spans_target, dataloader.dataset.max_music_duration)

        metrics = {
            "evaluation_loss": float(average_loss),
            "evaluation_miou": float(miou),
        }
        for key, value in retrieval.items():
            if isinstance(value, (int, float, np.floating)):
                metrics[f"evaluation_{key}"] = float(value)

        for idx, ret in enumerate(retrievals):
            for key, value in ret.items():
                if isinstance(value, (int, float, np.floating)):
                    metrics[f"evaluation_{key}_sim{idx}"] = float(value)

        return metrics

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        if self.rank != 0:
            return

        log_items = []
        for key in sorted(metrics.keys()):
            value = metrics[key]
            if isinstance(value, (int, float)):
                if "loss" in key or "MRR" in key:
                    log_items.append(f"{key}={value:.4f}")
                elif "miou" in key.lower():
                    log_items.append(f"{key}={value:.4f}")
                else:
                    log_items.append(f"{key}={value:.2f}")
                self.writer.add_scalar(f"evaluate/{key}", value, global_step=0)

        self.logger.info("[Evaluation] " + ", ".join(log_items))

    def run(
        self,
        evaluation_dataloader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        """Run complete evaluation.

        Args:
            evaluation_dataloader (DataLoader, optional): Evaluation data loader.
                If None, uses the dataloader provided during initialization.

        Returns:
            Dict[str, Any]: Dictionary containing evaluation results for both directions:
                - video-to-music: Metrics for video-to-music retrieval
                - music-to-video: Metrics for music-to-video retrieval

        """
        if evaluation_dataloader is None:
            evaluation_dataloader = self.evaluation_dataloader

        evaluate_config = self.config.evaluate

        metrics = self.evaluate(evaluation_dataloader)
        self.log_metrics(metrics)
        self.save_scores(metrics, path=evaluate_config.output.scores)

        if self.writer is not None:
            self.writer.close()

        return metrics
    
    def save_scores(
        self,
        scores: Dict[str, Any],
        path: str,
    ) -> None:
        """Save evaluation scores to JSON file.

        Args:
            scores (Dict[str, Any]): Scores dictionary.
            path (str): Path to save scores as JSON file.

        """
        output_dir = os.path.dirname(path)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(path, "w") as f:
            json.dump(scores, f, indent=4)

        self.logger.info(f"Scores saved to {path}")

    @classmethod
    def build_from_config(cls, config: DictConfig) -> "MaDEEvaluator":
        """Build evaluator from Hydra config.

        Args:
            config (DictConfig): Hydra configuration.

        Returns:
            MVPtEvaluator: Configured evaluator instance.

        Raises:
            ValueError: If checkpoint path is not specified in config.

        """
        dataloader_config = config.dataloader
        evaluation_config = config.evaluate

        set_seed(evaluation_config.seed)

        accelerator = "cuda" if torch.cuda.is_available() else "cpu"

        evaluation_dataloader = hydra.utils.instantiate(dataloader_config.evaluate)

        if not evaluation_config.checkpoint.pretrained_model:
            raise ValueError("evaluate.checkpoint.pretrained_model must be specified in config")

        checkpoint = torch.load(
            evaluation_config.checkpoint.pretrained_model,
            map_location="cpu",
        )

        resolved_config = checkpoint["resolved_config"]
        resolved_config = OmegaConf.create(resolved_config)

        # confirm given model config is compatible with resolved_config.model
        if resolved_config.model._target_ != config.model._target_:
            warnings.warn(
                f"Model type mismatch: checkpoint contains '{resolved_config.model._target_}' "
                f"but config specifies '{config.model._target_}'. "
                f"This may lead to incompatibility issues.",
                UserWarning,
                stacklevel=2,
            )

        model = hydra.utils.instantiate(resolved_config.model)
        model = set_device(
            model,
            accelerator=accelerator,
            is_distributed=False,
        )
        model.load_state_dict(checkpoint["model"])

        return cls(
            evaluation_dataloader=evaluation_dataloader,
            model=model,
            config=config,
            device=torch.device(accelerator),
        )