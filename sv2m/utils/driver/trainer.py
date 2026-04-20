"""
Training utilities for MVPt model.

This module provides training loop and utilities for MVPt contrastive learning.
"""

import copy
import os
from typing import Any, Dict, Optional, Tuple

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from sv2m.models.made import MaDE
from sv2m.criterion import retrieval_metrics
from sv2m.modules.aggregater import XPoolAggregator 

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


class MaDETrainer(Driver):
    """Trainer for MaDE contrastive learning model.

    Args:
        training_dataloader (DataLoader): Data loader for training set.
        validation_dataloader (DataLoader, optional): Data loader for validation set.
        model (MaDe): The MaDE model to train.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler.
        config (DictConfig): Configuration for training.
        device (torch.device, optional): Device to run training on.

    .. note::

        This class does not support criterion because model should contain it.

    """

    def __init__(
        self,
        *,
        training_dataloader: DataLoader = None,
        validation_dataloader: Optional[DataLoader] = None,
        model: MaDE = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        config: DictConfig = None,
        device: Optional[torch.device] = None,
    ) -> None:
        unwrapped_model = unwrap(model)

        assert isinstance(unwrapped_model, MaDE), "Only MaDE is supported as model."
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
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

        self.epoch = 0
        self.iteration = 0
        self.best_validation_loss = float("inf")
        self.history = {
            "training_loss": [],
            "validation_loss": [],
        }
        for index, _ in enumerate(self.optimizer.param_groups):
            self.history[f"learning_rate_{index}"] = []

        unwrapped_model = unwrap(self.model)
        self.logger.info(unwrapped_model)

        if training_config.checkpoint.resume_from:
            self.load_checkpoint(training_config.checkpoint.resume_from)

    def train_for_epoch(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            dataloader (DataLoader): Training data loader.

        Returns:
            dict: Training metrics for the epoch.

        """
        self.model.train()

        self.set_epoch_if_possible(dataloader)

        total_training_loss = 0
        num_training_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {self.epoch + 1}", disable=self.rank != 0)

        for _, batch in enumerate(pbar):
            video_feats = batch["video_feats"]
            music_feats = batch["music_feats"]
            video_masks = batch["video_masks"]
            music_masks = batch["music_masks"]
            music_ids = batch["music_id"]

            video_feats = video_feats.to(self.device)
            music_feats = music_feats.to(self.device)
            video_masks = video_masks.to(self.device)
            music_masks = music_masks.to(self.device)
            self.optimizer.zero_grad()

            _, _, _, _, loss = self.model(
                video_feats=video_feats,
                video_masks=video_masks,
                music_feats=music_feats,
                music_masks=music_masks,
                music_ids=music_ids,
                apply_normalization=True,
            )
            loss.backward()
            self.optimizer.step()

            total_training_loss += loss.item()
            num_training_batches += 1
            average_loss = total_training_loss / num_training_batches

            self.iteration += 1
            self.writer.add_scalar("training_loss/iteration", loss.item(), global_step=self.iteration)

            loss_fn = unwrap(self.model).loss_fn
            self.writer.add_scalar(
                "temperature/iteration",
                float(loss_fn.temperature.detach().item()),
                global_step=self.iteration,
            )

            for index, param_group in enumerate(self.optimizer.param_groups):
                self.writer.add_scalar(
                    f"learning_rate_{index}/iteration",
                    param_group["lr"],
                    global_step=self.iteration,
                )

            pbar.set_postfix(
                {
                    "loss": loss.item(),
                    "average_loss": average_loss,
                }
            )

            if self.scheduler is not None:
                self.scheduler.step()

        average_loss = total_training_loss / num_training_batches
        metrics = {
            "training_loss": average_loss,
        }

        loss_fn = unwrap(self.model).loss_fn
        metrics["temperature"] = float(loss_fn.temperature.detach().item())

        for index, param_group in enumerate(self.optimizer.param_groups):
            metrics[f"learning_rate_{index}"] = param_group["lr"]

        return metrics

    def validate_for_epoch(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        """Validate the model.

        Args:
            dataloader (DataLoader): Validation data loader.

        Returns:
            dict: Validation metrics.

        """
        self.model.eval()

        total_validation_loss = 0
        num_validation_batches = 0

        all_video_features = []
        all_video_masks = []
        all_music_features = []
        all_music_masks = []
        all_music_ids: list[str] = []

        pbar = tqdm(dataloader, desc="Validation", disable=self.rank != 0)

        with torch.no_grad():
            for _, batch in enumerate(pbar):
                video_feats = batch["video_feats"]
                music_feats = batch["music_feats"]
                video_masks = batch["video_masks"]
                music_masks = batch["music_masks"]
                music_ids = batch["music_id"]

                video_feats = video_feats.to(self.device)
                music_feats = music_feats.to(self.device)
                video_masks = video_masks.to(self.device)
                music_masks = music_masks.to(self.device)

                video_embeddings, video_masks, music_embeddings, music_masks, loss = self.model(
                    video_feats=video_feats,
                    music_feats=music_feats,
                    video_masks=video_masks,
                    music_masks=music_masks,
                    music_ids=music_ids,
                    apply_normalization=True,
                )

                total_validation_loss += loss.item()
                num_validation_batches += 1
                average_loss = total_validation_loss / num_validation_batches

                pbar.set_postfix(
                    {
                        "loss": loss.item(),
                        "average_loss": average_loss,
                    }
                )

                all_video_features.append(video_embeddings.detach().cpu())
                all_video_masks.append(video_masks.detach().cpu())
                all_music_features.append(music_embeddings.detach().cpu())
                all_music_masks.append(music_masks.detach().cpu())

                if isinstance(music_ids, (list, tuple)):
                    all_music_ids.extend([str(x) for x in music_ids])
                elif isinstance(music_ids, torch.Tensor):
                    all_music_ids.extend([str(x.item()) for x in music_ids])
                else:
                    all_music_ids.append(str(music_ids))

        average_loss = total_validation_loss / num_validation_batches

        local_video_features = torch.cat(all_video_features, dim=0)
        local_video_masks = torch.cat(all_video_masks, dim=0)
        local_music_features = torch.cat(all_music_features, dim=0)
        local_music_masks = torch.cat(all_music_masks, dim=0)

        if is_distributed_mode():
            world_size = dist.get_world_size()

            gathered_video_features = [None] * world_size
            gathered_video_masks = [None] * world_size
            gathered_music_features = [None] * world_size
            gathered_music_masks = [None] * world_size
            gathered_music_ids = [None] * world_size

            dist.all_gather_object(gathered_video_features, local_video_features)
            dist.all_gather_object(gathered_video_masks, local_video_masks)
            dist.all_gather_object(gathered_music_features, local_music_features)
            dist.all_gather_object(gathered_music_masks, local_music_masks)
            dist.all_gather_object(gathered_music_ids, all_music_ids)

            global_video_features = torch.cat(gathered_video_features, dim=0).to(self.device)
            global_video_masks = torch.cat(gathered_video_masks, dim=0).to(self.device)
            global_music_features = torch.cat(gathered_music_features, dim=0).to(self.device)
            global_music_masks = torch.cat(gathered_music_masks, dim=0).to(self.device)
            global_music_ids = [music_id for rank_ids in gathered_music_ids for music_id in rank_ids]
        else:
            global_video_features = local_video_features.to(self.device)
            global_video_masks = local_video_masks.to(self.device)
            global_music_features = local_music_features.to(self.device)
            global_music_masks = local_music_masks.to(self.device)
            global_music_ids = all_music_ids

        unwrapped_model = unwrap(self.model)
        loss_fn = unwrapped_model.loss_fn

        if loss_fn is not None and len(loss_fn.video_aggregators) > 0:
            similarity_matrix_sum = None
            for video_aggregator, music_aggregator in zip(loss_fn.video_aggregators, loss_fn.music_aggregators):
                video_emb = video_aggregator(global_video_features, global_video_masks)
                video_emb = F.normalize(video_emb, p=2, dim=-1)

                if isinstance(music_aggregator, XPoolAggregator):
                    music_emb = music_aggregator(video_emb, global_music_features, global_music_masks)
                    music_emb = F.normalize(music_emb, p=2, dim=-1)
                    sim = torch.einsum("vmd,vd->vm", music_emb, video_emb)
                else:
                    music_emb = music_aggregator(global_music_features, global_music_masks)
                    music_emb = F.normalize(music_emb, p=2, dim=-1)
                    sim = torch.matmul(video_emb, music_emb.T)
                

                if similarity_matrix_sum is None:
                    similarity_matrix_sum = sim
                else:
                    similarity_matrix_sum = similarity_matrix_sum + sim

            sim_matrix_np = similarity_matrix_sum.detach().cpu().numpy()
        else:
            sim_matrix_np = torch.matmul(global_video_features, global_music_features.T).detach().cpu().numpy()

        retrieval, _, _ = retrieval_metrics(sim_matrix_np, all_music_ids_list=global_music_ids)

        metrics = {
            "validation_loss": average_loss,
        }
        metrics.update({f"validation_{key}": float(value) for key, value in retrieval.items() if isinstance(value, (int, float, np.floating))})

        for key, value in retrieval.items():
            if isinstance(value, (int, float, np.floating)):
                metrics[f"validation_{key}"] = float(value)

        if self.rank == 0:
            self.logger.info(
                "[Validation] "
                + ", ".join(
                    [
                        f"loss={metrics['validation_loss']:.4f}",
                        f"R1={metrics.get('validation_R1', float('nan')):.2f}",
                        f"R5={metrics.get('validation_R5', float('nan')):.2f}",
                        f"R10={metrics.get('validation_R10', float('nan')):.2f}",
                        f"MRR={metrics.get('validation_MRR', float('nan')):.4f}",
                    ]
                )
            )

        return metrics

    def run_for_epoch(
        self,
        training_dataloader: DataLoader,
        validation_dataloader: Optional[DataLoader] = None,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Run training and validation for a single epoch.

        Args:
            training_dataloader (DataLoader): Training data loader.
            validation_dataloader (DataLoader, optional): Validation data loader.

        Returns:
            Tuple[Dict[str, float], Dict[str, float]]: A tuple of
                (training_metrics, validation_metrics). If validation_dataloader is None,
                validation_metrics will be an empty dict.

        """
        training_config = self.config.train
        checkpoint_config = training_config.output.model

        training_metrics = self.train_for_epoch(training_dataloader)

        for metric_key, metric_value in training_metrics.items():
            self.writer.add_scalar(f"{metric_key}/epoch", metric_value, global_step=self.epoch + 1)

        if validation_dataloader is not None:
            validation_metrics = self.validate_for_epoch(validation_dataloader)

            for metric_key, metric_value in validation_metrics.items():
                self.writer.add_scalar(f"{metric_key}/epoch", metric_value, global_step=self.epoch + 1)

            self.epoch += 1

            training_loss = training_metrics["training_loss"]
            validation_loss = validation_metrics["validation_loss"]

            if self.epoch % checkpoint_config.epoch.every == 0:
                path = checkpoint_config.epoch.path
                path = path.format(epoch=self.epoch)
                self.save_checkpoint(path)

            self.logger.info(f"[Epoch {self.epoch}]: training_loss={training_loss:.4f}, validation_loss={validation_loss:.4f}")

            if validation_metrics["validation_loss"] < self.best_validation_loss:
                self.best_validation_loss = validation_metrics["validation_loss"]
                path = checkpoint_config.best_epoch.path

                if path is not None:
                    path = path.format(epoch=self.epoch)
                    self.save_checkpoint(path)
        else:
            self.epoch += 1

            training_loss = training_metrics["training_loss"]

            if self.epoch % checkpoint_config.epoch.every == 0:
                path = checkpoint_config.epoch.path
                path = path.format(epoch=self.epoch)
                self.save_checkpoint(path)

            self.logger.info(f"[Epoch {self.epoch}]: training_loss={training_loss:.4f}")

            validation_metrics = {}

        path = checkpoint_config.last_epoch.path
        path = path.format(epoch=self.epoch)
        self.save_checkpoint(path)
        return training_metrics, validation_metrics

    def run(
        self,
        training_dataloader: Optional[DataLoader] = None,
        validation_dataloader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        """Train the model for multiple epochs.

        Args:
            training_dataloader (DataLoader, optional): Training data loader.
            validation_dataloader (DataLoader, optional): Validation data loader.

        Returns:
            Dict[str, Any]: Training history.

        """
        if training_dataloader is None:
            training_dataloader = self.training_dataloader

        if validation_dataloader is None:
            validation_dataloader = self.validation_dataloader

        training_config = self.config.train

        epochs = training_config.steps.epochs
        iterations = training_config.steps.iterations

        if (epochs is None) == (iterations is None):
            raise ValueError("Set either of config.train.steps.epochs or config.train.steps.iterations.")

        if iterations is not None:
            raise NotImplementedError("iteraions is not supported")

        history = {
            "training_loss": [],
            "validation_loss": [],
        }

        for index, _ in enumerate(self.optimizer.param_groups):
            history[f"learning_rate_{index}"] = []

        for _ in range(self.epoch, epochs):
            training_metrics, validation_metrics = self.run_for_epoch(training_dataloader, validation_dataloader)

            for metric_key, metric_value in training_metrics.items():
                if metric_key in history:
                    history[metric_key].append(metric_value)

            for metric_key, metric_value in validation_metrics.items():
                if metric_key in history:
                    history[metric_key].append(metric_value)

        if self.writer is not None:
            self.writer.close()

        return history

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint.

        Args:
            path (str): Path to save the checkpoint.

        """
        from ... import __version__ as _version

        model_dir = os.path.dirname(path)

        if model_dir:
            os.makedirs(model_dir, exist_ok=True)

        unwrapped_model = unwrap(self.model)
        optimizer = self.optimizer

        checkpoint = {
            "model": unwrapped_model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        if self.scheduler is not None:
            checkpoint["scheduler"] = self.scheduler.state_dict()

        checkpoint["epoch"] = self.epoch
        checkpoint["iteration"] = self.iteration
        checkpoint["best_validation_loss"] = self.best_validation_loss

        config = copy.deepcopy(self.config)
        replace_missing_with_none(config)
        checkpoint["resolved_config"] = OmegaConf.to_container(self.config, resolve=True)

        checkpoint["_metadata"] = {
            "version": _version,
            "driver": self.__class__.__name__,
            "commit_hash": self.commit_hash,
        }

        torch.save(checkpoint, path)

    def load_checkpoint(
        self,
        path: str,
        load_optimizer_state_dict: bool = True,
        load_scheduler_state_dict: bool = True,
    ) -> Dict[str, Any]:
        """Load model checkpoint.

        Args:
            path (str): Path to the checkpoint file.
            load_optimizer_state_dict (bool): Whether to load optimizer state dict.
                Defaults to True.
            load_scheduler_state_dict (bool): Whether to load scheduler state dict.
                Defaults to True.

        Returns:
            Dict[str, Any]: Checkpoint information.

        """
        self.logger.info(f"Load weights from {path}.")

        checkpoint = torch.load(path, map_location=self.device)

        unwrapped_model = unwrap(self.model)
        unwrapped_model.load_state_dict(checkpoint["model"])

        if load_optimizer_state_dict:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        if load_scheduler_state_dict and self.scheduler is not None and "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])

        self.epoch = checkpoint["epoch"]
        self.iteration = checkpoint["iteration"]
        self.best_validation_loss = checkpoint["best_validation_loss"]

    def set_epoch_if_possible(self, dataloader: DataLoader) -> None:
        """Set epoch of samplers if possible.

        Args:
            dataloader (DataLoader): DataLoader.

        """
        if hasattr(dataloader, "set_epoch") and callable(dataloader.set_epoch):
            dataloader.set_epoch(self.epoch)
        else:
            if hasattr(dataloader, "sampler") and dataloader.sampler is not None:
                if hasattr(dataloader.sampler, "set_epoch") and callable(dataloader.sampler.set_epoch):
                    dataloader.sampler.set_epoch(self.epoch)

            if hasattr(dataloader, "batch_sampler") and dataloader.batch_sampler is not None:
                if hasattr(dataloader.batch_sampler, "set_epoch") and callable(dataloader.batch_sampler.set_epoch):
                    dataloader.batch_sampler.set_epoch(self.epoch)

    @classmethod
    def build_from_config(cls, config: DictConfig) -> "MaDETrainer":
        dataloader_config = config.dataloader
        training_config = config.train
        model_config = config.model
        optimizer_config = config.optimizer.optimizer
        scheduler_config = config.optimizer.scheduler

        set_seed(training_config.seed)
        init_distributed_training_if_necessary()

        accelerator = "cuda" if torch.cuda.is_available() else "cpu"

        if is_distributed_mode():
            rank = dist.get_rank()
            world_size = dist.get_world_size()

            OmegaConf.resolve(dataloader_config)

            training_dataset = hydra.utils.instantiate(dataloader_config.train.dataset)
            validation_dataset = hydra.utils.instantiate(dataloader_config.validate.dataset)

            training_sampler = DistributedSampler(training_dataset, num_replicas=world_size, rank=rank, seed=training_config.seed)
            training_dataloader_kwargs = {
                "dataset": training_dataset,
                "sampler": training_sampler,
            }

            # shuffle = True is not supported if sampler is given to data loader
            if "shuffle" in dataloader_config.train and dataloader_config.train.shuffle:
                training_dataloader_kwargs["shuffle"] = False

            validation_sampler = DistributedSampler(
                validation_dataset,
                num_replicas=world_size,
                rank=rank,
                seed=training_config.seed,
            )
            validation_dataloader_kwargs = {
                "dataset": validation_dataset,
                "sampler": validation_sampler,
            }
            if "shuffle" in dataloader_config.validate and dataloader_config.validate.shuffle:
                validation_dataloader_kwargs["shuffle"] = False
        else:
            rank = 0
            world_size = 1

            training_dataloader_kwargs = {}
            validation_dataloader_kwargs = {}

        training_dataloader = hydra.utils.instantiate(dataloader_config.train, **training_dataloader_kwargs)
        validation_dataloader = hydra.utils.instantiate(dataloader_config.validate, **validation_dataloader_kwargs)

        model = hydra.utils.instantiate(model_config)
        model = set_device(
            model,
            accelerator=accelerator,
            is_distributed=is_distributed_mode(),
            ddp_kwargs=training_config.ddp_kwargs,
        )

        optimizer = hydra.utils.instantiate(optimizer_config, model.parameters())
        scheduler = hydra.utils.instantiate(scheduler_config, optimizer)
        device = next(model.parameters()).device

        return cls(
            training_dataloader=training_dataloader,
            validation_dataloader=validation_dataloader,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            device=device,
        )
