"""
Enhanced evaluation script with late interaction visualization.

This script extends the standard evaluation to include visualization of
similarity matrices for positive music-video pairs during late interaction.
"""

import os
import sys
import warnings
from pathlib import Path
from typing import Optional, Dict, Any

import hydra
import torch
from omegaconf import DictConfig

# Add the current directory to sys.path to import visualization
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visualization import (
    visualize_late_interaction_similarity_with_spans,
    analyze_late_interaction_patterns
)

import sv2m
from sv2m.utils.driver.evaluator import MaDEEvaluator
from sv2m.modules.aggregater import LateInteractionAggregator


class MaDEEvaluatorWithVisualization(MaDEEvaluator):
    """Extended MaDE evaluator that includes late interaction visualization."""

    def __init__(self, *args, visualization_config: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.visualization_config = visualization_config or {}

    def evaluate_with_visualization(self, dataloader) -> Dict[str, float]:
        """
        Evaluate with visualization of late interaction similarity matrices.
        """
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
        all_music_ids: list[list[str]] = []

        print("Starting evaluation with visualization...")

        # Process batches (similar to parent class but collect more detailed info)
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= self.visualization_config.get('max_batches', float('inf')):
                    break

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
                    all_music_ids.append([str(x) for x in music_ids])
                elif isinstance(music_ids, torch.Tensor):
                    all_music_ids.append([str(x.item()) for x in music_ids])
                else:
                    all_music_ids.append([str(music_ids)])

                print(f"Processed batch {batch_idx + 1}/{len(dataloader)}")

        print("Computing similarity matrices...")

        # Concatenate all features
        global_video_features = torch.cat(all_video_features, dim=0)
        global_video_masks = torch.cat(all_video_masks, dim=0)
        global_music_features = torch.cat(all_music_features, dim=0)
        global_music_masks = torch.cat(all_music_masks, dim=0)
        global_music_span_masks = torch.cat(all_music_span_masks, dim=0)
        global_spans_target = torch.cat(all_spans_target, dim=0)
        global_music_ids = [music_id for rank_ids in all_music_ids for music_id in rank_ids]

        # Compute similarity matrices
        unwrapped_model = sv2m.distributed.unwrap(self.model)
        loss_fn = unwrapped_model.loss_fn

        if loss_fn is not None and len(loss_fn.video_aggregators) > 0:
            chunk_size = self.config.dataloader.evaluate.batch_size
            similarity_matrixs, _ = loss_fn.compute_similarity_matrixs(
                video_features=global_video_features.to(self.device),
                music_features=global_music_features.to(self.device),
                video_masks=global_video_masks.to(self.device),
                music_masks=global_music_masks.to(self.device),
                music_span_masks=global_music_span_masks.to(self.device),
                chunk_size=chunk_size,
            )
            similarity_matrixs = [sim / loss_fn.temperature for sim in similarity_matrixs]
        else:
            raise ValueError("Unsupported loss function for visualization.")

        # Generate visualizations for late interaction aggregators
        for agg_idx, (video_agg, music_agg) in enumerate(zip(loss_fn.video_aggregators, loss_fn.music_aggregators)):
            if isinstance(video_agg, LateInteractionAggregator) and isinstance(music_agg, LateInteractionAggregator):
                print(f"Creating visualizations for aggregator {agg_idx}...")

                # Create visualization directory
                viz_dir = os.path.join(
                    self.visualization_config.get('output_dir', 'visualizations'),
                    f'aggregator_{agg_idx}'
                )

                # Get aggregator configuration
                aggregator_config = {
                    'aggregation': video_agg.aggregation,
                    'aggregation_temperature': getattr(video_agg, 'aggregation_temperature', None),
                    'top_k': getattr(video_agg, 'top_k', None),
                    'use_span_mask': video_agg.use_span_mask,
                }

                # Create detailed visualizations
                visualize_late_interaction_similarity_with_spans(
                    video_features=global_video_features,
                    music_features=global_music_features,
                    video_masks=global_video_masks,
                    music_masks=global_music_masks,
                    music_span_masks=global_music_span_masks if aggregator_config['use_span_mask'] else None,
                    spans_target=global_spans_target,
                    music_ids=global_music_ids,
                    similarity_matrix=similarity_matrixs[agg_idx],
                    aggregator_config=aggregator_config,
                    save_dir=viz_dir,
                    positive_pairs_only=self.visualization_config.get('positive_pairs_only', True),
                    max_pairs_to_plot=self.visualization_config.get('max_pairs_to_plot', 10),
                    figsize=self.visualization_config.get('figsize', (15, 8)),
                )

                # Create analysis
                stats = analyze_late_interaction_patterns(
                    video_features=global_video_features,
                    music_features=global_music_features,
                    video_masks=global_video_masks,
                    music_masks=global_music_masks,
                    music_span_masks=global_music_span_masks if aggregator_config['use_span_mask'] else None,
                    spans_target=global_spans_target,
                    music_ids=global_music_ids,
                    similarity_matrix=similarity_matrixs[agg_idx],
                    save_dir=viz_dir,
                )

                print(f"Visualization completed for aggregator {agg_idx}")
                print(f"  - Average positive pair similarity: {stats.get('avg_positive_similarity', 'N/A'):.4f}")
                if 'avg_gt_span_similarity' in stats:
                    print(f"  - Average GT span similarity: {stats['avg_gt_span_similarity']:.4f}")
                if 'avg_outside_span_similarity' in stats:
                    print(f"  - Average outside span similarity: {stats['avg_outside_span_similarity']:.4f}")

        # Compute standard evaluation metrics
        sim_matrix = torch.stack(similarity_matrixs).sum(dim=0).detach().cpu().numpy()

        from sv2m.criterion import retrieval_metrics, calculate_miou
        retrieval, _, _ = retrieval_metrics(sim_matrix, all_music_ids_list=global_music_ids)

        miou = calculate_miou(
            torch.cat(all_predicted_spans, dim=0).to(self.device),
            torch.cat(all_spans_target, dim=0).to(self.device),
            dataloader.dataset.max_music_duration
        )

        # Prepare metrics
        metrics = {
            "evaluation_loss": float(total_evaluation_loss / max(num_evaluation_batches, 1)),
            "evaluation_miou": float(miou),
        }

        for key, value in retrieval.items():
            if isinstance(value, (int, float)):
                metrics[f"evaluation_{key}"] = float(value)

        return metrics

    def run(self, evaluation_dataloader=None) -> Dict[str, Any]:
        """Run evaluation with visualization."""
        if evaluation_dataloader is None:
            evaluation_dataloader = self.evaluation_dataloader

        evaluate_config = self.config.evaluate

        # Run evaluation with visualization
        metrics = self.evaluate_with_visualization(evaluation_dataloader)

        # Log and save metrics as usual
        self.log_metrics(metrics)
        self.save_scores(metrics, path=evaluate_config.output.scores)

        if self.writer is not None:
            self.writer.close()

        return metrics

    @classmethod
    def build_from_config_with_visualization(
        cls,
        config: DictConfig,
        visualization_config: Optional[Dict[str, Any]] = None
    ) -> "MaDEEvaluatorWithVisualization":
        """Build evaluator with visualization from Hydra config."""
        # Use parent's build method and then wrap with visualization
        base_evaluator = super(MaDEEvaluatorWithVisualization, cls).build_from_config(config)

        # Create new instance with visualization
        return cls(
            evaluation_dataloader=base_evaluator.evaluation_dataloader,
            model=base_evaluator.model,
            config=base_evaluator.config,
            device=base_evaluator.device,
            visualization_config=visualization_config,
        )


@sv2m.main()
def main(config: DictConfig) -> None:
    """Main function with visualization support."""

    # Configuration for visualization
    visualization_config = {
        'output_dir': config.get('visualization', {}).get('output_dir', 'visualization_output'),
        'positive_pairs_only': config.get('visualization', {}).get('positive_pairs_only', True),
        'max_pairs_to_plot': config.get('visualization', {}).get('max_pairs_to_plot', 10),
        'max_batches': config.get('visualization', {}).get('max_batches', float('inf')),
        'figsize': config.get('visualization', {}).get('figsize', (15, 8)),
    }

    print("Building evaluator with visualization...")
    evaluator = MaDEEvaluatorWithVisualization.build_from_config_with_visualization(
        config, visualization_config
    )

    print("Starting evaluation with late interaction visualization...")
    results = evaluator.run()

    print("Evaluation completed!")
    print(f"Results: {results}")


if __name__ == "__main__":
    main()