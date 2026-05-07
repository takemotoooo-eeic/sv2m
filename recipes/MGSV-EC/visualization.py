"""
Visualization utilities for music-video retrieval evaluation.

This module provides functions to visualize similarity matrices for late interaction,
particularly for positive music-video pairs with ground truth spans.
"""

import os
from typing import Optional, List, Dict, Any
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path


def visualize_late_interaction_similarity_with_spans(
    video_features: torch.Tensor,
    music_features: torch.Tensor,
    video_masks: torch.Tensor,
    music_masks: torch.Tensor,
    music_span_masks: Optional[torch.Tensor],
    spans_target: torch.Tensor,
    music_ids: List[str],
    similarity_matrix: torch.Tensor,
    aggregator_config: Dict[str, Any],
    save_dir: str,
    positive_pairs_only: bool = True,
    max_pairs_to_plot: int = 10,
    figsize: tuple = (15, 8),
) -> None:
    """
    Visualize late interaction similarity matrices for positive music-video pairs.

    Args:
        video_features: Video features [B_v, T_v, D]
        music_features: Music features [B_m, T_m, D]
        video_masks: Video masks [B_v, T_v]
        music_masks: Music masks [B_m, T_m]
        music_span_masks: Music span masks [B_m, T_m] (optional)
        spans_target: Ground truth spans [B_m, 2] (start, end frames)
        music_ids: List of music IDs [B_m]
        similarity_matrix: Final similarity matrix [B_v, B_m]
        aggregator_config: Configuration of the late interaction aggregator
        save_dir: Directory to save visualizations
        positive_pairs_only: Whether to only plot positive pairs (diagonal)
        max_pairs_to_plot: Maximum number of pairs to plot
        figsize: Figure size for each plot
    """
    os.makedirs(save_dir, exist_ok=True)

    # Convert to numpy for easier processing
    video_features = video_features.detach().cpu()
    music_features = music_features.detach().cpu()
    video_masks = video_masks.detach().cpu().bool()
    music_masks = music_masks.detach().cpu().bool()
    if music_span_masks is not None:
        music_span_masks = music_span_masks.detach().cpu().bool()
    spans_target = spans_target.detach().cpu()
    similarity_matrix = similarity_matrix.detach().cpu().numpy()

    B_v, B_m = similarity_matrix.shape

    if positive_pairs_only:
        # Only plot diagonal pairs (positive pairs)
        pairs_to_plot = min(min(B_v, B_m), max_pairs_to_plot)
        pair_indices = [(i, i) for i in range(pairs_to_plot)]
    else:
        # Plot all pairs up to max_pairs_to_plot
        all_pairs = [(i, j) for i in range(B_v) for j in range(B_m)]
        pairs_to_plot = min(len(all_pairs), max_pairs_to_plot)
        pair_indices = all_pairs[:pairs_to_plot]

    print(f"Plotting {len(pair_indices)} music-video pairs...")

    for plot_idx, (v_idx, m_idx) in enumerate(pair_indices):
        # Get features for this pair
        v_feat = video_features[v_idx]  # [T_v, D]
        m_feat = music_features[m_idx]  # [T_m, D]
        v_mask = video_masks[v_idx]     # [T_v]
        m_mask = music_masks[m_idx]     # [T_m]
        span_target = spans_target[m_idx]  # [2]
        music_id = music_ids[m_idx]

        # Use span mask if available
        if music_span_masks is not None:
            m_span_mask = music_span_masks[m_idx]  # [T_m]
        else:
            m_span_mask = m_mask

        # Compute token-level similarity matrix for this pair
        token_similarity = compute_token_similarity(v_feat, m_feat, v_mask, m_mask, m_span_mask)

        # Create the plot
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'Late Interaction Analysis - Video {v_idx} × Music {m_idx} (ID: {music_id})\n'
                    f'Final Similarity: {similarity_matrix[v_idx, m_idx]:.4f}', fontsize=14)

        # Plot 1: Token-level similarity matrix
        ax1 = axes[0, 0]
        im1 = ax1.imshow(token_similarity, cmap='RdYlBu_r', aspect='auto')
        ax1.set_title('Token-level Similarities')
        ax1.set_xlabel('Music Tokens (Time →)')
        ax1.set_ylabel('Video Tokens (Time →)')
        plt.colorbar(im1, ax=ax1, shrink=0.6)

        # Add ground truth span overlay
        span_start, span_end = span_target[0].item(), span_target[1].item()
        if span_start >= 0 and span_end > span_start:  # Valid span
            rect = patches.Rectangle((span_start, 0), span_end - span_start,
                                   token_similarity.shape[0], linewidth=2,
                                   edgecolor='red', facecolor='none', linestyle='--')
            ax1.add_patch(rect)
            ax1.text(span_start, -2, f'GT: [{span_start}:{span_end}]',
                    color='red', fontweight='bold', fontsize=10)

        # Plot 2: Aggregated scores per video token (max over music tokens)
        ax2 = axes[0, 1]
        if aggregator_config.get("aggregation", "max") == "max":
            video_token_scores = np.max(token_similarity, axis=1)
            title_suffix = "Max Aggregation"
        elif aggregator_config.get("aggregation") == "log_sum":
            temp = aggregator_config.get("aggregation_temperature", 1.0)
            video_token_scores = temp * np.log(np.sum(np.exp(token_similarity / temp), axis=1))
            title_suffix = f"LogSum Aggregation (τ={temp})"
        elif aggregator_config.get("aggregation") == "top_k":
            k = aggregator_config.get("top_k", 3)
            video_token_scores = np.mean(np.partition(token_similarity, -k, axis=1)[:, -k:], axis=1)
            title_suffix = f"Top-{k} Aggregation"
        else:
            video_token_scores = np.max(token_similarity, axis=1)
            title_suffix = "Max Aggregation (default)"

        ax2.plot(video_token_scores, 'b-', linewidth=2)
        ax2.set_title(f'Video Token Scores ({title_suffix})')
        ax2.set_xlabel('Video Token Index (Time →)')
        ax2.set_ylabel('Aggregated Similarity Score')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Music token attention/activation
        ax3 = axes[1, 0]
        music_attention = np.max(token_similarity, axis=0)
        ax3.bar(range(len(music_attention)), music_attention, color='orange', alpha=0.7)
        ax3.set_title('Music Token Max Similarity')
        ax3.set_xlabel('Music Token Index (Time →)')
        ax3.set_ylabel('Max Similarity Score')

        # Highlight ground truth span
        if span_start >= 0 and span_end > span_start and span_end <= len(music_attention):
            ax3.axvspan(span_start, span_end, alpha=0.3, color='red', label=f'GT Span [{span_start}:{span_end}]')
            ax3.legend()

        # Plot 4: Similarity evolution over time
        ax4 = axes[1, 1]

        # Show average similarity in ground truth span vs outside
        if span_start >= 0 and span_end > span_start:
            gt_span_sim = token_similarity[:, span_start:span_end].mean()
            outside_span_mask = np.ones(token_similarity.shape[1], dtype=bool)
            outside_span_mask[span_start:span_end] = False
            outside_span_sim = token_similarity[:, outside_span_mask].mean() if outside_span_mask.any() else 0

            bars = ax4.bar(['GT Span', 'Outside Span'], [gt_span_sim, outside_span_sim],
                          color=['red', 'gray'], alpha=0.7)
            ax4.set_title('Average Similarity: GT Span vs Outside')
            ax4.set_ylabel('Average Similarity')

            # Add values on bars
            for bar, val in zip(bars, [gt_span_sim, outside_span_sim]):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{val:.3f}', ha='center', va='bottom')
        else:
            ax4.text(0.5, 0.5, 'No valid GT span', ha='center', va='center',
                    transform=ax4.transAxes, fontsize=14)
            ax4.set_title('Average Similarity: GT Span vs Outside')

        plt.tight_layout()

        # Save the plot
        filename = f'late_interaction_pair_{plot_idx:03d}_v{v_idx}_m{m_idx}_{music_id}.png'
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved visualization {plot_idx + 1}/{len(pair_indices)}: {filepath}")

    print(f"All visualizations saved to: {save_dir}")


def compute_token_similarity(
    video_features: torch.Tensor,
    music_features: torch.Tensor,
    video_mask: torch.Tensor,
    music_mask: torch.Tensor,
    music_span_mask: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """
    Compute token-level similarity matrix between video and music features.

    Args:
        video_features: Video features [T_v, D]
        music_features: Music features [T_m, D]
        video_mask: Video mask [T_v]
        music_mask: Music mask [T_m]
        music_span_mask: Music span mask [T_m] (optional)

    Returns:
        Token similarity matrix [T_v, T_m]
    """
    import torch.nn.functional as F

    # Normalize features
    video_features = F.normalize(video_features, p=2, dim=-1)
    music_features = F.normalize(music_features, p=2, dim=-1)

    # Compute similarity
    similarity = torch.matmul(video_features, music_features.T)  # [T_v, T_m]

    # Apply masks
    if music_span_mask is not None:
        similarity = similarity.masked_fill(~music_span_mask[None, :], float('-inf'))
    else:
        similarity = similarity.masked_fill(~music_mask[None, :], float('-inf'))

    similarity = similarity.masked_fill(~video_mask[:, None], 0.0)

    return similarity.numpy()


def create_summary_plot(
    similarity_matrices: List[np.ndarray],
    music_ids: List[str],
    spans_target: torch.Tensor,
    save_path: str,
    title: str = "Late Interaction Similarity Matrices Summary"
) -> None:
    """
    Create a summary plot showing multiple similarity matrices.

    Args:
        similarity_matrices: List of similarity matrices for positive pairs
        music_ids: List of music IDs
        spans_target: Ground truth spans [B, 2]
        save_path: Path to save the summary plot
        title: Title for the plot
    """
    num_pairs = len(similarity_matrices)
    if num_pairs == 0:
        return

    # Create subplots
    cols = min(4, num_pairs)
    rows = (num_pairs + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    if num_pairs == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(title, fontsize=16)

    for i in range(num_pairs):
        row, col = i // cols, i % cols
        ax = axes[row, col] if rows > 1 else axes[col]

        sim_matrix = similarity_matrices[i]
        music_id = music_ids[i] if i < len(music_ids) else f"Music_{i}"
        span = spans_target[i] if i < len(spans_target) else torch.tensor([0, 0])

        # Plot similarity matrix
        im = ax.imshow(sim_matrix, cmap='RdYlBu_r', aspect='auto')
        ax.set_title(f'Pair {i}: {music_id}\nGT: [{span[0]:.0f}:{span[1]:.0f}]', fontsize=10)
        ax.set_xlabel('Music Tokens')
        ax.set_ylabel('Video Tokens')

        # Add ground truth span
        span_start, span_end = span[0].item(), span[1].item()
        if span_start >= 0 and span_end > span_start:
            rect = patches.Rectangle((span_start, 0), span_end - span_start,
                                   sim_matrix.shape[0], linewidth=1.5,
                                   edgecolor='red', facecolor='none')
            ax.add_patch(rect)

    # Remove empty subplots
    for i in range(num_pairs, rows * cols):
        row, col = i // cols, i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.remove()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Summary plot saved to: {save_path}")


def analyze_late_interaction_patterns(
    video_features: torch.Tensor,
    music_features: torch.Tensor,
    video_masks: torch.Tensor,
    music_masks: torch.Tensor,
    music_span_masks: Optional[torch.Tensor],
    spans_target: torch.Tensor,
    music_ids: List[str],
    similarity_matrix: torch.Tensor,
    save_dir: str,
) -> Dict[str, Any]:
    """
    Analyze patterns in late interaction similarities and return statistics.

    Args:
        video_features: Video features [B_v, T_v, D]
        music_features: Music features [B_m, T_m, D]
        video_masks: Video masks [B_v, T_v]
        music_masks: Music masks [B_m, T_m]
        music_span_masks: Music span masks [B_m, T_m] (optional)
        spans_target: Ground truth spans [B_m, 2]
        music_ids: List of music IDs [B_m]
        similarity_matrix: Final similarity matrix [B_v, B_m]
        save_dir: Directory to save analysis results

    Returns:
        Dictionary containing analysis statistics
    """
    os.makedirs(save_dir, exist_ok=True)

    B_v, B_m = similarity_matrix.shape[0], similarity_matrix.shape[1]
    stats = {
        "total_pairs": B_v * B_m,
        "positive_pairs": min(B_v, B_m),
        "gt_span_coverage": [],
        "outside_span_similarity": [],
        "positive_pair_similarities": [],
    }

    # Analyze positive pairs (diagonal)
    positive_token_similarities = []

    for i in range(min(B_v, B_m)):
        v_feat = video_features[i]
        m_feat = music_features[i]
        v_mask = video_masks[i].bool()
        m_mask = music_masks[i].bool()
        span_target = spans_target[i]

        if music_span_masks is not None:
            m_span_mask = music_span_masks[i].bool()
        else:
            m_span_mask = m_mask

        # Compute token similarity for this positive pair
        token_sim = compute_token_similarity(v_feat, m_feat, v_mask, m_mask, m_span_mask)
        positive_token_similarities.append(token_sim)

        # Analyze GT span coverage
        span_start, span_end = span_target[0].item(), span_target[1].item()
        if span_start >= 0 and span_end > span_start and span_end <= token_sim.shape[1]:
            gt_span_sim = token_sim[:, span_start:span_end].mean()
            stats["gt_span_coverage"].append(gt_span_sim)

            # Similarity outside GT span
            outside_mask = np.ones(token_sim.shape[1], dtype=bool)
            outside_mask[span_start:span_end] = False
            if outside_mask.any():
                outside_sim = token_sim[:, outside_mask].mean()
                stats["outside_span_similarity"].append(outside_sim)

        stats["positive_pair_similarities"].append(similarity_matrix[i, i].item())

    # Create summary statistics
    if stats["gt_span_coverage"]:
        stats["avg_gt_span_similarity"] = np.mean(stats["gt_span_coverage"])
        stats["std_gt_span_similarity"] = np.std(stats["gt_span_coverage"])

    if stats["outside_span_similarity"]:
        stats["avg_outside_span_similarity"] = np.mean(stats["outside_span_similarity"])
        stats["std_outside_span_similarity"] = np.std(stats["outside_span_similarity"])

    stats["avg_positive_similarity"] = np.mean(stats["positive_pair_similarities"])
    stats["std_positive_similarity"] = np.std(stats["positive_pair_similarities"])

    # Create summary visualizations
    if positive_token_similarities:
        create_summary_plot(
            positive_token_similarities[:16],  # Limit to first 16 pairs
            music_ids,
            spans_target,
            os.path.join(save_dir, "summary_positive_pairs.png"),
            "Late Interaction: Positive Pairs Token Similarities"
        )

    # Save statistics
    import json
    stats_path = os.path.join(save_dir, "late_interaction_stats.json")
    with open(stats_path, 'w') as f:
        json.dump({k: v if not isinstance(v, np.ndarray) else v.tolist()
                  for k, v in stats.items()}, f, indent=2)

    print(f"Analysis statistics saved to: {stats_path}")
    return stats