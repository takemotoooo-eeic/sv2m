# Late Interaction Visualization for Music-Video Retrieval

This directory contains enhanced evaluation tools that provide detailed visualization of late interaction similarity matrices for positive music-video pairs.

## Overview

Late interaction is a method for computing similarity between video and music sequences by:
1. Computing token-level similarities between all video and music tokens
2. Aggregating these similarities (using max, log-sum, or top-k methods)
3. Producing a final similarity score for the video-music pair

This visualization tool helps you understand:
- How similarity evolves across time segments
- Whether the model focuses on the ground truth music intervals
- Token-level interaction patterns between video and music

## Files

- `visualization.py` - Core visualization functions
- `evaluate_with_visualization.py` - Enhanced evaluator with visualization
- `evaluate_visualize.sh` - Convenient shell script to run visualization
- `README_visualization.md` - This documentation file

## Quick Start

### 1. Basic Usage

Run the visualization evaluation:

```bash
./evaluate_visualize.sh
```

This will:
- Use the default pretrained model and dataset paths
- Generate visualizations for the first 10 positive pairs
- Process up to 5 batches (for faster execution)
- Save results to `exp/{timestamp}-viz/visualization_output/`

### 2. Custom Configuration

Modify parameters in `evaluate_visualize.sh` or pass them as arguments:

```bash
# Visualize more pairs
./evaluate_visualize.sh --max_pairs_to_plot 20

# Process more batches
./evaluate_visualize.sh --max_batches 10

# Use different model checkpoint
./evaluate_visualize.sh --pretrained_checkpoint /path/to/your/model.pth

# Save to custom directory
./evaluate_visualize.sh --visualization_output_dir my_viz_results
```

### 3. Advanced Configuration

You can also run the Python script directly with Hydra configuration:

```bash
python evaluate_with_visualization.py \
  dataloader=mgsvec \
  model=made \
  evaluate.checkpoint.pretrained_model=/path/to/model.pth \
  visualization.output_dir=custom_output \
  visualization.positive_pairs_only=true \
  visualization.max_pairs_to_plot=15 \
  visualization.max_batches=8
```

## Output Structure

The visualization creates the following directory structure:

```
exp/{timestamp}-viz/
├── evaluation/
│   └── scores.json                 # Standard evaluation metrics
├── logs/
│   └── {timestamp}/               # Hydra logs
└── visualization_output/
    ├── aggregator_0/              # Results for first aggregator
    │   ├── late_interaction_pair_000_v0_m0_{music_id}.png
    │   ├── late_interaction_pair_001_v1_m1_{music_id}.png
    │   ├── ...
    │   ├── summary_positive_pairs.png
    │   └── late_interaction_stats.json
    └── aggregator_1/              # Results for second aggregator (if exists)
        └── ...
```

## Understanding the Visualizations

### Individual Pair Plots

Each `late_interaction_pair_*.png` contains 4 subplots:

1. **Token-level Similarities** (top-left): Heatmap showing similarity between each video token and music token
   - X-axis: Music tokens (time segments)
   - Y-axis: Video tokens (time segments)
   - Red dashed rectangle: Ground truth music interval
   - Colors: Red = high similarity, Blue = low similarity

2. **Video Token Scores** (top-right): Aggregated similarity score for each video token
   - Shows how each video segment responds to the music
   - Aggregation method (max/log-sum/top-k) depends on model configuration

3. **Music Token Max Similarity** (bottom-left): Maximum similarity for each music token
   - Red highlighted region: Ground truth music interval
   - Shows which music segments are most attended to

4. **GT Span vs Outside Span** (bottom-right): Comparison of average similarities
   - Red bar: Average similarity within ground truth span
   - Gray bar: Average similarity outside ground truth span

### Summary Plots

- `summary_positive_pairs.png`: Grid view of token similarity matrices for multiple positive pairs
- `late_interaction_stats.json`: Quantitative analysis including:
  - Average similarity in ground truth spans vs outside
  - Statistics across all positive pairs
  - Coverage metrics

## Configuration Options

### Visualization Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `output_dir` | Directory to save visualizations | `visualization_output` |
| `positive_pairs_only` | Only visualize diagonal (positive) pairs | `true` |
| `max_pairs_to_plot` | Maximum number of pairs to visualize | `10` |
| `max_batches` | Maximum batches to process (for speed) | `5` |
| `figsize` | Figure size for each plot | `[15, 8]` |

### Model Parameters

The visualization works with any MaDE model that uses `LateInteractionAggregator`. The aggregation method (max, log-sum, top-k) is automatically detected from the model configuration.

## Interpreting Results

### Good Late Interaction Pattern:
- High similarity within ground truth music spans (red regions in plots)
- Low similarity outside ground truth spans
- Clear temporal alignment between video and music segments

### Signs of Issues:
- Random similarity patterns with no clear structure
- No preference for ground truth spans over other regions
- Very uniform similarities across all tokens (lack of discrimination)

### Key Metrics:
- **GT Span Similarity**: Average similarity within ground truth intervals
- **Outside Span Similarity**: Average similarity outside ground truth intervals
- **Ratio**: GT Span Similarity / Outside Span Similarity (should be > 1)

## Performance Notes

- Visualization processes the entire evaluation dataset to create comprehensive plots
- Use `max_batches` parameter to limit processing for faster iteration
- Large datasets may take several minutes to process
- Memory usage scales with batch size and sequence lengths

## Troubleshooting

### Common Issues:

1. **"No LateInteractionAggregator found"**: Your model doesn't use late interaction
   - Check that your model configuration includes `LateInteractionAggregator`

2. **Out of memory**: Large datasets or long sequences
   - Reduce `max_batches` or `max_pairs_to_plot`
   - Use smaller batch size in dataloader configuration

3. **Empty visualizations**: No positive pairs in the dataset
   - Check that your evaluation dataset contains matching video-music pairs
   - Verify `music_ids` are correctly set

4. **Import errors**: Missing dependencies
   - Ensure matplotlib and seaborn are installed: `pip install matplotlib seaborn`

### Performance Optimization:

```bash
# For quick testing (fast but fewer samples)
./evaluate_visualize.sh --max_batches 2 --max_pairs_to_plot 5

# For comprehensive analysis (slow but complete)
./evaluate_visualize.sh --max_batches inf --max_pairs_to_plot 50
```

## Example Analysis Workflow

1. **Quick Check**: Run with default settings to get overview
2. **Detailed Analysis**: Increase `max_pairs_to_plot` for more samples
3. **Pattern Investigation**: Look at individual pair plots to understand token interactions
4. **Quantitative Analysis**: Check `late_interaction_stats.json` for metrics
5. **Model Debugging**: If results are poor, examine whether:
   - Ground truth spans are being attended to
   - Temporal alignment makes sense
   - Token-level similarities have reasonable patterns

## References

This visualization tool is designed for the MaDE (Music and Video Retrieval with Disentangled Embeddings) architecture and specifically targets late interaction mechanisms in cross-modal retrieval systems.