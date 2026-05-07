#!/usr/bin/env bash

set -eu
set -o pipefail

tag=""

music_feat_dir="/Users/kentakem/sv2m/features/ast_feature2p5"
video_feat_dir="/Users/kentakem/sv2m/features/vit_feature1"
csv_root="/Users/kentakem/sv2m/dataset/MGSV-EC"

pretrained_checkpoint="/Users/kentakem/sv2m/recipes/MGSV-EC/exp/20260421-020807/model/best_epoch.pth"

exp_root="exp"

dataloader="mgsvec"
train="made_mgsvec"
evaluate="made"
model="made"

# Visualization configuration
visualization_output_dir="visualization_output"
positive_pairs_only=true
max_pairs_to_plot=10
max_batches=5  # Limit batches for faster visualization
figsize_width=15
figsize_height=8

. ../_common/parse_options.sh || exit 1;

if [ -z "${video_feat_dir}" ]; then
    echo "video_feat_dir is not set."
    exit 1;
fi

if [ -z "${music_feat_dir}" ]; then
    echo "music_feat_dir is not set."
    exit 1;
fi

if [ -z "${csv_root}" ]; then
    echo "csv_root is not set."
    exit 1;
fi

if [ -z "${pretrained_checkpoint}" ]; then
    echo "pretrained_checkpoint is not set."
    exit 1;
fi

if [ -z "${tag}" ]; then
    tag="$(date +"%Y%m%d-%H%M%S")-viz"
fi

cmd=$(sv2m-parse-run-command)

exp_dir="${exp_root}/${tag}"

echo "Running evaluation with late interaction visualization..."
echo "Output will be saved to: ${exp_dir}"
echo "Visualization output will be saved to: ${exp_dir}/${visualization_output_dir}"

${cmd} evaluate_with_visualization.py \
hydra.run.dir="${exp_dir}/logs/$(date +"%Y%m%d-%H%M%S")" \
dataloader="${dataloader}" \
train="${train}" \
evaluate="${evaluate}" \
model="${model}" \
dataloader.evaluate.dataset.music_feat_dir="${music_feat_dir}" \
dataloader.evaluate.dataset.video_feat_dir="${video_feat_dir}" \
dataloader.evaluate.dataset.csv_root="${csv_root}" \
evaluate.checkpoint.pretrained_model="${pretrained_checkpoint}" \
evaluate.output.exp_dir="${exp_dir}" \
visualization.output_dir="${exp_dir}/${visualization_output_dir}" \
visualization.positive_pairs_only=${positive_pairs_only} \
visualization.max_pairs_to_plot=${max_pairs_to_plot} \
visualization.max_batches=${max_batches} \
visualization.figsize="[${figsize_width}, ${figsize_height}]"

echo ""
echo "Evaluation with visualization completed!"
echo "Check the following directories for results:"
echo "  - Evaluation metrics: ${exp_dir}/evaluation/"
echo "  - Visualizations: ${exp_dir}/${visualization_output_dir}/"
echo "  - Logs: ${exp_dir}/logs/"