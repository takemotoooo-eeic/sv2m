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
    tag="$(date +"%Y%m%d-%H%M%S")"
fi

cmd=$(sv2m-parse-run-command)

exp_dir="${exp_root}/${tag}"

${cmd} local/evaluate.py \
hydra.run.dir="${exp_dir}/logs/$(date +"%Y%m%d-%H%M%S")" \
dataloader="${dataloader}" \
train="${train}" \
evaluate="${evaluate}" \
model="${model}" \
dataloader.evaluate.dataset.music_feat_dir="${music_feat_dir}" \
dataloader.evaluate.dataset.video_feat_dir="${video_feat_dir}" \
dataloader.evaluate.dataset.csv_root="${csv_root}" \
evaluate.checkpoint.pretrained_model="${pretrained_checkpoint}" \
evaluate.output.exp_dir="${exp_dir}"