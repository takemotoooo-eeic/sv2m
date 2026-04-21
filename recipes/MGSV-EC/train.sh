#!/usr/bin/env bash

set -eu
set -o pipefail

tag="test-xpool"

exp_root="exp"
tensorboard_root="tensorboard"

music_feat_dir="/Users/kentakem/sv2m/features/ast_feature2p5"
video_feat_dir="/Users/kentakem/sv2m/features/vit_feature1"
csv_root="/Users/kentakem/sv2m/dataset/MGSV-EC"

dataloader="mgsvec"
train="made_mgsvec"
model="made_li"
optimizer="made"

. ../_common/parse_options.sh || exit 1;


if [ -z "${tag}" ]; then
    tag="$(date +"%Y%m%d-%H%M%S")"
fi

cmd=$(sv2m-parse-run-command)

exp_dir="${exp_root}/${tag}"
tensorboard_dir="${tensorboard_root}/${tag}"

${cmd} local/train.py \
hydra.run.dir="${exp_dir}/logs/$(date +"%Y%m%d-%H%M%S")" \
dataloader="${dataloader}" \
model="${model}" \
dataloader.train.dataset.video_feat_dir="${video_feat_dir}" \
dataloader.train.dataset.music_feat_dir="${music_feat_dir}" \
dataloader.train.dataset.csv_root="${csv_root}" \
train="${train}" \
optimizer="${optimizer}" \
train.output.exp_dir="${exp_dir}" \
train.output.tensorboard_dir="${tensorboard_dir}"
