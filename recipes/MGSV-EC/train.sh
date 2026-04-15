#!/usr/bin/env bash

set -eu
set -o pipefail

tag=""

video_dir="/path/to/video_dir"

exp_root="exp"
tensorboard_root="tensorboard"

dataloader="himv61k"
pipeline="mvpt"
train="mvpt_himv61k"
model="modified_mvpt_finetuning"
optimizer="musreel_pretraining"

. ../_common/parse_options.sh || exit 1;

if [ -z "${video_dir}" ]; then
    echo "video_dir is not set."
    exit 1;
fi

if [ -z "${tag}" ]; then
    tag="$(date +"%Y%m%d-%H%M%S")"
fi

cmd=$(sv2m-parse-run-command)

exp_dir="${exp_root}/${tag}"
tensorboard_dir="${tensorboard_root}/${tag}"

${cmd} local/train.py \
hydra.run.dir="${exp_dir}/logs/$(date +"%Y%m%d-%H%M%S")" \
dataloader="${dataloader}" \
pipeline="${pipeline}" \
train="${train}" \
model="${model}" \
optimizer="${optimizer}" \
dataloader.train.dataset.video_dir="${video_dir}" \
train.output.exp_dir="${exp_dir}" \
train.output.tensorboard_dir="${tensorboard_dir}"