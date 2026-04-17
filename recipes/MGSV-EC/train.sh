#!/usr/bin/env bash

set -eu
set -o pipefail

tag=""

exp_root="exp"
tensorboard_root="tensorboard"

dataloader="mgsvec"
train="made_mgsvec"
model="modified_mvpt_finetuning"
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
train="${train}" \
optimizer="${optimizer}" \
train.output.exp_dir="${exp_dir}" \
train.output.tensorboard_dir="${tensorboard_dir}"