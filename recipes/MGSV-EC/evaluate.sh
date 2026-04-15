#!/usr/bin/env bash

set -eu
set -o pipefail

tag=""

video_dir=""

pretrained_checkpoint=""

exp_root="exp"

dataloader="himv61k"
pipeline="mvpt"
train="mvpt_himv61k"
evaluate="mvpt_himv61k"
model="modified_mvpt_finetuning"

. ../_common/parse_options.sh || exit 1;

if [ -z "${video_dir}" ]; then
    echo "video_dir is not set."
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
pipeline="${pipeline}" \
train="${train}" \
evaluate="${evaluate}" \
model="${model}" \
dataloader.evaluate.dataset.video_dir="${video_dir}" \
evaluate.checkpoint.pretrained_model="${pretrained_checkpoint}" \
evaluate.output.exp_dir="${exp_dir}"