#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

export POLICY=act
export DATASET_NAME=sound-m4-f10-s2-p0

uv run plot_tsne.py \
    --training-name ${POLICY}_${DATASET_NAME}_0_seed0 \
    --checkpoint-step last \
    --episode-num 100 \
    --hidden-reduction mean

export DATASET_NAME=soundShake-m4-f10-s2-p0

uv run plot_tsne.py \
    --training-name ${POLICY}_${DATASET_NAME}_0_seed0 \
    --checkpoint-step last \
    --episode-num 100 \
    --hidden-reduction mean

export POLICY=vqbet
export DATASET_NAME=soundDiff-m4-f10-s2-p0

uv run plot_tsne.py \
    --training-name ${POLICY}_${DATASET_NAME}_0_seed0 \
    --checkpoint-step last \
    --episode-num 100 \
    --hidden-reduction mean

export DATASET_NAME=soundSim-m4-f10-s2-p0

uv run plot_tsne.py \
    --training-name ${POLICY}_${DATASET_NAME}_0_seed0 \
    --checkpoint-step last \
    --episode-num 100 \
    --hidden-reduction mean