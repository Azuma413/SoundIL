#!/bin/bash

export POLICY=act

export DATASET_NAME=sound-m3-f15-s2-p0_0
uv run lerobot-train --dataset.repo_id=local/${DATASET_NAME} --dataset.root=datasets/${DATASET_NAME} --policy.type=$POLICY --output_dir=outputs/train/${POLICY}_${DATASET_NAME} --job_name=${POLICY}_${DATASET_NAME} --policy.device=cuda --policy.push_to_hub=false --wandb.enable=true --wandb.disable_artifact=true --batch_size=8 --steps=100000
export DATASET_NAME=sound-m3-f10-s2-p0_0
uv run lerobot-train --dataset.repo_id=local/${DATASET_NAME} --dataset.root=datasets/${DATASET_NAME} --policy.type=$POLICY --output_dir=outputs/train/${POLICY}_${DATASET_NAME} --job_name=${POLICY}_${DATASET_NAME} --policy.device=cuda --policy.push_to_hub=false --wandb.enable=true --wandb.disable_artifact=true --batch_size=8 --steps=100000
export DATASET_NAME=sound-m3-f5-s2-p0_0
uv run lerobot-train --dataset.repo_id=local/${DATASET_NAME} --dataset.root=datasets/${DATASET_NAME} --policy.type=$POLICY --output_dir=outputs/train/${POLICY}_${DATASET_NAME} --job_name=${POLICY}_${DATASET_NAME} --policy.device=cuda --policy.push_to_hub=false --wandb.enable=true --wandb.disable_artifact=true --batch_size=8 --steps=100000
export DATASET_NAME=sound-m3-f3-s2-p0_0
uv run lerobot-train --dataset.repo_id=local/${DATASET_NAME} --dataset.root=datasets/${DATASET_NAME} --policy.type=$POLICY --output_dir=outputs/train/${POLICY}_${DATASET_NAME} --job_name=${POLICY}_${DATASET_NAME} --policy.device=cuda --policy.push_to_hub=false --wandb.enable=true --wandb.disable_artifact=true --batch_size=8 --steps=100000
export DATASET_NAME=sound-m3-f1-s2-p0_0
uv run lerobot-train --dataset.repo_id=local/${DATASET_NAME} --dataset.root=datasets/${DATASET_NAME} --policy.type=$POLICY --output_dir=outputs/train/${POLICY}_${DATASET_NAME} --job_name=${POLICY}_${DATASET_NAME} --policy.device=cuda --policy.push_to_hub=false --wandb.enable=true --wandb.disable_artifact=true --batch_size=8 --steps=100000
