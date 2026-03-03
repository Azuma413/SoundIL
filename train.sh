#!/bin/bash

export POLICY=diffusion

export DATASET_NAME=sound-m4-f10-s0-p0_0
uv run lerobot-train --dataset.repo_id=local/${DATASET_NAME} --dataset.root=datasets/${DATASET_NAME} --policy.type=$POLICY --output_dir=outputs/train/${POLICY}_${DATASET_NAME} --job_name=${POLICY}_${DATASET_NAME} --policy.device=cuda --policy.push_to_hub=false --wandb.enable=true --wandb.disable_artifact=true --batch_size=8 --steps=100000
export DATASET_NAME=sound-m4-f10-s0-p0_1
uv run lerobot-train --dataset.repo_id=local/${DATASET_NAME} --dataset.root=datasets/${DATASET_NAME} --policy.type=$POLICY --output_dir=outputs/train/${POLICY}_${DATASET_NAME} --job_name=${POLICY}_${DATASET_NAME} --policy.device=cuda --policy.push_to_hub=false --wandb.enable=true --wandb.disable_artifact=true --batch_size=8 --steps=100000
export DATASET_NAME=sound-m4-f10-s0-p0_2
uv run lerobot-train --dataset.repo_id=local/${DATASET_NAME} --dataset.root=datasets/${DATASET_NAME} --policy.type=$POLICY --output_dir=outputs/train/${POLICY}_${DATASET_NAME} --job_name=${POLICY}_${DATASET_NAME} --policy.device=cuda --policy.push_to_hub=false --wandb.enable=true --wandb.disable_artifact=true --batch_size=8 --steps=100000
export DATASET_NAME=sound-m4-f10-s1-p0_0
uv run lerobot-train --dataset.repo_id=local/${DATASET_NAME} --dataset.root=datasets/${DATASET_NAME} --policy.type=$POLICY --output_dir=outputs/train/${POLICY}_${DATASET_NAME} --job_name=${POLICY}_${DATASET_NAME} --policy.device=cuda --policy.push_to_hub=false --wandb.enable=true --wandb.disable_artifact=true --batch_size=8 --steps=100000
export DATASET_NAME=sound-m4-f10-s1-p0_1
uv run lerobot-train --dataset.repo_id=local/${DATASET_NAME} --dataset.root=datasets/${DATASET_NAME} --policy.type=$POLICY --output_dir=outputs/train/${POLICY}_${DATASET_NAME} --job_name=${POLICY}_${DATASET_NAME} --policy.device=cuda --policy.push_to_hub=false --wandb.enable=true --wandb.disable_artifact=true --batch_size=8 --steps=100000
export DATASET_NAME=sound-m4-f10-s1-p0_2
uv run lerobot-train --dataset.repo_id=local/${DATASET_NAME} --dataset.root=datasets/${DATASET_NAME} --policy.type=$POLICY --output_dir=outputs/train/${POLICY}_${DATASET_NAME} --job_name=${POLICY}_${DATASET_NAME} --policy.device=cuda --policy.push_to_hub=false --wandb.enable=true --wandb.disable_artifact=true --batch_size=8 --steps=100000
export DATASET_NAME=sound-m4-f10-s2-p0_0
uv run lerobot-train --dataset.repo_id=local/${DATASET_NAME} --dataset.root=datasets/${DATASET_NAME} --policy.type=$POLICY --output_dir=outputs/train/${POLICY}_${DATASET_NAME} --job_name=${POLICY}_${DATASET_NAME} --policy.device=cuda --policy.push_to_hub=false --wandb.enable=true --wandb.disable_artifact=true --batch_size=8 --steps=100000
export DATASET_NAME=sound-m4-f10-s2-p0_1
uv run lerobot-train --dataset.repo_id=local/${DATASET_NAME} --dataset.root=datasets/${DATASET_NAME} --policy.type=$POLICY --output_dir=outputs/train/${POLICY}_${DATASET_NAME} --job_name=${POLICY}_${DATASET_NAME} --policy.device=cuda --policy.push_to_hub=false --wandb.enable=true --wandb.disable_artifact=true --batch_size=8 --steps=100000
export DATASET_NAME=sound-m4-f10-s2-p0_2
uv run lerobot-train --dataset.repo_id=local/${DATASET_NAME} --dataset.root=datasets/${DATASET_NAME} --policy.type=$POLICY --output_dir=outputs/train/${POLICY}_${DATASET_NAME} --job_name=${POLICY}_${DATASET_NAME} --policy.device=cuda --policy.push_to_hub=false --wandb.enable=true --wandb.disable_artifact=true --batch_size=8 --steps=100000