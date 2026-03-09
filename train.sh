#!/bin/bash
export DATASET_NAME=normal-fix_0

# export POLICY=act
# uv run lerobot-train --dataset.repo_id=local/${DATASET_NAME} --dataset.root=datasets/${DATASET_NAME} --policy.type=$POLICY --output_dir=outputs/train/${POLICY}_${DATASET_NAME} --job_name=${POLICY}_${DATASET_NAME} --policy.device=cuda --policy.push_to_hub=false --wandb.enable=true --wandb.disable_artifact=true --dataset.video_backend=pyav --batch_size=8 --steps=200000
# export POLICY=diffusion
# uv run lerobot-train --dataset.repo_id=local/${DATASET_NAME} --dataset.root=datasets/${DATASET_NAME} --policy.type=$POLICY --output_dir=outputs/train/${POLICY}_${DATASET_NAME} --job_name=${POLICY}_${DATASET_NAME} --policy.device=cuda --policy.push_to_hub=false --wandb.enable=true --wandb.disable_artifact=true --dataset.video_backend=pyav --batch_size=64 --steps=560000
# export POLICY=vqbet
# uv run lerobot-train --dataset.repo_id=local/${DATASET_NAME} --dataset.root=datasets/${DATASET_NAME} --policy.type=$POLICY --output_dir=outputs/train/${POLICY}_${DATASET_NAME} --job_name=${POLICY}_${DATASET_NAME} --policy.device=cuda --policy.push_to_hub=false --wandb.enable=true --wandb.disable_artifact=true --dataset.video_backend=pyav --batch_size=32 --steps=560000
export POLICY=pi0
uv run lerobot-train --dataset.repo_id=local/${DATASET_NAME} --dataset.root=datasets/${DATASET_NAME} --policy.type=$POLICY --output_dir=outputs/train/${POLICY}_${DATASET_NAME} --job_name=${POLICY}_${DATASET_NAME} --policy.device=cuda --policy.push_to_hub=false --wandb.enable=true --wandb.disable_artifact=true --dataset.video_backend=pyav --batch_size=32 --steps=560000 --policy.pretrained_path=lerobot/pi0_base --policy.compile_model=true --policy.gradient_checkpointing=true --policy.dtype=bfloat16 --policy.freeze_vision_encoder=false --policy.train_expert_only=false
