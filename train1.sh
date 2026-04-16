#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
export DATASET_NAME=soundShake-m4-f10-s0-p0_0
# soundShakeとsoundSimは200000ステップ．
export STEPS=200000
export SAVE_FREQ=20000

# export POLICY=act
# uv run lerobot-train --dataset.repo_id=local/${DATASET_NAME} --dataset.root=datasets/${DATASET_NAME} --policy.type=$POLICY --output_dir=outputs/train/${POLICY}_${DATASET_NAME}_seed${SEED} --job_name=${POLICY}_${DATASET_NAME}_seed${SEED} --policy.device=cuda --policy.push_to_hub=false --wandb.enable=true --wandb.disable_artifact=true --dataset.video_backend=pyav --batch_size=8 --steps=$STEPS --save_freq=$SAVE_FREQ --seed=$SEED
# export POLICY=diffusion
# uv run lerobot-train --dataset.repo_id=local/${DATASET_NAME} --dataset.root=datasets/${DATASET_NAME} --policy.type=$POLICY --output_dir=outputs/train/${POLICY}_${DATASET_NAME}_seed${SEED} --job_name=${POLICY}_${DATASET_NAME}_seed${SEED} --policy.device=cuda --policy.push_to_hub=false --wandb.enable=true --wandb.disable_artifact=true --dataset.video_backend=pyav --batch_size=32 --steps=$STEPS --save_freq=$SAVE_FREQ --seed=$SEED --policy.use_separate_rgb_encoder_per_camera=true
# export POLICY=vqbet
# uv run lerobot-train --dataset.repo_id=local/${DATASET_NAME} --dataset.root=datasets/${DATASET_NAME} --policy.type=$POLICY --output_dir=outputs/train/${POLICY}_${DATASET_NAME}_seed${SEED} --job_name=${POLICY}_${DATASET_NAME}_seed${SEED} --policy.device=cuda --policy.push_to_hub=false --wandb.enable=true --wandb.disable_artifact=true --dataset.video_backend=pyav --batch_size=32 --steps=$STEPS --save_freq=$SAVE_FREQ --seed=$SEED
export POLICY=pi0
# uv run lerobot-train --dataset.repo_id=local/${DATASET_NAME} --dataset.root=datasets/${DATASET_NAME} --policy.type=$POLICY --output_dir=outputs/train/${POLICY}_${DATASET_NAME}_seed${SEED} --job_name=${POLICY}_${DATASET_NAME}_seed${SEED} --policy.device=cuda --policy.push_to_hub=false --wandb.enable=true --wandb.disable_artifact=true --dataset.video_backend=pyav --batch_size=4 --steps=$STEPS --save_freq=$SAVE_FREQ --seed=$SEED --policy.pretrained_path=lerobot/pi0_base --policy.gradient_checkpointing=true --policy.dtype=bfloat16

export SEED=0
uv run lerobot-train --dataset.repo_id=local/${DATASET_NAME} --dataset.root=datasets/${DATASET_NAME} --policy.type=$POLICY --output_dir=outputs/train/${POLICY}_${DATASET_NAME}_seed${SEED} --job_name=${POLICY}_${DATASET_NAME}_seed${SEED} --policy.device=cuda --policy.push_to_hub=false --wandb.enable=true --wandb.disable_artifact=true --dataset.video_backend=pyav --batch_size=4 --steps=$STEPS --save_freq=$SAVE_FREQ --seed=$SEED --policy.pretrained_path=lerobot/pi0_base --policy.gradient_checkpointing=true --policy.dtype=bfloat16
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step $STEPS
export SEED=1
uv run lerobot-train --dataset.repo_id=local/${DATASET_NAME} --dataset.root=datasets/${DATASET_NAME} --policy.type=$POLICY --output_dir=outputs/train/${POLICY}_${DATASET_NAME}_seed${SEED} --job_name=${POLICY}_${DATASET_NAME}_seed${SEED} --policy.device=cuda --policy.push_to_hub=false --wandb.enable=true --wandb.disable_artifact=true --dataset.video_backend=pyav --batch_size=4 --steps=$STEPS --save_freq=$SAVE_FREQ --seed=$SEED --policy.pretrained_path=lerobot/pi0_base --policy.gradient_checkpointing=true --policy.dtype=bfloat16
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step $STEPS
export SEED=2
uv run lerobot-train --dataset.repo_id=local/${DATASET_NAME} --dataset.root=datasets/${DATASET_NAME} --policy.type=$POLICY --output_dir=outputs/train/${POLICY}_${DATASET_NAME}_seed${SEED} --job_name=${POLICY}_${DATASET_NAME}_seed${SEED} --policy.device=cuda --policy.push_to_hub=false --wandb.enable=true --wandb.disable_artifact=true --dataset.video_backend=pyav --batch_size=4 --steps=$STEPS --save_freq=$SAVE_FREQ --seed=$SEED --policy.pretrained_path=lerobot/pi0_base --policy.gradient_checkpointing=true --policy.dtype=bfloat16
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step $STEPS

export DATASET_NAME=soundShake-m4-f10-s1-p0_0
export SEED=0
uv run lerobot-train --dataset.repo_id=local/${DATASET_NAME} --dataset.root=datasets/${DATASET_NAME} --policy.type=$POLICY --output_dir=outputs/train/${POLICY}_${DATASET_NAME}_seed${SEED} --job_name=${POLICY}_${DATASET_NAME}_seed${SEED} --policy.device=cuda --policy.push_to_hub=false --wandb.enable=true --wandb.disable_artifact=true --dataset.video_backend=pyav --batch_size=4 --steps=$STEPS --save_freq=$SAVE_FREQ --seed=$SEED --policy.pretrained_path=lerobot/pi0_base --policy.gradient_checkpointing=true --policy.dtype=bfloat16
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step $STEPS
export SEED=1
uv run lerobot-train --dataset.repo_id=local/${DATASET_NAME} --dataset.root=datasets/${DATASET_NAME} --policy.type=$POLICY --output_dir=outputs/train/${POLICY}_${DATASET_NAME}_seed${SEED} --job_name=${POLICY}_${DATASET_NAME}_seed${SEED} --policy.device=cuda --policy.push_to_hub=false --wandb.enable=true --wandb.disable_artifact=true --dataset.video_backend=pyav --batch_size=4 --steps=$STEPS --save_freq=$SAVE_FREQ --seed=$SEED --policy.pretrained_path=lerobot/pi0_base --policy.gradient_checkpointing=true --policy.dtype=bfloat16
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step $STEPS
export SEED=2
uv run lerobot-train --dataset.repo_id=local/${DATASET_NAME} --dataset.root=datasets/${DATASET_NAME} --policy.type=$POLICY --output_dir=outputs/train/${POLICY}_${DATASET_NAME}_seed${SEED} --job_name=${POLICY}_${DATASET_NAME}_seed${SEED} --policy.device=cuda --policy.push_to_hub=false --wandb.enable=true --wandb.disable_artifact=true --dataset.video_backend=pyav --batch_size=4 --steps=$STEPS --save_freq=$SAVE_FREQ --seed=$SEED --policy.pretrained_path=lerobot/pi0_base --policy.gradient_checkpointing=true --policy.dtype=bfloat16
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step $STEPS
