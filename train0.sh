#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export DATASET_NAME=sound-m4-f10-s2-p0_0
# soundShakeとsoundSimは200000ステップ．
export STEPS=100000
export SAVE_FREQ=10000

export POLICY=act
# uv run lerobot-train --dataset.repo_id=local/${DATASET_NAME} --dataset.root=datasets/${DATASET_NAME} --policy.type=$POLICY --output_dir=outputs/train/${POLICY}_${DATASET_NAME}_seed${SEED} --job_name=${POLICY}_${DATASET_NAME}_seed${SEED} --policy.device=cuda --policy.push_to_hub=false --wandb.enable=true --wandb.disable_artifact=true --dataset.video_backend=pyav --batch_size=8 --steps=$STEPS --save_freq=$SAVE_FREQ --seed=$SEED
# export POLICY=diffusion
# uv run lerobot-train --dataset.repo_id=local/${DATASET_NAME} --dataset.root=datasets/${DATASET_NAME} --policy.type=$POLICY --output_dir=outputs/train/${POLICY}_${DATASET_NAME}_seed${SEED} --job_name=${POLICY}_${DATASET_NAME}_seed${SEED} --policy.device=cuda --policy.push_to_hub=false --wandb.enable=true --wandb.disable_artifact=true --dataset.video_backend=pyav --batch_size=32 --steps=$STEPS --save_freq=$SAVE_FREQ --seed=$SEED --policy.use_separate_rgb_encoder_per_camera=true
# export POLICY=vqbet
# uv run lerobot-train --dataset.repo_id=local/${DATASET_NAME} --dataset.root=datasets/${DATASET_NAME} --policy.type=$POLICY --output_dir=outputs/train/${POLICY}_${DATASET_NAME}_seed${SEED} --job_name=${POLICY}_${DATASET_NAME}_seed${SEED} --policy.device=cuda --policy.push_to_hub=false --wandb.enable=true --wandb.disable_artifact=true --dataset.video_backend=pyav --batch_size=32 --steps=$STEPS --save_freq=$SAVE_FREQ --seed=$SEED
# export POLICY=pi0
# uv run lerobot-train --dataset.repo_id=local/${DATASET_NAME} --dataset.root=datasets/${DATASET_NAME} --policy.type=$POLICY --output_dir=outputs/train/${POLICY}_${DATASET_NAME}_seed${SEED} --job_name=${POLICY}_${DATASET_NAME}_seed${SEED} --policy.device=cuda --policy.push_to_hub=false --wandb.enable=true --wandb.disable_artifact=true --dataset.video_backend=pyav --batch_size=4 --steps=$STEPS --save_freq=$SAVE_FREQ --seed=$SEED --policy.pretrained_path=lerobot/pi0_base --policy.gradient_checkpointing=true --policy.dtype=bfloat16

export SEED=1
uv run lerobot-train --dataset.repo_id=local/${DATASET_NAME} --dataset.root=datasets/${DATASET_NAME} --policy.type=$POLICY --output_dir=outputs/train/${POLICY}_${DATASET_NAME}_seed${SEED} --job_name=${POLICY}_${DATASET_NAME}_seed${SEED} --policy.device=cuda --policy.push_to_hub=false --wandb.enable=true --wandb.disable_artifact=true --dataset.video_backend=pyav --batch_size=8 --steps=$STEPS --save_freq=$SAVE_FREQ --seed=$SEED
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 10000
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 20000
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 30000
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 40000
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 50000
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 60000
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 70000
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 80000
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 90000
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 100000

export SEED=2
uv run lerobot-train --dataset.repo_id=local/${DATASET_NAME} --dataset.root=datasets/${DATASET_NAME} --policy.type=$POLICY --output_dir=outputs/train/${POLICY}_${DATASET_NAME}_seed${SEED} --job_name=${POLICY}_${DATASET_NAME}_seed${SEED} --policy.device=cuda --policy.push_to_hub=false --wandb.enable=true --wandb.disable_artifact=true --dataset.video_backend=pyav --batch_size=8 --steps=$STEPS --save_freq=$SAVE_FREQ --seed=$SEED
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 10000
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 20000
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 30000
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 40000
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 50000
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 60000
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 70000
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 80000
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 90000
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 100000

export DATASET_NAME=soundDiff-m4-f10-s2-p0_0
export POLICY=vqbet
export SEED=1
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 10000
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 20000
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 30000
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 40000
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 50000
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 60000
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 70000
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 80000
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 90000
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 100000

export SEED=2
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 10000
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 20000
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 30000
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 40000
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 50000
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 60000
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 70000
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 80000
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 90000
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 100000

export POLICY=act
export DATASET_NAME=soundShake-m4-f10-s2-p0_0
export SEED=1
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 20000
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 40000
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 60000
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 80000
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 100000
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 120000
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 140000
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 160000
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 180000
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 200000

export SEED=2
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 20000
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 40000
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 60000
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 80000
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 100000
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 120000
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 140000
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 160000
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 180000
uv run src/eval_policy.py --training-name ${POLICY}_${DATASET_NAME}_seed${SEED} --checkpoint-step 200000
