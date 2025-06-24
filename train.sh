#!/bin/bash
#SBATCH --partition part-group_25b505
#SBATCH --nodelist=aic-gh2b-310036
# [310033-310036]のどれか
#SBATCH --gres=gpu:1

export DATASET_NAME="dummy-ep150" # "sound-ep100"
export POLICY="act"
# export WANDB_API_KEY="your_wandb_api_key_here"  # Replace with your actual WandB API key
export WANDB_API_KEY="5767e2baca4de66a67547e79fdf0e61f3be358bd"  # Replace with your actual WandB API key

PROJECT_ROOT="${HOME}/SourceCode/sound_dp"
SIF_IMAGE="singularity.sif"
echo "Running lerobot/lerobot/scripts/train.py in ${SIF_IMAGE}..."
singularity exec --nv -B "${PROJECT_ROOT}:/workspace" "${SIF_IMAGE}" uv run lerobot/lerobot/scripts/train.py \
  --dataset.repo_id="local/${DATASET_NAME}" \
  --dataset.root="/workspace/datasets/${DATASET_NAME}" \
  --policy.type="$POLICY" \
  --output_dir="/workspace/outputs/train/${POLICY}-${DATASET_NAME}" \
  --job_name="${POLICY}-${DATASET_NAME}" \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.disable_artifact=true \
  --batch_size=8 \
  --steps=100000

echo "Done."

# sbatch train.sh