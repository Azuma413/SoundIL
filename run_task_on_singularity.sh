#!/bin/bash
#SBATCH --partition part-group_25b505
#SBATCH --nodelist=aic-gh2b-310033
# [310033-310036]のどれか

SIF_IMAGE="singularity.sif"
PYTHON_SCRIPT="src/make_sim_dataset.py"
# export WANDB_API_KEY="your_wandb_api_key_here"  # Replace with your actual WandB API key
export WANDB_API_KEY="5767e2baca4de66a67547e79fdf0e61f3be358bd"  # Replace with your actual WandB API key

echo "Running ${PYTHON_SCRIPT} in ${SIF_IMAGE}..."
singularity exec --nv "${SIF_IMAGE}" uv run "${PYTHON_SCRIPT}"
echo "Done."

# sbatch run_task_on_singularity.sh