#!/bin/bash
#SBATCH --partition part-group_25b505
#SBATCH --nodelist=aic-gh2b-310036
# 310033 ~ 310036 のどれか
#SBATCH --gres=gpu:1

module load singularitypro/4.1

PROJECT_ROOT="${HOME}/SourceCode/sound_dp"
SIF_IMAGE="singularity.sif"
singularity exec --nv -B "${PROJECT_ROOT}:/workspace" "${SIF_IMAGE}" uv run src/eval_policy.py
echo "Done."

# sbatch eval.sh