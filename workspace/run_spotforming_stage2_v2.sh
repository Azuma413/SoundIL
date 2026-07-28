#!/bin/bash
# Stage 2 v2: データセット生成(既存ならスキップ) → ACT学習(pyav, resume対応) → クリーン評価 → OOD評価
# 使い方: bash workspace/run_spotforming_stage2_v2.sh <spectrogram_mode 0|1|2> <gpu_id>
set -euo pipefail
MODE=$1
GPU=$2
export CUDA_VISIBLE_DEVICES=${GPU}
TASK="soundDiff-m4-f10-s2-p0-nx${MODE}"
DS="${TASK}_0"
TRAIN_DIR="outputs/train/act_${DS}_seed0"
cd /workspace/myproject

if [ ! -d "datasets/${DS}" ]; then
  echo "=== [mode ${MODE}] dataset generation (${TASK}, 50 episodes) ==="
  uv run --no-sync python src/make_sim_dataset.py ${TASK} --episode-num 50
else
  echo "=== [mode ${MODE}] dataset exists, skipping generation ==="
fi

echo "=== [mode ${MODE}] video integrity check/repair ==="
uv run --no-sync python workspace/repair_dataset_videos.py datasets/${DS}

RESUME_ARGS=""
if [ -d "${TRAIN_DIR}/checkpoints" ] && [ -n "$(ls -A ${TRAIN_DIR}/checkpoints 2>/dev/null)" ]; then
  echo "=== [mode ${MODE}] resuming training from existing checkpoint ==="
  RESUME_ARGS="--resume=true --config_path=${TRAIN_DIR}/checkpoints/last/pretrained_model/train_config.json"
elif [ -d "${TRAIN_DIR}" ]; then
  echo "=== [mode ${MODE}] removing incomplete train dir (no checkpoints) ==="
  rm -rf "${TRAIN_DIR}"
fi

echo "=== [mode ${MODE}] ACT training (100k steps) ==="
uv run --no-sync lerobot-train \
  --dataset.repo_id=local/${DS} \
  --dataset.root=datasets/${DS} \
  --policy.type=act \
  --output_dir=${TRAIN_DIR} \
  --job_name=act_${DS}_seed0 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable=true \
  --wandb.disable_artifact=true \
  --seed=0 \
  --batch_size=8 \
  --steps=100000 \
  ${RESUME_ARGS}

echo "=== [mode ${MODE}] eval: in-distribution (clean) ==="
uv run --no-sync python src/eval_policy.py \
  --training-name act_${DS}_seed0 --checkpoint-step 100000 --episode-num 50

for NI in 0.25 0.5 1.0; do
  echo "=== [mode ${MODE}] eval: OOD opposite-sound noise ni=${NI} ==="
  uv run --no-sync python src/eval_policy.py \
    --training-name act_${DS}_seed0 --checkpoint-step 100000 --episode-num 50 \
    --env-task soundDiff-m4-f10-s2-p0-no${MODE}-ni${NI}-nopp
done

echo "=== [mode ${MODE}] ALL DONE ==="
