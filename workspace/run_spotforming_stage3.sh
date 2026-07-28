#!/bin/bash
# Stage 3: 弱ノイズ入り学習(-no*, 白色点音源ノイズ強度0.1) → 強ノイズOOD評価
#   データセットは1回のシミュレーションで3モード同時生成(--all-modes)し、
#   3手法のエピソード・軌道を完全に一致させる。
#
# 使い方:
#   bash workspace/run_spotforming_stage3.sh gen <gpu>          # 生成+修復（1回だけ）
#   bash workspace/run_spotforming_stage3.sh train <mode> <gpu> # GPU空き待ち→学習→評価
set -euo pipefail
cd /workspace/myproject
BASE="soundDiff-m4-f10-s2-p0"
CMD=$1

wait_for_gpu() {
  local gpu=$1
  local ok=0
  echo "waiting for GPU${gpu} to be free..."
  while [ $ok -lt 3 ]; do
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu")
    if [ "$used" -lt 1000 ]; then ok=$((ok+1)); else ok=0; fi
    sleep 60
  done
  echo "GPU${gpu} is free"
}

if [ "$CMD" = "gen" ]; then
  GPU=$2
  export CUDA_VISIBLE_DEVICES=${GPU}
  echo "=== [stage3 gen] all-modes dataset generation (${BASE}-no0, 50 episodes) ==="
  uv run --no-sync python src/make_sim_dataset.py ${BASE}-no0 --episode-num 50 --all-modes
  for M in 0 1 2; do
    echo "=== [stage3 gen] video integrity check/repair: no${M} ==="
    uv run --no-sync python workspace/repair_dataset_videos.py datasets/${BASE}-no${M}_0
  done
  touch workspace/logs/stage3_gen.done
  echo "=== [stage3 gen] DONE ==="

elif [ "$CMD" = "train" ]; then
  MODE=$2
  GPU=$3
  SEED=${4:-0}
  DS="${BASE}-no${MODE}_0"
  TRAIN_DIR="outputs/train/act_${DS}_seed${SEED}"
  echo "waiting for stage3 dataset generation..."
  until [ -f workspace/logs/stage3_gen.done ]; do sleep 60; done
  wait_for_gpu ${GPU}
  export CUDA_VISIBLE_DEVICES=${GPU}

  RESUME_ARGS=""
  if [ -d "${TRAIN_DIR}/checkpoints" ] && [ -n "$(ls -A ${TRAIN_DIR}/checkpoints 2>/dev/null)" ]; then
    echo "=== [stage3 mode ${MODE}] resuming from checkpoint ==="
    RESUME_ARGS="--resume=true --config_path=${TRAIN_DIR}/checkpoints/last/pretrained_model/train_config.json"
  elif [ -d "${TRAIN_DIR}" ]; then
    rm -rf "${TRAIN_DIR}"
  fi

  echo "=== [stage3 mode ${MODE}] ACT training (100k steps) on GPU${GPU} ==="
  uv run --no-sync lerobot-train \
    --dataset.repo_id=local/${DS} \
    --dataset.root=datasets/${DS} \
    --policy.type=act \
    --output_dir=${TRAIN_DIR} \
    --job_name=act_${DS}_seed${SEED} \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --seed=${SEED} \
    --batch_size=8 \
    --steps=100000 \
    ${RESUME_ARGS}

  echo "=== [stage3 mode ${MODE}] eval: in-distribution (white noise 0.1) ==="
  uv run --no-sync python src/eval_policy.py \
    --training-name act_${DS}_seed${SEED} --checkpoint-step 100000 --episode-num 50

  for NI in 0.25 0.5 1.0; do
    echo "=== [stage3 mode ${MODE}] eval: OOD opposite-sound ni=${NI} ==="
    uv run --no-sync python src/eval_policy.py \
      --training-name act_${DS}_seed${SEED} --checkpoint-step 100000 --episode-num 50 \
      --env-task ${BASE}-no${MODE}-ni${NI}-nopp
  done
  echo "=== [stage3 mode ${MODE}] ALL DONE ==="
else
  echo "unknown command: $CMD"; exit 1
fi
