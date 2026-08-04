#!/bin/bash
# Step3: 3条件 (nx0/nx1/nx2) x 3シード (0/1/2) = 9本の学習をGPU並列で実行する。
# 学習データは3シードで同一のものを使う。
# 既に checkpoints/100000 がある学習はスキップする。
set -eu
cd "$(dirname "$0")/.."
source noise_exp/common.sh

JOBS_FILE="${LOG_ROOT}/03_train.jobs"
: > "$JOBS_FILE"

step_dir=$(printf "%06d" "$STEPS")

for m in "${MODES[@]}"; do
  ds=$(dataset_name "$m")
  if [ ! -d "datasets/${ds}" ]; then
    echo "[train] ERROR: データセットがありません: datasets/${ds}"
    exit 1
  fi
  for s in "${SEEDS[@]}"; do
    name=$(training_name "$m" "$s")
    if [ -d "outputs/train/${name}/checkpoints/${step_dir}/pretrained_model" ]; then
      echo "[train] skip (学習済み): ${name}"
      continue
    fi
    cmd="$UV lerobot-train \
--dataset.repo_id=local/${ds} \
--dataset.root=datasets/${ds} \
--policy.type=${POLICY} \
--output_dir=outputs/train/${name} \
--job_name=${name} \
--policy.device=cuda \
--policy.push_to_hub=false \
${WANDB_ARGS} \
--dataset.video_backend=pyav \
--batch_size=${BATCH_SIZE} \
--steps=${STEPS} \
--save_freq=${SAVE_FREQ} \
--seed=${s}"
    printf '%s\t%s\n' "$name" "$cmd" >> "$JOBS_FILE"
  done
done

bash noise_exp/run_jobs.sh "$JOBS_FILE" "$GPUS" "$TRAIN_SLOTS_PER_GPU" "${LOG_ROOT}/train"
