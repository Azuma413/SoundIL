#!/bin/bash
# Step4: 学習済み9モデル x ノイズ条件13通り = 117本の評価をGPU並列で実行する。
#
# ノイズ条件 (mode m の学習モデルには必ず -no{m} を対応させる):
#   -ni0.0                       ノイズ無し (RMS比0.0。ホワイト/反対音で結果は同一)
#   -ni{0.25..1.5}               ホワイトノイズ
#   -ni{0.25..1.5}-nopp          反対音 (soundDiffの逆側タスク音)
# ノイズ音源位置は作業空間中心から半径2mの円周上のランダム点 (既定値、エピソード内固定)。
#
# 既に success_rate.txt がある評価はスキップする。
set -eu
cd "$(dirname "$0")/.."
source noise_exp/common.sh

JOBS_FILE="${LOG_ROOT}/04_eval.jobs"
: > "$JOBS_FILE"

step_dir=$(printf "%06d" "$STEPS")

for m in "${MODES[@]}"; do
  for s in "${SEEDS[@]}"; do
    name=$(training_name "$m" "$s")
    if [ ! -d "outputs/train/${name}/checkpoints/${step_dir}/pretrained_model" ]; then
      echo "[eval] skip (未学習): ${name}"
      continue
    fi
    while IFS= read -r env_task; do
      out="outputs/eval/${name}_${step_dir}_${env_task}"
      if [ -f "${out}/success_rate.txt" ]; then
        echo "[eval] skip (評価済み): ${name} / ${env_task}"
        continue
      fi
      # 1プロセスが32コア分のスレッドプールを複数張ると、数本並列で走らせただけで
      # スレッド数が1000を超えて競合する。1本あたりのコア数を絞る。
      cmd="OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
$UV python -u src/eval_policy.py \
--training-name ${name} \
--checkpoint-step ${STEPS} \
--env-task ${env_task} \
--episode-num ${EVAL_EPISODES}"
      printf '%s\t%s\n' "${name}__${env_task}" "$cmd" >> "$JOBS_FILE"
    done < <(eval_env_tasks "$m")
  done
done

bash noise_exp/run_jobs.sh "$JOBS_FILE" "$GPUS" "$EVAL_SLOTS_PER_GPU" "${LOG_ROOT}/eval"
