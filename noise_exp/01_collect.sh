#!/bin/bash
# Step1: 追加の学習データを50エピソード収集する。
# --all-modes により1回のシミュレーションで
#   nx0 (Spotforming) / nx1 (単一マイク) / nx2 (平均)
# の3データセットを同時生成する。
#
# 出力: datasets/soundDiff-m4-f10-s2-p0-nx{0,1,2}_1
#       (既存の _0 が残っているため自動採番で _1 になる)
set -eu
cd "$(dirname "$0")/.."
source noise_exp/common.sh

GPU="${COLLECT_GPU:-$(echo "$GPUS" | cut -d, -f1)}"
LOG="${LOG_ROOT}/01_collect.log"

echo "[collect] task=${BASE_TASK}-nx0 episodes=${COLLECT_EPISODES} gpu=${GPU}"
echo "[collect] log -> ${LOG}"

CUDA_VISIBLE_DEVICES="$GPU" $UV python -u src/make_sim_dataset.py \
  "${BASE_TASK}-nx0" \
  --episode-num "${COLLECT_EPISODES}" \
  --all-modes \
  > "$LOG" 2>&1

echo "[collect] 完了。生成されたデータセット:"
for m in "${MODES[@]}"; do
  for d in datasets/${BASE_TASK}-nx${m}_*; do
    [ -d "$d" ] || continue
    n=$($UV python -c "import json;print(json.load(open('$d/meta/info.json'))['total_episodes'])" 2>/dev/null || echo "?")
    echo "  $d : ${n} ep"
  done
done
