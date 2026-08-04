#!/bin/bash
# 実行中のデータ収集 (01_collect.sh) の完了を待ってから
# マージ -> 学習 -> 評価 -> 集計 を通しで実行する。
#
# 使い方: setsid nohup bash noise_exp/run_rest.sh > noise_exp/logs/run_rest.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
source noise_exp/common.sh

echo "[run_rest] 開始 $(date '+%F %T')  GPUS=${GPUS}"

# --- 収集の完了待ち ---------------------------------------------------------
if pgrep -f "make_sim_dataset.py" > /dev/null; then
  echo "[run_rest] データ収集の完了を待機中..."
  while pgrep -f "make_sim_dataset.py" > /dev/null; do
    sleep 60
  done
  echo "[run_rest] データ収集が終了しました $(date '+%F %T')"
fi

# 収集結果の検証: 3条件すべてが揃い、合計100epになるか
for m in "${MODES[@]}"; do
  d="datasets/${BASE_TASK}-nx${m}_1"
  if [ ! -d "$d" ]; then
    echo "[run_rest] ERROR: $d がありません。収集が失敗している可能性があります。"
    exit 1
  fi
  n=$($UV python -c "import json;print(json.load(open('$d/meta/info.json'))['total_episodes'])")
  echo "[run_rest] $d : ${n} ep"
  if [ "$n" -ne "$COLLECT_EPISODES" ]; then
    echo "[run_rest] ERROR: ${COLLECT_EPISODES} ep のはずが ${n} ep です。中断します。"
    exit 1
  fi
done

echo "--- Step2: マージ (50ep + 50ep -> 100ep) ---"
$UV python noise_exp/02_merge.py || exit 1

echo "--- Step3: 学習 (3条件 x 3シード = 9本) ---"
bash noise_exp/03_train.sh
echo "[run_rest] 学習フェーズ終了 $(date '+%F %T')"

echo "--- Step4: 評価 (9モデル x 13ノイズ条件 = 117本) ---"
bash noise_exp/04_eval.sh
echo "[run_rest] 評価フェーズ終了 $(date '+%F %T')"

echo "--- Step5: 集計 ---"
$UV python noise_exp/05_summarize.py

echo "[run_rest] 全工程終了 $(date '+%F %T')"
