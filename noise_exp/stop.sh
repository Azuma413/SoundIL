#!/bin/bash
# 実験ジョブを安全に停止する。
#
#   bash noise_exp/stop.sh eval    評価だけ止める（学習は続行）
#   bash noise_exp/stop.sh train   学習だけ止める
#   bash noise_exp/stop.sh all     両方止める
#
# 重要: ディスパッチャ(run_jobs.sh)を先に落とさないと、個々のジョブを殺すたびに
# 次のジョブが投入されて止まらない。必ず親から順に停止する。
set -u
target="${1:-}"

kill_tree() {  # $1 = 説明, $2 = ディスパッチャのpgrepパターン, $3 = ジョブのpgrepパターン
  local label="$1" dispatcher="$2" job="$3" pids

  # 1) ループドライバ
  pkill -f "run_eval_loop.sh" 2>/dev/null
  pkill -f "run_rest.sh" 2>/dev/null

  # 2) ディスパッチャ
  pids=$(pgrep -f "$dispatcher" | tr '\n' ' ')
  [ -n "$pids" ] && kill -KILL $pids 2>/dev/null
  sleep 2

  # 3) 個々のジョブ (uvラッパと実プロセスの両方)
  pids=$(pgrep -f "$job" | tr '\n' ' ')
  [ -n "$pids" ] && kill -TERM $pids 2>/dev/null
  sleep 5
  pids=$(pgrep -f "$job" | tr '\n' ' ')
  [ -n "$pids" ] && kill -KILL $pids 2>/dev/null
  sleep 3

  local left
  left=$(pgrep -cf "$job")
  echo "[stop] ${label}: 残 ${left} プロセス"
}

case "$target" in
  eval)
    kill_tree "評価" "run_jobs.sh.*04_eval" "src/eval_policy.py"
    ;;
  train)
    kill_tree "学習" "run_jobs.sh.*03_train" "bin/lerobot-train"
    ;;
  all)
    kill_tree "評価" "run_jobs.sh.*04_eval" "src/eval_policy.py"
    kill_tree "学習" "run_jobs.sh.*03_train" "bin/lerobot-train"
    ;;
  *)
    echo "使い方: bash noise_exp/stop.sh {eval|train|all}"
    exit 1
    ;;
esac

echo "[stop] CPU合計: $(ps -eo pcpu --no-headers | awk '{s+=$1} END {printf "%.0f%%", s}')  $(uptime | sed 's/.*load/load/')"
