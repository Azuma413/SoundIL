#!/bin/bash
# 評価117本を全件終わるまで繰り返し実行するドライバ。
#
# 04_eval.sh は「学習済みモデル」かつ「未評価」のものだけをジョブ化するため、
# 学習がまだ走っている条件は最初のラウンドではスキップされる。
# 本スクリプトはそれを検知して待機し、学習完了後に自動で拾い直す。
#
# 使い方:
#   GPUS=1,1,2,2,3 setsid nohup bash noise_exp/run_eval_loop.sh > noise_exp/logs/run_eval_loop.log 2>&1 &
#
# GPUS にGPU番号を重複させると、そのGPUに多くスロットを割り当てられる。
# (例: "1,1,2,2,3" は GPU1に2本・GPU2に2本・GPU3に1本)
set -u
cd "$(dirname "$0")/.."
source noise_exp/common.sh

step_dir=$(printf "%06d" "$STEPS")

# 未完了の評価件数を数える
remaining_evals() {
  local n=0 m s name env_task
  for m in "${MODES[@]}"; do
    for s in "${SEEDS[@]}"; do
      name=$(training_name "$m" "$s")
      while IFS= read -r env_task; do
        if [ ! -f "outputs/eval/${name}_${step_dir}_${env_task}/success_rate.txt" ]; then
          n=$((n + 1))
        fi
      done < <(eval_env_tasks "$m")
    done
  done
  echo "$n"
}

total=$(( ${#MODES[@]} * ${#SEEDS[@]} * 13 ))
round=0

while true; do
  round=$((round + 1))
  before=$(remaining_evals)
  echo "[eval_loop] round=${round} 残り ${before}/${total} 件  $(date '+%F %T')"

  if [ "$before" -eq 0 ]; then
    echo "[eval_loop] 全件完了"
    break
  fi

  bash noise_exp/04_eval.sh

  after=$(remaining_evals)
  echo "[eval_loop] round=${round} 終了: ${before} -> ${after} 件  $(date '+%F %T')"

  if [ "$after" -eq 0 ]; then
    echo "[eval_loop] 全件完了"
    break
  fi

  if [ "$after" -lt "$before" ]; then
    continue          # 進捗があったので次ラウンドへ
  fi

  # 進捗ゼロ。学習待ちなら待つ、そうでなければ失敗が残っているので抜ける。
  if pgrep -f "bin/lerobot-train" > /dev/null; then
    echo "[eval_loop] 学習中のモデルを待機 (5分)..."
    sleep 300
  else
    echo "[eval_loop] 進捗がなく学習も走っていません。${after} 件を残して終了します。"
    echo "[eval_loop] 失敗ログ: ${LOG_ROOT}/eval/_failures.txt"
    break
  fi
done

echo "--- 集計 ---"
$UV python noise_exp/05_summarize.py
echo "[eval_loop] 終了 $(date '+%F %T')"
