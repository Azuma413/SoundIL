#!/bin/bash
# GPUスロットにジョブを割り当てて並列実行する簡易キューランナー。
#
# 使い方: run_jobs.sh <jobs_file> <gpu_list> <slots_per_gpu> <log_dir>
#   jobs_file: 1行 = "<ジョブ名><TAB><シェルコマンド>"
#   各ジョブは CUDA_VISIBLE_DEVICES を割り当てられた状態で実行される。
set -u

JOBS_FILE="$1"
GPUS_ARG="${2:-0,1,2,3}"
SLOTS_PER_GPU="${3:-1}"
LOG_DIR="${4:-./logs}"

mkdir -p "$LOG_DIR"

IFS=',' read -ra GPU_ARR <<< "$GPUS_ARG"
TOKENS=()
for ((i = 0; i < SLOTS_PER_GPU; i++)); do
  for g in "${GPU_ARR[@]}"; do TOKENS+=("$g"); done
done

TOTAL=$(grep -cve '^[[:space:]]*$' "$JOBS_FILE")
if [ "$TOTAL" -eq 0 ]; then
  echo "[run_jobs] 実行するジョブがありません: $JOBS_FILE"
  exit 0
fi

echo "[run_jobs] jobs=$TOTAL gpus=$GPUS_ARG slots/gpu=$SLOTS_PER_GPU (並列度=${#TOKENS[@]})"
echo "[run_jobs] logs -> $LOG_DIR"

# GPUスロットのトークンプールをFIFOで管理する
FIFO=$(mktemp -u)
mkfifo "$FIFO"
exec 3<>"$FIFO"
rm -f "$FIFO"

# STAGGER_SEC を指定すると、スロットを一気に開けずに間隔をあけて開放する。
# 別ランナーのジョブが動いている状態で並列数を上げるとメモリを食い潰すため、
# 既存ジョブが抜けるのに合わせて徐々に立ち上げたいときに使う。
STAGGER_SEC="${STAGGER_SEC:-0}"
if [ "$STAGGER_SEC" -gt 0 ]; then
  ( for t in "${TOKENS[@]}"; do echo "$t" >&3; sleep "$STAGGER_SEC"; done ) &
  echo "[run_jobs] スロットを ${STAGGER_SEC} 秒間隔で開放します"
else
  for t in "${TOKENS[@]}"; do echo "$t" >&3; done
fi

FAIL_LOG="$LOG_DIR/_failures.txt"
: > "$FAIL_LOG"

idx=0
while IFS=$'\t' read -r name cmd; do
  [ -z "${name:-}" ] && continue
  read -r gpu <&3
  idx=$((idx + 1))
  echo "[$(date '+%m-%d %H:%M:%S')] ($idx/$TOTAL) START gpu=$gpu  $name"
  (
    log="$LOG_DIR/${name}.log"
    start=$(date +%s)
    CUDA_VISIBLE_DEVICES="$gpu" bash -c "$cmd" > "$log" 2>&1
    rc=$?
    dur=$(( ($(date +%s) - start) / 60 ))
    if [ $rc -eq 0 ]; then
      echo "[$(date '+%m-%d %H:%M:%S')] DONE  gpu=$gpu (${dur}min)  $name"
    else
      echo "[$(date '+%m-%d %H:%M:%S')] FAIL  gpu=$gpu rc=$rc (${dur}min)  $name  -> $log"
      echo -e "${name}\trc=${rc}\t${log}" >> "$FAIL_LOG"
    fi
    echo "$gpu" >&3
  ) &
done < "$JOBS_FILE"

wait
exec 3>&-

nfail=$(grep -cve '^[[:space:]]*$' "$FAIL_LOG" 2>/dev/null || echo 0)
echo "[run_jobs] 完了: $TOTAL 件中 $nfail 件失敗"
[ "$nfail" -eq 0 ]
