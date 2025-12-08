#!/bin/bash
set -euo pipefail

GPUS=(0 1)
MAX_PER_GPU=1
NUM_GPUS=${#GPUS[@]}
TOTAL_SLOTS=$((NUM_GPUS * MAX_PER_GPU))

LOG_DIR=./logs/convert_a2d_slice1
DATA_DIR=/mnt/sfs_turbo/huawei-meadow/post-training-lerobot-data/a2d_slice1
OUTPUT_DIR=/mnt/sfs_turbo/rlds
OUTPUT_PREFIX="a2d_slice1"
mkdir -p "$LOG_DIR"

# 要跑的 STEPS（以列表形式定义，方便增删）
STEPS=(
  batch_000
  batch_001
  batch_002
  batch_003
)

# initialization
declare -a JOB_PIDS
for ((i=0; i<TOTAL_SLOTS; i++)); do
  JOB_PIDS[i]=0
done

start_job() {
  local STEP=$1
  local SLOT=$2
  # calculate respective slot
  local GPU_INDEX=$(( SLOT / MAX_PER_GPU ))
  local GPU=${GPUS[$GPU_INDEX]}

  echo "[$(date +'%H:%M:%S')] START ${OUTPUT_PREFIX}_${STEP} on slot ${SLOT}"
  python lerobot/scripts/lerobot2rlds.py \
    --src-dir "${DATA_DIR}/${STEP}" \
    --output-dir ${OUTPUT_DIR} \
    --task-name "${OUTPUT_PREFIX}_${STEP}" > ${LOG_DIR}/"${OUTPUT_PREFIX}_${STEP}.log" 2>&1 &

  JOB_PIDS[$SLOT]=$!
}

# tranverse all STEP，start if there are empty slots
for STEP in "${STEPS[@]}"; do
  while :; do
    for ((slot=0; slot<TOTAL_SLOTS; slot++)); do
      pid=${JOB_PIDS[slot]}
      if [[ $pid -eq 0 ]] || ! kill -0 "$pid" 2>/dev/null; then
        start_job "$STEP" "$slot"
        break 2
      fi
    done
    sleep 2
  done
done

# wait
for pid in "${JOB_PIDS[@]}"; do
  [[ $pid -ne 0 ]] && wait "$pid"
done

echo "All data processed."

