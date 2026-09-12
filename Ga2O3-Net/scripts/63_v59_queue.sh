#!/usr/bin/env bash
# Phase 59 V59-CALM — per-GPU serial ablation queue (GPU-serial rule).
# Usage: bash scripts/63_v59_queue.sh <target> <gpu> [seeds] [folds] [ablations...]
#   e.g. bash scripts/63_v59_queue.sh vacancy_concentration 0 5 10 A C E D B
set -uo pipefail
cd "$(dirname "$0")/.."

TARGET="$1"; GPU="$2"; SEEDS="${3:-5}"; FOLDS="${4:-10}"; shift 4 || true
ABLS=("$@"); [ ${#ABLS[@]} -eq 0 ] && ABLS=(A C E D B)
if [ "$TARGET" = "vacancy_concentration" ]; then TT="vc"; else TT="pdr"; fi

QLOG="logs/phase59_queue_${TT}_gpu${GPU}.log"
mkdir -p logs
echo "[$(date +%F_%H:%M:%S)] QUEUE start target=$TARGET gpu=$GPU seeds=$SEEDS folds=$FOLDS ablations=${ABLS[*]}" | tee -a "$QLOG"
for A in "${ABLS[@]}"; do
  echo "[$(date +%F_%H:%M:%S)] >>> ablation $A start" | tee -a "$QLOG"
  bash scripts/62_v59_run_ablation.sh "$A" "$TARGET" "$GPU" "$SEEDS" "$FOLDS" >> "$QLOG" 2>&1
  rc=$?
  echo "[$(date +%F_%H:%M:%S)] <<< ablation $A done rc=$rc → results/phase59v59${A}_${TT}_5seed" | tee -a "$QLOG"
done
echo "[$(date +%F_%H:%M:%S)] QUEUE COMPLETE target=$TARGET gpu=$GPU" | tee -a "$QLOG"
