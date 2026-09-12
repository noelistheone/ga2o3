#!/usr/bin/env bash
# Phase 59 V59-CALM — run ONE ablation (A–F) for ONE target on ONE GPU.
# Ablation matrix (doc §9):
#   A baseline V55-Ext (no CALM, no DCC)            mm=v54a1_sincere   fu=v55ext
#   B + zero-gated z_expert only (no DCC)           mm=v59_calm_expert fu=v55ext
#   C + DCC only (no new modalities)                mm=v54a1_sincere   fu=v59_dcc
#   D + Δ-learning head only (CALM, no modal, noDCC)mm=v59_calm_none   fu=v55ext
#   E full V59-CALM (all modalities + DCC)          mm=v59_calm_full   fu=v59_dcc
#   F λ_dcc sweep handled by FUSION_OVERRIDE env (dcc_weight variants)
#
# Usage: bash scripts/62_v59_run_ablation.sh <A|B|C|D|E> <vacancy_concentration|photo_dark_ratio> <gpu> [seeds] [folds] [fusion_override]
set -euo pipefail
cd "$(dirname "$0")/.."

ABL="$1"; TARGET="$2"; GPU="$3"; SEEDS="${4:-5}"; FOLDS="${5:-10}"; FU_OVERRIDE="${6:-}"
if [ "$TARGET" = "vacancy_concentration" ]; then TT="vc"; else TT="pdr"; fi

case "$ABL" in
  A) MM=config/multimodal_v54a1_sincere.yaml;      FU=config/fusion_v55ext_qwen_${TT}.yaml ;;
  B) MM=config/multimodal_v59_calm_expert.yaml;    FU=config/fusion_v55ext_qwen_${TT}.yaml ;;
  C) MM=config/multimodal_v54a1_sincere.yaml;      FU=config/fusion_v59_dcc_${TT}.yaml ;;
  D) MM=config/multimodal_v59_calm_none.yaml;      FU=config/fusion_v55ext_qwen_${TT}.yaml ;;
  E) MM=config/multimodal_v59_calm_full.yaml;      FU=config/fusion_v59_dcc_${TT}.yaml ;;
  *) echo "unknown ablation $ABL"; exit 1 ;;
esac
[ -n "$FU_OVERRIDE" ] && FU="$FU_OVERRIDE"

SESSION="phase59v59${ABL}_${TT}_5seed"
LOG="logs/phase59_${ABL}_${TT}.log"
mkdir -p logs
echo "[$(date +%H:%M:%S)] V59 ablation $ABL ($TARGET) → results/$SESSION  | mm=$MM fu=$FU gpu=$GPU seeds=$SEEDS folds=$FOLDS"

PYTHONPATH=. conda run --no-capture-output -n ga2o3 python -u scripts/04_finetune_predict.py \
  --multimodal-config "$MM" \
  --fusion-config "$FU" \
  --single-target "$TARGET" \
  --cv-group-by doi --cv-folds "$FOLDS" --ensemble-seeds "$SEEDS" \
  --session "$SESSION" --gpu "$GPU" 2>&1 | tee "$LOG" | grep -vE "WARNING|UserWarning|warnings.warn|FutureWarning" || true
echo "[$(date +%H:%M:%S)] DONE V59 ablation $ABL ($TARGET) → results/$SESSION"
