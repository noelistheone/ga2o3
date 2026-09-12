#!/bin/bash
# Phase 50 sequential ablation chain.
# Step 1: V18a — V5 + ordered structures (tests A alone)
# Step 2: Phase 50.B pretrain — multi-DB encoder (MP+Witman)
# Step 3: V18b — V5 + new pretrain encoder, disordered structures (tests B alone)
# Step 4: V18 — V5 + new pretrain encoder + ordered structures (tests A+B together)
#
# Each step writes a success marker file; subsequent steps gate on it.
# If any step fails, chain stops and morning user can resume from the failed step.

set -e
cd /home/lawrence/Physics/Ga2O3-Net

LOG_DIR="logs/phase50_chain"
MARKER_DIR="logs/phase50_chain/markers"
mkdir -p "$LOG_DIR" "$MARKER_DIR"

CONDA_RUN="conda run -n ga2o3 python -u"

echo "===================================================================="
echo " Phase 50 chain start: $(date)"
echo "===================================================================="

# ---- Step 1: V18a (A alone) ----
if [ ! -f "$MARKER_DIR/v18a.done" ]; then
    echo "[$(date)] Step 1/4: V18a (V5 + ordered structures)..."
    CUDA_VISIBLE_DEVICES=0 $CONDA_RUN scripts/04_finetune_predict.py \
        --multimodal-config config/multimodal_v2_film_method_offset_distill_v18a.yaml \
        --fusion-config     config/fusion_head128_phase50v18a_ordered_vc.yaml \
        --gpu 0 --ensemble-seeds 5 --cv-folds 10 --cv-group-by doi \
        --single-target vacancy_concentration \
        --session phase50v18a_ordered_vc_5seed \
        > "$LOG_DIR/v18a.log" 2>&1
    touch "$MARKER_DIR/v18a.done"
    echo "[$(date)] Step 1/4 V18a: DONE"
fi

# ---- Step 2: B pretrain (multi-DB encoder) ----
if [ ! -f "$MARKER_DIR/pretrain_p50.done" ]; then
    echo "[$(date)] Step 2/4: Phase 50.B multi-DB pretrain..."
    $CONDA_RUN scripts/02b_pretrain_encoder_multidb.py \
        --epochs 30 --gpu 0 --batch-size 128 \
        --mp-fraction 0.15 --lr 1.0e-4 \
        --mp-loss-weight 0.3 --vacancy-loss-weight 1.0 \
        --out checkpoints/pretrained_encoder_phase50.pt \
        > "$LOG_DIR/pretrain.log" 2>&1
    touch "$MARKER_DIR/pretrain_p50.done"
    echo "[$(date)] Step 2/4 pretrain: DONE"
fi

# ---- Step 3: V18b (B alone — new pretrain, disordered structures) ----
if [ ! -f "$MARKER_DIR/v18b.done" ]; then
    echo "[$(date)] Step 3/4: V18b (V5 + new pretrain encoder, disordered)..."
    CUDA_VISIBLE_DEVICES=0 $CONDA_RUN scripts/04_finetune_predict.py \
        --multimodal-config config/multimodal_v2_film_method_offset_distill_v18b.yaml \
        --fusion-config     config/fusion_head128_phase50v18b_newpretrain_vc.yaml \
        --gpu 0 --ensemble-seeds 5 --cv-folds 10 --cv-group-by doi \
        --single-target vacancy_concentration \
        --session phase50v18b_newpretrain_vc_5seed \
        > "$LOG_DIR/v18b.log" 2>&1
    touch "$MARKER_DIR/v18b.done"
    echo "[$(date)] Step 3/4 V18b: DONE"
fi

# ---- Step 4: V18 (A+B combined) ----
if [ ! -f "$MARKER_DIR/v18.done" ]; then
    echo "[$(date)] Step 4/4: V18 (V5 + new pretrain + ordered)..."
    CUDA_VISIBLE_DEVICES=0 $CONDA_RUN scripts/04_finetune_predict.py \
        --multimodal-config config/multimodal_v2_film_method_offset_distill_v18.yaml \
        --fusion-config     config/fusion_head128_phase50v18_combined_vc.yaml \
        --gpu 0 --ensemble-seeds 5 --cv-folds 10 --cv-group-by doi \
        --single-target vacancy_concentration \
        --session phase50v18_combined_vc_5seed \
        > "$LOG_DIR/v18.log" 2>&1
    touch "$MARKER_DIR/v18.done"
    echo "[$(date)] Step 4/4 V18: DONE"
fi

echo "===================================================================="
echo " Phase 50 chain complete: $(date)"
echo "===================================================================="
