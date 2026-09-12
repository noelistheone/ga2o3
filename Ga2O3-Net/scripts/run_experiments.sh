#!/usr/bin/env bash
# run_experiments.sh
# Sequential pretrain experiment chain. Runs inside a tmux session so it
# survives SSH disconnection. Each run saves its own checkpoint and metrics CSV.
#
# Usage (from project root, inside tmux):
#   bash scripts/run_experiments.sh
#
# Each run:
#   1. Downloads data (once, idempotent)
#   2. Runs 02_pretrain_encoder.py with the run's config
#   3. Writes logs/runXX_*.log and logs/runXX_*_metrics.csv

set -euo pipefail
cd "$(dirname "$0")/.."
PROJECT="$(pwd)"
PYTHON="conda run -n ga2o3 python -u"
LOG="$PROJECT/logs"
mkdir -p "$LOG" checkpoints

echo "================================================================"
echo " Ga2O3-Net Pretrain Experiment Chain"
echo " Start: $(date)"
echo "================================================================"

# ── Step 0: Download MP data (idempotent — skips already-downloaded) ─────────
echo ""
echo ">>> [$(date +%H:%M:%S)] Downloading MP data (Pool A: ~5000 + Pool B targeted)..."
$PYTHON scripts/01_download_mp.py --max-broad 5000 2>&1 | tee "$LOG/00_download.log"
echo ">>> Download done."

# ── Helper: run one experiment ────────────────────────────────────────────────
run_exp() {
    local name="$1"
    local cfg="$2"
    local gpu="${3:-0}"
    echo ""
    echo "================================================================"
    echo " >>> START: $name  [$(date)]"
    echo "================================================================"
    $PYTHON scripts/02_pretrain_encoder.py \
        --config "$cfg" \
        --run-name "$name" \
        --gpu "$gpu" \
        2>&1 | tee "$LOG/${name}.log"
    echo ">>> DONE: $name  [$(date)]"
    echo ""
    # Print final best val metrics from CSV
    CSV=$(ls -t "$LOG"/${name}_metrics.csv 2>/dev/null | head -1)
    if [ -n "$CSV" ]; then
        echo "  Best metrics for $name:"
        awk -F',' 'NR>1 {val=$4; if(val<min || NR==2){min=val; row=$0}} END{print "    " row}' "$CSV"
    fi
}

# ── Experiments ───────────────────────────────────────────────────────────────
run_exp "run00_baseline"       "config/experiments/run00_baseline.yaml"       0
run_exp "run01_residual_silu"  "config/experiments/run01_residual_silu.yaml"  0
run_exp "run02_deeper_wider"   "config/experiments/run02_deeper_wider.yaml"   0
run_exp "run03_attention_pool" "config/experiments/run03_attention_pool.yaml" 0
run_exp "run04_best"           "config/experiments/run04_best.yaml"           0

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo " ALL EXPERIMENTS COMPLETE — $(date)"
echo "================================================================"
echo ""
echo " Best val_loss per run:"
for name in run00_baseline run01_residual_silu run02_deeper_wider run03_attention_pool run04_best; do
    CSV=$(ls -t "$LOG"/${name}_metrics.csv 2>/dev/null | head -1)
    if [ -f "$CSV" ]; then
        BEST=$(awk -F',' 'NR>1 {v=$4+0; if(v<min || NR==2){min=v}} END{printf "%.6f", min}' "$CSV")
        EPOCHS=$(tail -n +2 "$CSV" | wc -l)
        printf "  %-30s  best_val_loss=%-10s  epochs=%s\n" "$name" "$BEST" "$EPOCHS"
    else
        printf "  %-30s  (no results)\n" "$name"
    fi
done
echo ""
echo " Checkpoints: $(ls checkpoints/run*.pt 2>/dev/null | tr '\n' ' ')"
echo " View curves: bash scripts/monitor.sh --tb"
