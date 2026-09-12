#!/usr/bin/env bash
# monitor.sh — Live training monitor. Run from any SSH session.
#
# Shows:
#   - Active tmux sessions
#   - GPU utilization (both RTX 3090s)
#   - Latest metrics from the most recent log file
#   - TensorBoard URL (with SSH tunnel instructions)
#
# Usage:
#   bash scripts/monitor.sh            # one-shot status
#   bash scripts/monitor.sh --watch    # refresh every 10s (Ctrl+C to stop)
#   bash scripts/monitor.sh --tb       # start TensorBoard server (port 6006)

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$PROJECT/logs"
REFRESH=10

# ── Parse args ────────────────────────────────────────────────────────────────
WATCH=false
START_TB=false
for arg in "$@"; do
    case "$arg" in
        --watch) WATCH=true ;;
        --tb)    START_TB=true ;;
    esac
done

# ── TensorBoard mode ─────────────────────────────────────────────────────────
if $START_TB; then
    TB_DIR="$LOG_DIR/tensorboard"
    if [ ! -d "$TB_DIR" ]; then
        echo "No TensorBoard logs found at $TB_DIR"
        echo "Run training first."
        exit 1
    fi
    echo "Starting TensorBoard on port 6006..."
    echo "Access from your local machine:"
    echo "  ssh -L 6006:localhost:6006 $(whoami)@$(hostname -I | awk '{print $1}')"
    echo "Then open: http://localhost:6006"
    echo ""
    conda run -n ga2o3 tensorboard --logdir="$TB_DIR" --port=6006 --host=0.0.0.0
    exit 0
fi

# ── Status display ────────────────────────────────────────────────────────────
show_status() {
    clear
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Ga2O3-Net Training Monitor — $(date '+%Y-%m-%d %H:%M:%S')"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Active tmux sessions
    echo ""
    echo "  [Sessions]"
    if tmux ls 2>/dev/null | grep -q ga2o3; then
        tmux ls 2>/dev/null | grep ga2o3 | while read -r line; do
            echo "    ▸ $line"
        done
        echo "    Reattach: tmux attach -t <session_name>"
    else
        echo "    No active training sessions."
    fi

    # GPU status
    echo ""
    echo "  [GPU]"
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu \
            --format=csv,noheader,nounits 2>/dev/null | \
        while IFS=',' read -r idx name util mem_used mem_total temp; do
            printf "    GPU %s (%s): util=%s%%  mem=%s/%s MiB  temp=%s°C\n" \
                "$idx" "${name## }" "${util## }" "${mem_used## }" "${mem_total## }" "${temp## }"
        done
    else
        echo "    nvidia-smi not found"
    fi

    # Latest log
    echo ""
    echo "  [Latest Log]"
    LATEST_LOG=$(ls -t "$LOG_DIR"/*.log 2>/dev/null | head -1)
    if [ -n "$LATEST_LOG" ]; then
        echo "    File: $(basename "$LATEST_LOG")"
        echo "    Last 8 lines:"
        tail -8 "$LATEST_LOG" 2>/dev/null | while read -r line; do
            echo "      $line"
        done
    else
        echo "    No log files found in $LOG_DIR"
    fi

    # Latest CSV metrics (most recent run)
    echo ""
    echo "  [Metrics (latest run)]"
    LATEST_CSV=$(ls -t "$LOG_DIR"/*.csv 2>/dev/null | head -1)
    if [ -n "$LATEST_CSV" ]; then
        TOTAL=$(tail -n +2 "$LATEST_CSV" | wc -l)
        LAST=$(tail -1 "$LATEST_CSV" 2>/dev/null)
        if [ -n "$LAST" ]; then
            FOLD=$(echo "$LAST" | cut -d',' -f1)
            EPOCH=$(echo "$LAST" | cut -d',' -f2)
            TRAIN=$(echo "$LAST" | cut -d',' -f3)
            VAL=$(echo "$LAST" | cut -d',' -f4)
            MAE=$(echo "$LAST" | cut -d',' -f5)
            LR=$(echo "$LAST" | cut -d',' -f6)
            echo "    Run: $(basename "$LATEST_CSV" _metrics.csv)"
            echo "    Fold: $FOLD | Epoch: $EPOCH ($TOTAL total rows)"
            echo "    train_loss=$TRAIN | val_loss=$VAL | val_mae=$MAE eV | lr=$LR"
        fi
    else
        echo "    No metric CSV files found."
    fi

    # All runs summary (for comparison)
    CSV_COUNT=$(ls "$LOG_DIR"/*.csv 2>/dev/null | wc -l)
    if [ "$CSV_COUNT" -gt 1 ]; then
        echo ""
        echo "  [All Runs — Best Val Loss per Run]"
        for csv in $(ls -t "$LOG_DIR"/*.csv 2>/dev/null); do
            NAME=$(basename "$csv" _metrics.csv)
            BEST=$(tail -n +2 "$csv" | awk -F',' 'NR==1{min=$4} $4<min{min=$4} END{print min}')
            EPOCHS=$(tail -n +2 "$csv" | wc -l)
            printf "    %-35s  best_val=%-8s  epochs=%s\n" "$NAME" "$BEST" "$EPOCHS"
        done
    fi

    # TensorBoard hint
    echo ""
    echo "  [TensorBoard]"
    TB_DIR="$LOG_DIR/tensorboard"
    if [ -d "$TB_DIR" ]; then
        RUNS=$(ls "$TB_DIR" 2>/dev/null | wc -l)
        echo "    $RUNS run(s) available."
        echo "    Start server: bash scripts/monitor.sh --tb"
        echo "    SSH tunnel:   ssh -L 6006:localhost:6006 $(whoami)@$(hostname -I | awk '{print $1}' 2>/dev/null || echo '<server_ip>')"
        echo "    Open:         http://localhost:6006"
    else
        echo "    No TensorBoard data yet (run training first)."
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    if $WATCH; then
        echo "  Refreshing every ${REFRESH}s — Ctrl+C to stop"
    fi
}

# ── Main ──────────────────────────────────────────────────────────────────────
if $WATCH; then
    while true; do
        show_status
        sleep "$REFRESH"
    done
else
    show_status
fi
