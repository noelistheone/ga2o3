#!/usr/bin/env bash
# launch.sh — Start a training script inside a persistent tmux session.
#
# The tmux session survives SSH disconnection. Training output is written to
# both the terminal (inside tmux) and a log file under logs/.
#
# Usage:
#   bash scripts/launch.sh <stage> [args...]
#
# Examples:
#   bash scripts/launch.sh 01
#   bash scripts/launch.sh 02 --run-name lr1e3_batch256 --cv-folds 5
#   bash scripts/launch.sh 02 --run-name deeper_4layer
#
# Session commands:
#   Detach (keep running):  Ctrl+B, then D
#   Reattach:               tmux attach -t ga2o3_<stage>
#   List sessions:          tmux ls
#   Kill a session:         tmux kill-session -t ga2o3_<stage>

set -euo pipefail

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="${1:?Usage: bash scripts/launch.sh <01|02|03|04> [args...]}"
shift
ARGS="${*:-}"
SESSION="ga2o3_${STAGE}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/stage${STAGE}_${TIMESTAMP}.log"

if ! command -v tmux &>/dev/null; then
    echo "ERROR: tmux not found. Install: sudo apt-get install tmux"; exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo ""
    echo "  Session '$SESSION' is already running."
    echo "  Reattach: tmux attach -t $SESSION"
    echo "  Kill first if you want to restart: tmux kill-session -t $SESSION"
    echo ""
    exit 1
fi

# Build the command
CMD="cd '$PROJECT' && conda run -n ga2o3 python -u scripts/${STAGE}_*.py $ARGS 2>&1 | tee '$LOG'; echo '=== DONE ===' | tee -a '$LOG'"

tmux new-session -d -s "$SESSION" -x 220 -y 50
tmux send-keys -t "$SESSION" "$CMD" Enter

echo ""
echo "  ┌─────────────────────────────────────────────────────┐"
echo "  │  Stage $STAGE started in tmux session: $SESSION"
echo "  │  Log: $LOG"
echo "  ├─────────────────────────────────────────────────────┤"
echo "  │  Reattach:  tmux attach -t $SESSION"
echo "  │  Detach:    Ctrl+B then D  (training keeps running)"
echo "  │  Monitor:   bash scripts/monitor.sh"
echo "  └─────────────────────────────────────────────────────┘"
echo ""
echo "Auto-attaching (Ctrl+C to skip)..."
sleep 1
tmux attach -t "$SESSION"
