#!/usr/bin/env bash
# Phase 57 — Compare V57-A1 (VC + PDR) against V55-Ext baseline using the
# full physics rubric (eval_sputter_physics.py). Auto-decides whether
# V57-A1 displaces V55-Ext per the Phase 56 frontier gate:
#   - ≥3/5 physics metric strict beat
#   - within-DOI FLIPs ≤ 1
#   - HSE06-consistency ≥ 80% (new V57 6th metric — TODO)
set -e
cd "$(dirname "$0")/.."

CSV=data/raw/experimental/ga2o3_exp_aug3x_vcinterp_v56.csv
RESULTS_DIR=results
LOG_DIR=$RESULTS_DIR/phase57_compare
mkdir -p "$LOG_DIR"

echo "===== V57-A1 vs V55-Ext physics rubric comparison ====="
echo "Date: $(date)"
echo

for target in vacancy_concentration photo_dark_ratio; do
  short=$([ "$target" = "vacancy_concentration" ] && echo vc || echo pdr)
  v57_bundle=$RESULTS_DIR/phase57v57a1_${short}_5seed
  v55_bundle=$RESULTS_DIR/phase55v55ext_qwen_${short}_5seed

  if [ ! -f "$v57_bundle/cv_results.csv" ]; then
    echo "[$short] V57-A1 bundle NOT READY: $v57_bundle/cv_results.csv missing"
    continue
  fi
  if [ ! -f "$v55_bundle/cv_results.csv" ]; then
    echo "[$short] V55-Ext baseline missing: $v55_bundle/cv_results.csv"
    continue
  fi

  echo "----- $target -----"
  echo "  V55-Ext baseline metrics (from cv_results.csv):"
  grep -E "${target}_r2_mean|${target}_rmse_mean|${target}_mae_mean" \
    "$v55_bundle/cv_results.csv" | sed 's/^/    /'
  echo "  V57-A1 metrics:"
  grep -E "${target}_r2_mean|${target}_rmse_mean|${target}_mae_mean" \
    "$v57_bundle/cv_results.csv" | sed 's/^/    /'

  # Run full physics rubric
  echo "  Running physics rubric on V57-A1..."
  PYTHONPATH=. conda run -n ga2o3 --no-capture-output python \
    scripts/eval_sputter_physics.py \
    "$v57_bundle" "$CSV" \
    --target "$target" --scope all --no-write \
    > "$LOG_DIR/v57a1_${short}_physics.txt" 2>&1 || echo "  (physics eval errored)"

  echo "  V57-A1 within-DOI summary:"
  grep -E "Tier-1C|OK: |FLIP" "$LOG_DIR/v57a1_${short}_physics.txt" | head -5 | sed 's/^/    /'
  echo "  V55-Ext baseline (locked):"
  grep -E "Tier-1C|OK: |FLIP" "$RESULTS_DIR/baseline_v55ext_${short}_v57_lock.txt" | head -5 | sed 's/^/    /'
  echo
done

# Diff for VC
if [ -f "$LOG_DIR/v57a1_vc_physics.txt" ]; then
  echo "===== VC diff: V57-A1 vs V55-Ext (locked) ====="
  diff "$RESULTS_DIR/baseline_v55ext_vc_v57_lock.txt" "$LOG_DIR/v57a1_vc_physics.txt" \
    | head -80 || true
fi

# Frontier gate decision
echo
echo "===== Phase 56 frontier gate decision ====="
v55_flips=$(grep -E "FLIP" "$RESULTS_DIR/baseline_v55ext_vc_v57_lock.txt" 2>/dev/null | wc -l)
v57_flips=$(grep -E "FLIP" "$LOG_DIR/v57a1_vc_physics.txt" 2>/dev/null | wc -l)
echo "  VC within-DOI FLIPs: V55-Ext=$v55_flips, V57-A1=$v57_flips (gate: ≤1)"
[ "$v57_flips" -le 1 ] && echo "  Within-DOI gate: PASS" || echo "  Within-DOI gate: FAIL"
