#!/bin/bash
# Autonomous endgame: finish V_Ga τ + Fe_Ga τ, then HSE-perfect energetics (dual-GPU).
cd /home/lawrence/Physics/Ga2O3-Sandbox
OL=logs/orchestrator.log
CONDA="conda run -n ga2o3 python"
wd() { while ! grep -q "JOB DONE" "dft/qe_cc/$1.out" 2>/dev/null; do sleep 60; done; }

# --- 1. V_Ga τ (cross-points launched on GPU1) ---
wd cc_VGa_q0_at_R-3; wd cc_VGa_q-3_at_R0
$CONDA scripts/cc_tau_general.py VGa 0 -3 0.8 1e17 >/dev/null 2>&1 && echo "VGA_TAU_DONE $(date +%H:%M)" >> $OL

# --- 2. Fe_Ga τ (q0 done, q-1 launched on GPU0) ---
wd cc_FeGa_q-1_relax
$CONDA scripts/cc_process_general.py FeGa 0 -1 >/dev/null 2>&1
# Fe_Ga cross-points serial on GPU0 (V_Ga freed GPU1 already; use GPU0)
bash scripts/run_cc_qe.sh cc_FeGa_q0_at_R-1 0 >/dev/null 2>&1
bash scripts/run_cc_qe.sh cc_FeGa_q-1_at_R0 0 >/dev/null 2>&1
$CONDA scripts/cc_tau_general.py FeGa 0 -1 0.86 1e17 >/dev/null 2>&1 && echo "FEGA_TAU_DONE $(date +%H:%M)" >> $OL

echo "ALL_TAU_DONE $(date +%H:%M)" >> $OL

# --- 3. HSE-perfect energetics (dual-GPU, both cards now free) ---
bash scripts/run_cc_qe_dual.sh gate2_perfect_hse >/dev/null 2>&1
grep -q "JOB DONE" dft/qe_cc/gate2_perfect_hse.out 2>/dev/null && echo "HSE_PERFECT_DONE $(date +%H:%M)" >> $OL || echo "HSE_PERFECT_FAIL $(date +%H:%M)" >> $OL

echo "ORCHESTRATOR_DONE $(date +%H:%M)" >> $OL
