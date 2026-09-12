#!/bin/bash
# Autonomous disorder-dEg DOS campaign. Waits for the MACE melt-quench ensemble, generates
# QE-PBE DOS inputs (crystal ref + amorphous cells), runs them 2-at-a-time (one per GPU),
# then extracts the crystal->amorphous gap narrowing + Urbach energy.
set +u
PROJ=/home/lawrence/Physics/Ga2O3-Sandbox
cd "$PROJ"
LOG=logs/disorder_orchestrator.log
ENS=dft/disorder_ensemble
CC=dft/qe_cc
echo "[$(date '+%H:%M:%S')] disorder DOS campaign START" | tee -a $LOG

# 1. wait for the ensemble to finish (summary has GaO_std_ensemble_mean at the end)
while ! grep -q "GaO_std_ensemble_mean" "$ENS/ensemble_summary.json" 2>/dev/null; do sleep 30; done
echo "[$(date '+%H:%M:%S')] ensemble ready ($(ls $ENS/cell_*.xyz | wc -l) cells)" | tee -a $LOG

# 2. generate DOS inputs (crystal + all cells)
conda run -n ga2o3 python scripts/tier3_disorder_dos_gen.py 2>&1 | tee -a $LOG

# 3. run DOS calcs 2-at-a-time (GPU0 + GPU1). Build the job list: crystal first, then cells.
jobs=(crystal $(ls $ENS/cell_*.xyz | sed 's#.*/##; s/\.xyz//'))
i=0
while [ $i -lt ${#jobs[@]} ]; do
  j0="dos_${jobs[$i]}"; j1="dos_${jobs[$((i+1))]}"
  echo "[$(date '+%H:%M:%S')] DOS $j0 (GPU0) + ${jobs[$((i+1))]:+$j1 (GPU1)}" | tee -a $LOG
  bash scripts/run_cc_qe.sh "$j0" 0 >> $LOG 2>&1 &
  P0=$!
  P1=""
  if [ $((i+1)) -lt ${#jobs[@]} ]; then bash scripts/run_cc_qe.sh "$j1" 1 >> $LOG 2>&1 & P1=$!; fi
  wait $P0 ${P1:+$P1}
  i=$((i+2))
done
echo "[$(date '+%H:%M:%S')] all DOS done" | tee -a $LOG

# 4. extract disorder-dEg + Urbach
conda run -n ga2o3 python scripts/tier3_disorder_edge.py 2>&1 | tee -a $LOG
echo "[$(date '+%H:%M:%S')] DISORDER_DEG_CAMPAIGN_DONE" | tee -a $LOG
