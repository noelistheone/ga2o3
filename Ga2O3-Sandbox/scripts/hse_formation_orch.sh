#!/bin/bash
cd /home/lawrence/Physics/Ga2O3-Sandbox
OL=logs/orchestrator.log
for p in hse_VO_q0 hse_VO_q2 hse_O2; do
  bash scripts/run_cc_qe_dual.sh $p >/dev/null 2>&1
  grep -q "JOB DONE" dft/qe_cc/$p.out 2>/dev/null && echo "${p}_DONE $(date +%H:%M)" >> $OL || echo "${p}_FAIL $(date +%H:%M)" >> $OL
done
echo "HSE_FORMATION_DONE $(date +%H:%M)" >> $OL
