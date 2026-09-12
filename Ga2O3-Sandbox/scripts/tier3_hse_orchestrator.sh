#!/bin/bash
# Autonomous HSE transition-level campaign for the new dopants (Sb, Bi).
# Each HSE single-point is dual-GPU (~21.5 GB/card) so they run SERIALLY. ~overnight.
set +u
cd /home/lawrence/Physics/Ga2O3-Sandbox
LOG=logs/hse_dopant_orchestrator.log
echo "[$(date '+%H:%M:%S')] HSE dopant campaign START" | tee -a $LOG

conda run -n ga2o3 python scripts/tier3_hse_setup.py 2>&1 | tee -a $LOG

# serial dual-GPU HSE single-points (VBM first, then Sb, then Bi)
for job in hse_perfect_fixocc hse_dop_Sb_q0 hse_dop_Sb_q2 hse_dop_Bi_q0 hse_dop_Bi_q2; do
  echo "[$(date '+%H:%M:%S')] running $job (dual-GPU HSE)" | tee -a $LOG
  bash scripts/run_cc_qe_dual.sh $job >> $LOG 2>&1
  if grep -q "JOB DONE" dft/qe_cc/$job.out 2>/dev/null; then
    echo "[$(date '+%H:%M:%S')] $job DONE $(grep '^!' dft/qe_cc/$job.out | tail -1)" | tee -a $LOG
  else
    echo "[$(date '+%H:%M:%S')] $job FAILED (see dft/qe_cc/$job.out)" | tee -a $LOG
  fi
done

conda run -n ga2o3 python scripts/tier3_hse_assemble.py 2>&1 | tee -a $LOG
echo "[$(date '+%H:%M:%S')] HSE_DOPANT_CAMPAIGN_DONE" | tee -a $LOG
