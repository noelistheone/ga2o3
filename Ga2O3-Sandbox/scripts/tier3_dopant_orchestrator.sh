#!/bin/bash
# Autonomous new-dopant (Sb/Bi) charged-state campaign.
# Waits for the neutral relaxes, then runs q=+1/+2 short relaxes (Sb on GPU0, Bi on GPU1,
# serial per card), then assembles transition levels + donor/acceptor classification.
set +u
cd /home/lawrence/Physics/Ga2O3-Sandbox
LOG=logs/dopant_orchestrator.log
CC=dft/qe_cc
echo "[$(date '+%H:%M:%S')] orchestrator START" | tee -a $LOG

wait_done () {  # $1 = prefix
  while ! grep -q "JOB DONE" $CC/$1.out 2>/dev/null; do sleep 30; done
  echo "[$(date '+%H:%M:%S')] $1 DONE" | tee -a $LOG
}

# 1. wait for the two neutral relaxations
wait_done dop_Sb_relax
wait_done dop_Bi_relax
echo "[$(date '+%H:%M:%S')] both neutral relaxes done -> setting up charged states" | tee -a $LOG

# 2. set up charged inputs from the relaxed neutral geoms
conda run -n ga2o3 python scripts/tier3_dopant_charged_setup.py Sb 2>&1 | tee -a $LOG
conda run -n ga2o3 python scripts/tier3_dopant_charged_setup.py Bi 2>&1 | tee -a $LOG

# 3. run charged states: Sb q1,q2 on GPU0 ; Bi q1,q2 on GPU1 (serial within each card)
( bash scripts/run_cc_qe.sh dop_Sb_q1 0 && bash scripts/run_cc_qe.sh dop_Sb_q2 0 ) >> $LOG 2>&1 &
P0=$!
( bash scripts/run_cc_qe.sh dop_Bi_q1 1 && bash scripts/run_cc_qe.sh dop_Bi_q2 1 ) >> $LOG 2>&1 &
P1=$!
wait $P0 $P1
echo "[$(date '+%H:%M:%S')] all charged states done" | tee -a $LOG

# 4. assemble transition levels + classification
conda run -n ga2o3 python scripts/tier3_dopant_assemble.py 2>&1 | tee -a $LOG

# 5. close the loop: inject into KROGER DB -> full 9-property forward prediction
conda run -n ga2o3 python scripts/tier3_new_dopant_predict.py 2>&1 | tee -a $LOG
echo "[$(date '+%H:%M:%S')] DOPANT_CAMPAIGN_DONE" | tee -a $LOG
