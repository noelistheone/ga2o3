#!/bin/bash
# R7: HSE k-point spot check for Sb (reviewer #2 optional item) — serial dual-GPU.
cd /home/lawrence/Physics/Ga2O3-Sandbox
for job in kpt_hse_Sb_q0_222 kpt_hse_Sb_q2_222; do
  bash scripts/run_cc_qe_dual.sh "$job"
done
echo R7_HSE_KPT_QUEUE_DONE
