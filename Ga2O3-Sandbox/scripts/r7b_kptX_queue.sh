#!/bin/bash
# R7 plan B: zone-boundary (1/2,1/2,1/2) single-k dispersion at PBE and HSE tiers, Sb q0/q2.
cd /home/lawrence/Physics/Ga2O3-Sandbox
for job in kptX_pbe_Sb_q0 kptX_pbe_Sb_q2 kptX_hse_Sb_q0 kptX_hse_Sb_q2; do
  bash scripts/run_cc_qe_dual.sh "$job"
done
echo R7B_KPTX_QUEUE_DONE
