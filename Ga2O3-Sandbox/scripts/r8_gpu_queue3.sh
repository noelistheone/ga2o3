#!/bin/bash
# R8 queue 3: HSE single points for the Nb positive control and the Sb tetrahedral ladder.
cd /home/lawrence/Physics/Ga2O3-Sandbox
for job in hse_dop_Nb_q0 hse_dop_Nb_q1 hse_dop_Nb_q2 \
           hse_dop_Sbtet_q0 hse_dop_Sbtet_q1 hse_dop_Sbtet_q2; do
  bash scripts/run_cc_qe_dual.sh "$job"
done
echo R8_GPU_QUEUE3_DONE
