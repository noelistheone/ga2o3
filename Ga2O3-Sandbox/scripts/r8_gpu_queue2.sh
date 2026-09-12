#!/bin/bash
# R8 wave-G chunk 2 (serial dual-GPU): CC cross points (TAU-08), Nb chain PBE relaxes
# (SB-17a positive control), Sb tetrahedral ladder PBE relaxes (SB-17b), alpha=0.33 refs (A33-40).
cd /home/lawrence/Physics/Ga2O3-Sandbox
for job in cc_VO_q2_at_R1 cc_VO_q1_at_R2 cc_VO_q1_at_R0 cc_VO_q0_at_R1 \
           cc_VGa_q0_at_R-1 cc_VGa_q-1_at_R0 cc_VGa_q-1_at_R-2 cc_VGa_q-2_at_R-1 \
           cc_VGa_q-2_at_R-3 cc_VGa_q-3_at_R-2 \
           hse_O2_a033 gate2_perfect_hse_a033 \
           dop_Nb_relax dop_Nb_q1_relax dop_Nb_q2_relax \
           dop_Sbtet_relax dop_Sbtet_q1_relax dop_Sbtet_q2_relax; do
  bash scripts/run_cc_qe_dual.sh "$job"
done
echo R8_GPU_QUEUE2_DONE
