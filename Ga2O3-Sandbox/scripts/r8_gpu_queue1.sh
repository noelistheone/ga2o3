#!/bin/bash
# R8 wave-G chunk 1 (serial dual-GPU): sequential-capture relaxes (TAU-08), DFT site-preference
# singles (SB-17b), spin-stability perturbed relaxes (SPIN-09).
cd /home/lawrence/Physics/Ga2O3-Sandbox
for job in dop_Sb_Ga_I_tet_at_mace dop_Sb_Ga_II_oct_at_mace \
           dop_Bi_Ga_I_tet_at_mace dop_Bi_Ga_II_oct_at_mace \
           cc_VO_q1_relax cc_VGa_q-1_relax cc_VGa_q-2_relax \
           spin_Sb_q1_pert spin_Bi_q1_pert; do
  bash scripts/run_cc_qe_dual.sh "$job"
done
echo R8_GPU_QUEUE1_DONE
