#!/bin/bash
# R6 follow-on GPU queue (serial; dual-GPU runner handles both PBE and HSE).
cd /home/lawrence/Physics/Ga2O3-Sandbox
for job in kpt_VO_q0_gamma kpt_VO_q0_222 kpt_VO_q2_gamma kpt_VO_q2_222 \
           hse_dop_Sb_q1 hse_dop_Sb_q1_a033 hse_dop_Bi_q1 hse_dop_Bi_q1_a033 \
           hse_VO_q2_scan_l085 hse_VO_q2_scan_l095 hse_VO_q2_scan_l100 \
           hse_VO_q2_scan_l105 hse_VO_q2_scan_l115; do
  bash scripts/run_cc_qe_dual.sh "$job"
done
echo R6_GPU_QUEUE_DONE
