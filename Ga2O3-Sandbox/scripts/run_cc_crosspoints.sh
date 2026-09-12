#!/bin/bash
cd /home/lawrence/Physics/Ga2O3-Sandbox
bash scripts/run_cc_qe.sh cc_VO_q0_at_R2 1
bash scripts/run_cc_qe.sh cc_VO_q2_at_R0 1
echo "CROSSPOINTS_DONE" >> logs/cc_crosspoints.log
