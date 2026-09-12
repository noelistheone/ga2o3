#!/bin/bash
cd /home/lawrence/Physics/Ga2O3-Sandbox
bash scripts/run_cc_qe.sh cc_VGa_q0_at_R-3 1
bash scripts/run_cc_qe.sh cc_VGa_q-3_at_R0 1
echo "VGA_CROSS_DONE" >> logs/vga_cross.log
