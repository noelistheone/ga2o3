#!/bin/bash
# Dual-GPU QE runner for HSE (memory distributed across both RTX 3090s). Usage: <prefix>
set +u
CC=/home/lawrence/Physics/Ga2O3-Sandbox/dft/qe_cc
LOG=/home/lawrence/Physics/Ga2O3-Sandbox/logs
prefix=$1
out="$CC/${prefix}.out"
[ -f "$out" ] && grep -q "JOB DONE" "$out" 2>/dev/null && { echo "SKIP $prefix"; exit 0; }
source /home/lawrence/qe-gpu-build/setup_env.sh
PW=/home/lawrence/qe-gpu-build/q-e-qe-7.5/bin/pw.x
rm -rf "$CC/outdir/${prefix}" 2>/dev/null
cd "$CC"
echo "[$(date '+%H:%M:%S')] START(dual) $prefix" >> "$LOG/dual.log"
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 mpirun -np 2 "$PW" -in "${prefix}.in" > "$out" 2>&1
grep -q "JOB DONE" "$out" && echo "[$(date '+%H:%M:%S')] DONE(dual) $prefix" >> "$LOG/dual.log" || echo "[$(date '+%H:%M:%S')] FAIL(dual) $prefix" >> "$LOG/dual.log"
