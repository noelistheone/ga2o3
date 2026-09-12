#!/bin/bash
# tau CC-diagram QE-GPU driver. Usage: run_cc_qe.sh <prefix> <gpu_id>
# Relaxes a charged V_O supercell (PBE) on one GPU. Resume-safe (skips JOB DONE).
set +u
CC_DIR=/home/lawrence/Physics/Ga2O3-Sandbox/dft/qe_cc
LOG_DIR=/home/lawrence/Physics/Ga2O3-Sandbox/logs
mkdir -p "$CC_DIR/outdir" "$LOG_DIR"

prefix=$1
gpu=$2
in_file="$CC_DIR/${prefix}.in"
out_file="$CC_DIR/${prefix}.out"
log_file="$LOG_DIR/cc_${prefix}.log"

source /home/lawrence/qe-gpu-build/setup_env.sh
PW_X=/home/lawrence/qe-gpu-build/q-e-qe-7.5/bin/pw.x
[ -x "$PW_X" ] || { echo "FATAL: GPU pw.x not found" | tee -a "$log_file"; exit 2; }

if [ -f "$out_file" ] && grep -q "JOB DONE" "$out_file" 2>/dev/null; then
    echo "[$(date '+%H:%M:%S')] SKIP ${prefix} (done)" | tee -a "$log_file"; exit 0
fi
rm -rf "$CC_DIR/outdir/${prefix}" 2>/dev/null
echo "[$(date '+%H:%M:%S')] START ${prefix} on GPU${gpu}" | tee -a "$log_file"
cd "$CC_DIR"
CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=8 \
    mpirun -np 1 "$PW_X" -in "${prefix}.in" > "$out_file" 2>&1
if grep -q "JOB DONE" "$out_file" 2>/dev/null; then
    E=$(grep '^!' "$out_file" | tail -1)
    echo "[$(date '+%H:%M:%S')] DONE ${prefix}  ${E}" | tee -a "$log_file"
else
    echo "[$(date '+%H:%M:%S')] FAIL/INCOMPLETE ${prefix} (see $out_file)" | tee -a "$log_file"
fi
