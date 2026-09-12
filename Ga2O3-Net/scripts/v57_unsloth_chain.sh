#!/usr/bin/env bash
# V57 Unsloth chain: CPT-DFT → ICL-V3 → DPO-DFT → V57-A1 retrain.
# Run AFTER vLLM is killed and the V57-Ext-3 / Coscientist / SR-Evolve
# stages have already produced their on-disk outputs.

set -euo pipefail
cd /home/lawrence/Physics/Ga2O3-Net

export CUDA_HOME=/home/lawrence/anaconda3/envs/ga2o3
export PATH=$CUDA_HOME/bin:$PATH
SP_NVIDIA=/home/lawrence/anaconda3/envs/ga2o3/lib/python3.11/site-packages/nvidia
export CPATH=$SP_NVIDIA/cu13/include:$SP_NVIDIA/curand/include:$SP_NVIDIA/cublas/include:$SP_NVIDIA/cusparse/include:$SP_NVIDIA/cusolver/include:$SP_NVIDIA/cudnn/include:${CPATH:-}
export LIBRARY_PATH=$SP_NVIDIA/cu13/lib:$SP_NVIDIA/curand/lib:$SP_NVIDIA/cublas/lib:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=$SP_NVIDIA/cu13/lib:${LD_LIBRARY_PATH:-}

PY=/home/lawrence/anaconda3/envs/ga2o3/bin/python
LOG_DIR=logs
mkdir -p $LOG_DIR

# Sanity: vLLM must be off
if pgrep -fa "VLLM::EngineCore\|VLLM::Worker_TP" > /dev/null; then
  echo "ERROR: vLLM still running. Kill it first."
  pgrep -fa "VLLM::"
  exit 1
fi

echo "[$(date)] V57-CPT-DFT starting (block_size=2048 to fit 24GB)"
PYTHONPATH=. $PY -u scripts/32_v57_cpt_dft.py \
    --block-size 2048 \
    --epochs 1 --lr 2e-5 \
    --lora-r 32 --lora-alpha 64 \
    --batch-size 1 --grad-accum 8 \
    --save-steps 200 \
    > $LOG_DIR/v57_cpt_dft.log 2>&1
echo "[$(date)] V57-CPT-DFT done"

echo "[$(date)] V57-ICL-V3 starting (init from CPT adapter)"
PYTHONPATH=. $PY -u scripts/33_v57_icl_v3.py \
    --use-cpt \
    --epochs 2 --lr 1e-4 \
    --max-seq-len 1024 --batch-size 1 --grad-accum 8 \
    > $LOG_DIR/v57_icl_v3.log 2>&1
echo "[$(date)] V57-ICL-V3 done"

echo "[$(date)] V57-DPO-DFT starting"
PYTHONPATH=. $PY -u scripts/38_v57_dpo_dft.py \
    --n-pairs 200 --epochs 1 --beta 0.1 \
    > $LOG_DIR/v57_dpo_dft.log 2>&1
echo "[$(date)] V57-DPO-DFT done"

echo "[$(date)] Unsloth chain complete. Ready for V57-A1 retrain on enhanced CSV."
