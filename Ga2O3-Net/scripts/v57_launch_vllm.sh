#!/usr/bin/env bash
# V57 — vLLM Qwen3-30B-A3B launcher (dual RTX 3090, TP=2, no docker).
#
# Usage:
#   ./scripts/v57_launch_vllm.sh thinking   # for Coscientist / SR-Evolve / Ext-3 validate
#   ./scripts/v57_launch_vllm.sh instruct   # for Ext-3 structured extraction
#   ./scripts/v57_launch_vllm.sh vl         # for Ext-3 multimodal triage/extract
#
# Kill with:  pkill -f "vllm serve"

set -euo pipefail

cd "$(dirname "$0")/.."
MODE="${1:-thinking}"

case "$MODE" in
  thinking)
    MODEL_PATH="checkpoints/qwen3-30b-a3b-thinking-awq"
    SERVED_NAME="cpatonn/Qwen3-30B-A3B-Thinking-2507-AWQ-4bit"
    EXTRA="--reasoning-parser deepseek_r1"
    MAX_LEN=32768
    ;;
  instruct)
    MODEL_PATH="checkpoints/qwen3-30b-a3b-instruct-awq"
    SERVED_NAME="cpatonn/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit"
    EXTRA=""
    MAX_LEN=32768
    ;;
  vl)
    MODEL_PATH="checkpoints/qwen3-vl-30b-instruct-awq"
    SERVED_NAME="cpatonn/Qwen3-VL-30B-A3B-Instruct-AWQ-4bit"
    EXTRA="--limit-mm-per-prompt image=4"
    MAX_LEN=16384
    ;;
  *)
    echo "Usage: $0 {thinking|instruct|vl}" >&2
    exit 2
    ;;
esac

if [ ! -d "$MODEL_PATH" ]; then
  echo "Model not found: $MODEL_PATH" >&2
  exit 3
fi

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export CUDA_VISIBLE_DEVICES=0,1
# vLLM 0.21 needs CUDA 13 runtime; site-packages ships it but it's not on
# LD_LIBRARY_PATH by default. Prepend so libcudart.so.13 is findable.
export LD_LIBRARY_PATH=/home/lawrence/anaconda3/envs/ga2o3/lib/python3.11/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
# vLLM triton sampler needs nvcc — installed via `conda install -n ga2o3 -c nvidia cuda-nvcc`.
export CUDA_HOME=/home/lawrence/anaconda3/envs/ga2o3
export PATH=$CUDA_HOME/bin:$PATH
# FlashInfer's JIT compile of sampler kernels needs curand.h etc., which the
# nvidia/cu13 pip wheel ships under nvidia/cu13/include. nvcc by default only
# searches $CUDA_HOME/include — point it at the wheel include dir too.
SP_NVIDIA=/home/lawrence/anaconda3/envs/ga2o3/lib/python3.11/site-packages/nvidia
export CPATH=$SP_NVIDIA/cu13/include:$SP_NVIDIA/curand/include:$SP_NVIDIA/cublas/include:$SP_NVIDIA/cusparse/include:$SP_NVIDIA/cusolver/include:$SP_NVIDIA/cudnn/include:${CPATH:-}
export LIBRARY_PATH=$SP_NVIDIA/cu13/lib:$SP_NVIDIA/curand/lib:$SP_NVIDIA/cublas/lib:${LIBRARY_PATH:-}

echo "[v57-vllm] Launching $MODE backbone (TP=2 on GPU 0+1, AWQ-4bit, max_len=$MAX_LEN)"
echo "[v57-vllm] Model:  $MODEL_PATH"
echo "[v57-vllm] Served as: $SERVED_NAME"

# Throughput-tuned defaults — 24GB × 2, max_num_seqs=8 keeps the decode
# pipeline saturated for concurrent client requests (V57-Ext-3 658 PDFs).
# CUDA graphs enabled (eager off) since the first-run flashinfer kernel
# compile already populated the cache.
exec /home/lawrence/anaconda3/envs/ga2o3/bin/vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED_NAME" \
  --quantization compressed-tensors \
  --dtype bfloat16 \
  --tensor-parallel-size 2 \
  --max-model-len "$MAX_LEN" \
  --max-num-seqs 8 \
  --max-num-batched-tokens 16384 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.90 \
  --disable-custom-all-reduce \
  $EXTRA \
  --host 0.0.0.0 \
  --port 8000
