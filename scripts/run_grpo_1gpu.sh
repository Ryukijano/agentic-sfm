#!/bin/bash
# 1-GPU GRPO launcher — runs vLLM + tool server + training all on GPU 0
# Usage: run this inside an srun interactive session with 1 GPU
set -euo pipefail

PROJECT_ROOT="/scratch/kcwp264/agentic-sfm"
CONFIG="${PROJECT_ROOT}/configs/phase1_grpo.yaml"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/phase1"
LOG_DIR="/scratch/kcwp264/logs/agentic-sfm"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

# Environment
export PATH="/scratch/kcwp264/.conda_envs/agentic-sfm/bin:$PATH"
export CONDA_PREFIX="/scratch/kcwp264/.conda_envs/agentic-sfm"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export NCCL_P2P_DISABLE=1
export NCCL_NET=Socket
export NCCL_IB_DISABLE=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TRITON_CACHE_DIR="/scratch/kcwp264/.triton_cache"
export FLASHINFER_WORKSPACE_DIR="/scratch/kcwp264/.flashinfer"
export FLASHINFER_JIT_DIR="/scratch/kcwp264/.flashinfer/jit"
export XDG_CACHE_HOME="/scratch/kcwp264/.cache/xdg"
export HF_HOME="/scratch/kcwp264/.cache/huggingface"
export TORCH_HOME="/scratch/kcwp264/.cache/torch"
export VLLM_CACHE_ROOT="/scratch/kcwp264/.cache/vllm"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
mkdir -p "${TRITON_CACHE_DIR}" "${FLASHINFER_WORKSPACE_DIR}" "${FLASHINFER_JIT_DIR}" "${XDG_CACHE_HOME}" "${VLLM_CACHE_ROOT}"

cd "${PROJECT_ROOT}"

echo "=== 1-GPU GRPO Training ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Start: $(date)"

# Start vLLM on GPU 0 (Qwen3-VL-2B-Instruct ~8GB weights; rest of the 48GB card is for KV + training on GPU 1)
echo "Starting vLLM server on GPU 0 (shared memory mode)..."
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model "Qwen/Qwen3-VL-2B-Instruct" \
    --enable-lora \
    --max-loras 1 \
    --max-lora-rank 32 \
    --port 8000 \
    --gpu-memory-utilization 0.55 \
    --max-model-len 3072 \
    --max-num-seqs 4 \
    --limit-mm-per-prompt '{"image":2,"video":0}' \
    --mm-processor-kwargs '{"max_pixels":501760,"min_pixels":3136}' \
    --mm-processor-cache-gb 0 \
    --trust-remote-code \
    --dtype bfloat16 \
    --api-key EMPTY \
    --enforce-eager &
VLLM_PID=$!

# Wait for vLLM
echo "Waiting for vLLM..."
for i in $(seq 1 240); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "vLLM ready (took ${i}x2s = $((i*2))s)"
        break
    fi
    if [ $i -eq 240 ]; then
        echo "ERROR: vLLM failed to start after 480s"
        nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
        kill ${VLLM_PID} 2>/dev/null || true
        exit 1
    fi
    if [ $((i % 30)) -eq 0 ]; then
        echo "  Still waiting for vLLM... (${i}x2s = $((i*2))s elapsed)"
    fi
    sleep 2
done

# Start tool server on GPU 0 (CPU-only, minimal VRAM)
echo "Starting tool server..."
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="${PROJECT_ROOT}/src" python -m uvicorn \
    agentic_sfm.tools.server:app \
    --host 0.0.0.0 \
    --port 8765 \
    --workers 1 &
TOOL_PID=$!

# Wait for tool server
echo "Waiting for tool server..."
for i in $(seq 1 30); do
    if curl -s http://localhost:8765/health > /dev/null 2>&1; then
        echo "Tool server ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "ERROR: Tool server failed to start"
        kill ${VLLM_PID} ${TOOL_PID} 2>/dev/null || true
        exit 1
    fi
    sleep 2
done

# Run GRPO training on GPU 0 (shared with vLLM)
echo "Starting GRPO training on GPU 0 (shared with vLLM)..."
CUDA_VISIBLE_DEVICES=0 python scripts/run_grpo.py \
    --config "${CONFIG}" \
    --tool-server-url "http://localhost:8765" \
    --vllm-url "http://localhost:8000" \
    --output-dir "${OUTPUT_DIR}" \
    2>&1 | tee "${LOG_DIR}/grpo_1gpu_$(date +%Y%m%d_%H%M%S).log"

TRAIN_EXIT=$?

# Cleanup
echo "Cleaning up servers..."
kill ${VLLM_PID} ${TOOL_PID} 2>/dev/null || true
wait ${VLLM_PID} ${TOOL_PID} 2>/dev/null || true

echo "Training exit code: ${TRAIN_EXIT}"
echo "End: $(date)"
exit ${TRAIN_EXIT}
