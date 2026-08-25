#!/bin/bash
#SBATCH --job-name=agentic-sfm-grpo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/kcwp264/logs/agentic-sfm/grpo_%j.out
#SBATCH --error=/scratch/kcwp264/logs/agentic-sfm/grpo_%j.err

set -euo pipefail

echo "=== Agentic-SFM Phase 1 GRPO Training ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)"
echo "Start: $(date)"

# Paths
PROJECT_ROOT="/scratch/kcwp264/agentic-sfm"
CONFIG="${PROJECT_ROOT}/configs/phase1_grpo.yaml"
OUTPUT_DIR="/scratch/kcwp264/agentic-sfm/outputs/phase1"
LOG_DIR="/scratch/kcwp264/logs/agentic-sfm"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

# Environment
export PATH="/scratch/kcwp264/.conda_envs/agentic-sfm/bin:$PATH"
export CONDA_PREFIX="/scratch/kcwp264/.conda_envs/agentic-sfm"
module load cuda/12.6.2 2>/dev/null || true
export CUDA_HOME="${CUDA_HOME:-/opt/apps/pkg/compilers/cuda/12.6.2}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export NCCL_P2P_DISABLE=1
export NCCL_NET=Socket
export NCCL_IB_DISABLE=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Redirect all caches to scratch to avoid home disk quota issues
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

# Start vLLM server on GPU 0
echo "Starting vLLM server on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model "Qwen/Qwen3-VL-8B-Instruct" \
    --port 8000 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --limit-mm-per-prompt '{"image":2,"video":0}' \
    --mm-processor-kwargs '{"max_pixels":1003520,"min_pixels":3136}' \
    --mm-processor-cache-gb 0 \
    --trust-remote-code \
    --dtype bfloat16 \
    --api-key EMPTY \
    --enforce-eager &
VLLM_PID=$!

# Wait for vLLM to be ready
echo "Waiting for vLLM..."
for i in $(seq 1 240); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "vLLM ready (took ${i}x2s = $((i*2))s)"
        break
    fi
    if [ $i -eq 240 ]; then
        echo "ERROR: vLLM failed to start after 480s"
        echo "=== GPU status ==="
        nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || true
        echo "=== vLLM process ==="
        ps aux | grep vllm | grep -v grep || true
        kill ${VLLM_PID} 2>/dev/null || true
        exit 1
    fi
    if [ $((i % 30)) -eq 0 ]; then
        echo "  Still waiting for vLLM... (${i}x2s = $((i*2))s elapsed)"
    fi
    sleep 2
done

# Start tool server on GPU 1 (shared with training — matcher uses minimal VRAM)
echo "Starting tool server on GPU 1..."
CUDA_VISIBLE_DEVICES=1 PYTHONPATH="${PROJECT_ROOT}/src" python -m uvicorn \
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

# Run GRPO training on GPU 1 (shared with tool server)
echo "Starting GRPO training..."
CUDA_VISIBLE_DEVICES=1 python scripts/run_grpo.py \
    --config "${CONFIG}" \
    --tool-server-url "http://localhost:8765" \
    --vllm-url "http://localhost:8000" \
    --output-dir "${OUTPUT_DIR}" \
    2>&1 | tee "${LOG_DIR}/grpo_training_${SLURM_JOB_ID}.log"

TRAIN_EXIT=$?

# Cleanup
echo "Cleaning up servers..."
kill ${VLLM_PID} ${TOOL_PID} 2>/dev/null || true
wait ${VLLM_PID} ${TOOL_PID} 2>/dev/null || true

echo "Training exit code: ${TRAIN_EXIT}"
echo "End: $(date)"
exit ${TRAIN_EXIT}
