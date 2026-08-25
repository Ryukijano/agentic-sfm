#!/usr/bin/env bash
# =============================================================================
# Agentic-SFM GRPO Training — 3-GPU interactive launcher
#
# GPU 0: vLLM rollout server (Qwen3-VL-8B)
# GPU 1: Training (LoRA policy gradient)
# GPU 2: Tool server (LoFTR matcher + crop + doppelganger)
#
# Usage:
#   bash scripts/run_grpo_3gpu.sh                    # full training
#   bash scripts/run_grpo_3gpu.sh --smoke             # smoke test
#   bash scripts/run_grpo_3gpu.sh --config configs/phase1_grpo.yaml
# =============================================================================
set -euo pipefail

# --- Activate conda env by full path (not discoverable by name) ---
AGENTIC_ENV_PATH="${AGENTIC_ENV_PATH:-/scratch/kcwp264/.conda_envs/agentic-sfm}"
if [ "${CONDA_DEFAULT_ENV:-}" != "${AGENTIC_ENV_PATH}" ]; then
  for conda_root in \
    "/opt/apps/pkg/interpreters/miniforge/24.7.1" \
    "/scratch/kcwp264/.conda_envs" \
    "${HOME}/miniforge3" \
    "/opt/miniforge3"; do
    if [ -f "${conda_root}/etc/profile.d/conda.sh" ]; then
      source "${conda_root}/etc/profile.d/conda.sh"
      break
    fi
  done
  conda activate "${AGENTIC_ENV_PATH}" || { echo "Could not activate ${AGENTIC_ENV_PATH}"; exit 1; }
fi

cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export HF_HOME="/scratch/kcwp264/.cache/huggingface"
export TORCH_HOME="/scratch/kcwp264/.cache/torch"

# --- Load CUDA toolkit (needed by vLLM flashinfer JIT) ---
module load cuda/12.6.2 2>/dev/null || true
export CUDA_HOME="${CUDA_HOME:-/opt/apps/pkg/compilers/cuda/12.6.2}"

# --- Parse args ---
CONFIG="configs/phase1_grpo.yaml"
SMOKE=0
for arg in "$@"; do
  case "$arg" in
    --smoke) SMOKE=1; CONFIG="configs/phase1_grpo_smoke.yaml" ;;
    --config) shift; CONFIG="$1" ;;
    --config=*) CONFIG="${arg#--config=}" ;;
  esac
done

echo "============================================================"
echo "  Agentic-SFM GRPO Training"
echo "  Config: ${CONFIG}"
echo "  Smoke:  ${SMOKE}"
echo "  Start:  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# --- Check GPU availability ---
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo "GPUs visible: ${NUM_GPUS}"
if [ "${NUM_GPUS}" -lt 3 ]; then
  echo "WARNING: Expected 3 GPUs, got ${NUM_GPUS}. Servers will share GPUs."
fi

# --- Cleanup any existing servers ---
echo "Cleaning up existing servers..."
pkill -f "vllm.entrypoints" 2>/dev/null || true
pkill -f "uvicorn.*agentic_sfm" 2>/dev/null || true
sleep 2

# --- Start vLLM server on GPU 0 ---
echo ""
echo ">>> Starting vLLM server on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model "Qwen/Qwen3-VL-8B-Instruct" \
    --port 8000 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 16384 \
    --trust-remote-code \
    --dtype bfloat16 \
    --api-key EMPTY \
    --disable-log-requests \
    --enforce-eager \
    > logs/vllm_server.log 2>&1 &
VLLM_PID=$!
echo "    vLLM PID: ${VLLM_PID}"

# --- Start tool server on GPU 2 ---
echo ">>> Starting tool server on GPU 2..."
CUDA_VISIBLE_DEVICES=2 PYTHONPATH="${PWD}/src" python -m uvicorn \
    agentic_sfm.tools.server:app \
    --host 0.0.0.0 \
    --port 8765 \
    --workers 1 \
    > logs/tool_server.log 2>&1 &
TOOL_PID=$!
echo "    Tool server PID: ${TOOL_PID}"

# --- Wait for both servers ---
echo ""
echo ">>> Waiting for servers to be ready..."

VLLM_READY=0
TOOL_READY=0
for i in $(seq 1 120); do
  if [ ${VLLM_READY} -eq 0 ] && curl -s http://localhost:8000/health > /dev/null 2>&1; then
    VLLM_READY=1
    echo "    vLLM ready (took ~$((i*2))s)"
  fi
  if [ ${TOOL_READY} -eq 0 ] && curl -s http://localhost:8765/health > /dev/null 2>&1; then
    TOOL_READY=1
    echo "    Tool server ready (took ~$((i*2))s)"
  fi
  if [ ${VLLM_READY} -eq 1 ] && [ ${TOOL_READY} -eq 1 ]; then
    break
  fi
  # Check for dead processes
  if [ ${VLLM_READY} -eq 0 ] && ! kill -0 ${VLLM_PID} 2>/dev/null; then
    echo "ERROR: vLLM process died. Check logs/vllm_server.log"
    tail -20 logs/vllm_server.log
    exit 1
  fi
  if [ ${TOOL_READY} -eq 0 ] && ! kill -0 ${TOOL_PID} 2>/dev/null; then
    echo "ERROR: Tool server process died. Check logs/tool_server.log"
    tail -20 logs/tool_server.log
    exit 1
  fi
  sleep 2
done

if [ ${VLLM_READY} -eq 0 ]; then
  echo "ERROR: vLLM failed to start within 240s"
  tail -30 logs/vllm_server.log
  kill ${VLLM_PID} ${TOOL_PID} 2>/dev/null || true
  exit 1
fi
if [ ${TOOL_READY} -eq 0 ]; then
  echo "ERROR: Tool server failed to start within 240s"
  tail -30 logs/tool_server.log
  kill ${VLLM_PID} ${TOOL_PID} 2>/dev/null || true
  exit 1
fi

echo ""
echo "============================================================"
echo "  Both servers ready. Starting GRPO training..."
echo "============================================================"

# --- Cleanup on exit ---
trap "echo 'Cleaning up servers...'; kill ${VLLM_PID} ${TOOL_PID} 2>/dev/null || true" EXIT

# --- Run GRPO training on GPU 1 ---
CUDA_VISIBLE_DEVICES=1 python scripts/run_grpo.py \
    --config "${CONFIG}" \
    --tool-server-url http://localhost:8765 \
    --vllm-url http://localhost:8000 \
    --output-dir $([ ${SMOKE} -eq 1 ] && echo "outputs/smoke" || echo "outputs/phase1")

echo ""
echo "============================================================"
echo "  GRPO training complete: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
