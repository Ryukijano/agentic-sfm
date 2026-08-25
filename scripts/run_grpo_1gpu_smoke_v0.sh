#!/usr/bin/env bash
set -euo pipefail

AGENTIC_ENV="/scratch/kcwp264/.conda_envs/agentic-sfm"
PY="${AGENTIC_ENV}/bin/python"

module load cuda/12.6.2 2>/dev/null || true
export CUDA_HOME="${CUDA_HOME:-/opt/apps/pkg/compilers/cuda/12.6.2}"
export VLLM_USE_V1=0

cd /scratch/kcwp264/agentic-sfm
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export HF_HOME="/scratch/kcwp264/.cache/huggingface"
export TORCH_HOME="/scratch/kcwp264/.cache/torch"

mkdir -p logs outputs/smoke

pkill -f "vllm.entrypoints" 2>/dev/null || true
pkill -f "uvicorn.*agentic_sfm" 2>/dev/null || true
sleep 2

echo ">>> Starting vLLM (V0 engine) on GPU 0..."
CUDA_VISIBLE_DEVICES=0 ${PY} -m vllm.entrypoints.openai.api_server \
  --model "Qwen/Qwen3-VL-8B-Instruct" \
  --port 8000 \
  --gpu-memory-utilization 0.45 \
  --max-model-len 4096 \
  --trust-remote-code \
  --dtype bfloat16 \
  --api-key EMPTY \
  --disable-log-requests \
  --enforce-eager \
  > logs/vllm_server.log 2>&1 &
VLLM_PID=$!

echo ">>> Starting tool server on GPU 0..."
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="${PWD}/src" ${PY} -m uvicorn \
  agentic_sfm.tools.server:app \
  --host 0.0.0.0 \
  --port 8765 \
  --workers 1 \
  > logs/tool_server.log 2>&1 &
TOOL_PID=$!

VLLM_READY=0; TOOL_READY=0
for i in $(seq 1 120); do
  [ ${VLLM_READY} -eq 0 ] && curl -s http://localhost:8000/health >/dev/null 2>&1 && { VLLM_READY=1; echo "vLLM ready"; }
  [ ${TOOL_READY} -eq 0 ] && curl -s http://localhost:8765/health >/dev/null 2>&1 && { TOOL_READY=1; echo "Tool server ready"; }
  [ ${VLLM_READY} -eq 1 ] && [ ${TOOL_READY} -eq 1 ] && break
  ! kill -0 ${VLLM_PID} 2>/dev/null && { echo "vLLM died"; tail -30 logs/vllm_server.log; exit 1; }
  ! kill -0 ${TOOL_PID} 2>/dev/null && { echo "Tool server died"; tail -20 logs/tool_server.log; exit 1; }
  sleep 2
done

if [ ${VLLM_READY} -eq 0 ] || [ ${TOOL_READY} -eq 0 ]; then
  echo "ERROR: servers failed to start"; exit 1
fi

echo ">>> Running GRPO smoke test..."
CUDA_VISIBLE_DEVICES=0 ${PY} scripts/run_grpo.py \
  --config configs/phase1_grpo_smoke.yaml \
  --tool-server-url http://localhost:8765 \
  --vllm-url http://localhost:8000 \
  --output-dir outputs/smoke

echo "=== SMOKE TEST COMPLETE ==="
kill ${VLLM_PID} ${TOOL_PID} 2>/dev/null || true
