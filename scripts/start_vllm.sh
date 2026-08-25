#!/bin/bash
# Start vLLM server for Qwen3-VL-8B rollout sampling on GPU 0
set -euo pipefail

MODEL_NAME="${1:-Qwen/Qwen3-VL-8B-Instruct}"
PORT="${2:-8000}"
GPU_ID="${3:-0}"

echo "Starting vLLM server for ${MODEL_NAME} on GPU ${GPU_ID}, port ${PORT}"

CUDA_VISIBLE_DEVICES=${GPU_ID} python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_NAME}" \
    --port "${PORT}" \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096 \
    --trust-remote-code \
    --dtype bfloat16 \
    --api-key EMPTY \
    --disable-log-requests &

VLLM_PID=$!
echo "vLLM PID: ${VLLM_PID}"

# Wait for server to be ready
echo "Waiting for vLLM to be ready..."
for i in $(seq 1 120); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
        echo "vLLM server ready at http://localhost:${PORT}"
        wait ${VLLM_PID}
        exit 0
    fi
    # Check if process died
    if ! kill -0 ${VLLM_PID} 2>/dev/null; then
        echo "ERROR: vLLM process died"
        exit 1
    fi
    sleep 2
done
echo "ERROR: vLLM server failed to start within 240s"
kill ${VLLM_PID} 2>/dev/null || true
exit 1
