#!/bin/bash
# Start the agentic-SFM tool server (matcher + crop + doppelganger) on GPU 2
set -euo pipefail

PORT="${1:-8765}"
GPU_ID="${2:-2}"

echo "Starting tool server on GPU ${GPU_ID}, port ${PORT}"

cd "$(dirname "$0")/.."

CUDA_VISIBLE_DEVICES=${GPU_ID} python -m uvicorn \
    agentic_sfm.tools.server:app \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --workers 1 &

TOOL_PID=$!
echo "Tool server PID: ${TOOL_PID}"

# Wait for server to be ready
echo "Waiting for tool server to be ready..."
for i in $(seq 1 60); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
        echo "Tool server ready at http://localhost:${PORT}"
        wait ${TOOL_PID}
        exit 0
    fi
    if ! kill -0 ${TOOL_PID} 2>/dev/null; then
        echo "ERROR: Tool server process died"
        exit 1
    fi
    sleep 2
done
echo "ERROR: Tool server failed to start within 120s"
kill ${TOOL_PID} 2>/dev/null || true
exit 1
