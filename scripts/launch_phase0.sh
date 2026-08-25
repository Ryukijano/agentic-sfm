#!/usr/bin/env bash
# Launch script for starting tool server + running Phase 0 eval interactively.
# Usage: bash scripts/launch_phase0.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

echo "=== Agentic SfM Phase 0 Launcher ==="

# Start tool server in background
echo "[1/3] Starting tool server..."
CUDA_VISIBLE_DEVICES=1 python tools_server/server.py &
TOOL_PID=$!
sleep 3

# Verify
if curl -s http://localhost:8765/health > /dev/null 2>&1; then
    echo "  Tool server: OK"
else
    echo "  Tool server: FAILED"
    kill ${TOOL_PID} 2>/dev/null || true
    exit 1
fi

# Run evaluation
echo "[2/3] Running zero-shot evaluation..."
CUDA_VISIBLE_DEVICES=0 python scripts/run_zeroshot.py \
    --config configs/phase0_zeroshot.yaml \
    --tool-server-url http://localhost:8765

# Cleanup
echo "[3/3] Cleaning up..."
kill ${TOOL_PID} 2>/dev/null || true
echo "Done. Results in outputs/phase0/"
