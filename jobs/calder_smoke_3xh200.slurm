#!/bin/bash
# =============================================================================
# SLURM: Phase-2 SMOKE — 3x NVIDIA H200, ~1 epoch on 2 scenes.
#
# Exercises the real training path end-to-end before the 8xH200 job:
#   GPU 0 : vLLM server (TP=1, max-model-len 16384)
#   GPU 1 : tool server (LoFTR + MASt3R + VGGT-1B + COLMAP pycolmap)
#   GPU 2 : run_scene_grpo.py (trainable LoRA; frozen KL on CPU)
#
# Verifies: W&B auth+logging, fused GRPO/OPSFT kernels, reconstruction
# reward plumbing (mAA/ATE/F1/depth/ortho), LoRA->vLLM on-policy reload.
#
# Submit:  sbatch jobs/calder_smoke_3xh200.slurm
# =============================================================================
#SBATCH --job-name=asfm-smoke-3xh200
#SBATCH --partition=calder-gpu-hopper
#SBATCH --gres=gpu:nvidia_h200:3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=500G
#SBATCH --time=02:00:00
#SBATCH --output=logs/agentic-sfm/smoke_3xh200_%j.out
#SBATCH --error=logs/agentic-sfm/smoke_3xh200_%j.err

set -euo pipefail

echo "=== Agentic-SFM SMOKE — Calder 3xH200 ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)"
echo "Start: $(date)"

# Slurm runs a spooled copy of this script, so BASH_SOURCE points at
# /var/spool/slurm; resolve the repo from the submit dir (or an override).
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?submit from the repo root}}"
[[ -f "${PROJECT_ROOT}/scripts/calder_agentic_sfm_env.sh" ]] || { echo "ERROR: ${PROJECT_ROOT} is not the agentic-sfm repo root"; exit 1; }
CONFIG="${PROJECT_ROOT}/configs/phase2_calder_smoke.yaml"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/phase2_smoke/run_${SLURM_JOB_ID}"
SCENE_SFT="${PROJECT_ROOT}/outputs/scene_sft_graph_retry2/checkpoints/epoch_5"
LOG_DIR="${PROJECT_ROOT}/logs/agentic-sfm"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

source "${PROJECT_ROOT}/scripts/calder_agentic_sfm_env.sh"
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1  # required for /v1/load_lora_adapter (otherwise 404)

FLASH="${CALDER_FLASH:-/dev/shm/${USER}}"
export SFM_OUTPUT_ROOT="${FLASH}/asfm_sfm_smoke_${SLURM_JOB_ID}"
export LORA_STAGING_DIR="${FLASH}/asfm_lora_smoke_${SLURM_JOB_ID}"
mkdir -p "${SFM_OUTPUT_ROOT}" "${LORA_STAGING_DIR}"

VLLM_PORT=$(find_free_port $((39000 + SLURM_JOB_ID % 20000)))
TOOL_PORT=$(find_free_port $((19000 + SLURM_JOB_ID % 20000)))
export VLLM_PORT TOOL_PORT
echo "Ports -> vLLM:${VLLM_PORT} tool:${TOOL_PORT}"

cd "${PROJECT_ROOT}"

[[ -f "${SCENE_SFT}/adapter_model.safetensors" ]] || { echo "ERROR: missing ${SCENE_SFT}"; exit 1; }

python - <<'PY'
from agentic_sfm.constants import assert_qwen35_runtime
assert_qwen35_runtime()
print("Policy runtime check passed")
PY

# --- W&B connectivity check (fail fast if compute node can't reach it) ---
python - <<'PY'
import os, time
import wandb
try:
    t0 = time.time()
    user = wandb.api.viewer()["username"]
    print(f"W&B auth OK as {user} ({time.time()-t0:.1f}s)")
except Exception as e:
    print(f"W&B OFFLINE ({type(e).__name__}: {e}) — exporting WANDB_MODE=offline")
    with open(os.path.expandvars("${OUTPUT_DIR}/.wandb_offline"), "w") as f:
        f.write("offline\n")
PY
if [[ -f "${OUTPUT_DIR}/.wandb_offline" ]]; then
    export WANDB_MODE=offline
    echo "WANDB_MODE=offline (compute node has no WAN route)"
else
    echo "W&B live logging enabled"
fi

VLLM_PID=""; TOOL_PID=""; TRAIN_PID=""
cleanup() {
    echo "Cleaning up servers..."
    kill ${TRAIN_PID:-} ${VLLM_PID:-} ${TOOL_PID:-} 2>/dev/null || true
    if [[ -n "${TOOL_PID_FILE:-}" && -f "${TOOL_PID_FILE}" ]]; then
        kill "$(cat "${TOOL_PID_FILE}")" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# --- GPU 0: vLLM rollout server (TP=1 for smoke) ---
echo "Starting vLLM on GPU 0 (TP=1, 16k ctx)..."
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model "Qwen/Qwen3.5-2B" \
    --tensor-parallel-size 1 \
    --enable-lora \
    --max-loras 1 \
    --max-lora-rank 32 \
    --port "${VLLM_PORT}" \
    --gpu-memory-utilization 0.85 \
    --max-model-len 16384 \
    --limit-mm-per-prompt '{"image":4,"video":0}' \
    --mm-processor-kwargs '{"max_pixels":401408,"min_pixels":3136}' \
    --mm-processor-cache-gb 0 \
    --trust-remote-code \
    --dtype bfloat16 \
    --api-key EMPTY \
    > "${LOG_DIR}/smoke_vllm_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!

for i in $(seq 1 300); do
    if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "vLLM ready (took $((i*2))s)"; break
    fi
    if ! kill -0 ${VLLM_PID} 2>/dev/null; then
        echo "ERROR: vLLM died; tail of log:"
        tail -30 "${LOG_DIR}/smoke_vllm_${SLURM_JOB_ID}.log"
        exit 1
    fi
    if [ $i -eq 300 ]; then echo "ERROR: vLLM never became ready"; exit 1; fi
    sleep 2
done

# --- GPU 1: tool server ---
echo "Starting tool server on GPU 1..."
TOOL_PID_FILE="${OUTPUT_DIR}/.tool_server_${TOOL_PORT}.pid"
CUDA_VISIBLE_DEVICES=1 PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/../mast3r:${PROJECT_ROOT}/../mast3r/dust3r:${PROJECT_ROOT}/../mast3r/dust3r/croco" python \
    "${PROJECT_ROOT}/tools_server/server.py" --port "${TOOL_PORT}" \
    > "${LOG_DIR}/smoke_tools_${SLURM_JOB_ID}.log" 2>&1 &
TOOL_PID=$!
echo "${TOOL_PID}" > "${TOOL_PID_FILE}"

for i in $(seq 1 120); do
    if curl -s "http://localhost:${TOOL_PORT}/health" > /dev/null 2>&1; then
        echo "Tool server ready on :${TOOL_PORT} (pid ${TOOL_PID})"; break
    fi
    if ! kill -0 ${TOOL_PID} 2>/dev/null; then
        echo "ERROR: tool server died; tail of log:"
        tail -30 "${LOG_DIR}/smoke_tools_${SLURM_JOB_ID}.log"
        exit 1
    fi
    if [ $i -eq 120 ]; then echo "ERROR: tool server never ready"; tail -30 "${LOG_DIR}/smoke_tools_${SLURM_JOB_ID}.log"; exit 1; fi
    sleep 2
done

# --- GPU 2: scene GRPO training (1 epoch, 2 scenes, group 2) ---
echo "Starting smoke training on GPU 2 (KL anchor on CPU)..."
set +e
CUDA_VISIBLE_DEVICES=2 python scripts/run_scene_grpo.py \
    --config "${CONFIG}" \
    --tool-server-url "http://localhost:${TOOL_PORT}" \
    --vllm-url "http://localhost:${VLLM_PORT}" \
    --output-dir "${OUTPUT_DIR}" \
    2>&1 | tee "${LOG_DIR}/smoke_train_${SLURM_JOB_ID}.log"
TRAIN_EXIT=${PIPESTATUS[0]}
set -e
echo "Training exit code: ${TRAIN_EXIT}"

echo "=== SMOKE DONE ==="
echo "End: $(date)"
exit ${TRAIN_EXIT}
