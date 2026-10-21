#!/bin/bash
# =============================================================================
# SLURM: Phase 2 Hybrid Scene GRPO on Isambard-AI Phase 2 (1 node, 4x GH200).
#
# Architecture: aarch64 NVIDIA Grace Hopper Superchip (4x GH200 96GB HBM3 +
# 288 Grace CPU cores per node). Uses the documented NGC PyTorch Apptainer
# image with `apptainer exec --nv` (never pip-install x86 wheels).
#
# GPU layout:
#   GPU 0   : vLLM rollout server (Qwen/Qwen3.5-2B, max-model-len 16384)
#   GPU 1   : Tool server (LoFTR + MASt3R + VGGT-1B + pycolmap on 288 Grace CPUs)
#   GPU 2,3 : run_scene_grpo.py (cuda:0 = GPU 2 trainable LoRA, cuda:1 = GPU 3 frozen KL)
# =============================================================================
#SBATCH --job-name=asfm-isambard-4xgh200
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --time=00:30:00
#SBATCH --output=logs/agentic-sfm/isambard_4xgh200_%j.out
#SBATCH --error=logs/agentic-sfm/isambard_4xgh200_%j.err

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${PROJECT_ROOT}/configs/phase2_isambard_4xgh200.yaml"
GIT_SHA="$(git -C "${PROJECT_ROOT}" rev-parse HEAD 2>/dev/null || echo 'unknown')"
SEED="${SEED:-42}"
SIF_IMAGE="${SIF_IMAGE:-/projects/u6xn/sif-images/pytorch_26.09-py3.sif}"
MAX_EPOCHS_THIS_RUN="${MAX_EPOCHS_THIS_RUN:-2}"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/phase2_isambard/run_${SLURM_JOB_ID:-local}"
LOG_DIR="${PROJECT_ROOT}/logs/agentic-sfm"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

echo "=== Agentic-SfM Phase 2 Scene GRPO — Isambard-AI 4xGH200 (aarch64) ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURM_NODELIST:-localhost}"
echo "Arch: $(uname -m)"
echo "Config: ${CONFIG}"
echo "Git SHA: ${GIT_SHA}"
echo "Seed: ${SEED}"
echo "Apptainer image: ${SIF_IMAGE}"
echo "Start: $(date -Iseconds)"

export SFM_OUTPUT_ROOT="/dev/shm/${USER}/asfm_sfm_${SLURM_JOB_ID:-0}"
export LORA_STAGING_DIR="/dev/shm/${USER}/asfm_lora_${SLURM_JOB_ID:-0}"
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1  # required for /v1/load_lora_adapter (otherwise 404)
mkdir -p "${SFM_OUTPUT_ROOT}" "${LORA_STAGING_DIR}"

VLLM_PORT=$((39000 + ${SLURM_JOB_ID:-0} % 10000))
TOOL_PORT=$((19000 + ${SLURM_JOB_ID:-0} % 10000))
export VLLM_PORT TOOL_PORT

VLLM_PID=""; TOOL_PID=""; TRAIN_PID=""
cleanup() {
    kill ${TRAIN_PID:-} ${VLLM_PID:-} ${TOOL_PID:-} 2>/dev/null || true
}
trap cleanup EXIT

cd "${PROJECT_ROOT}"

# GPU 0: vLLM rollout server inside NGC aarch64 container
CUDA_VISIBLE_DEVICES=0 apptainer exec --nv "${SIF_IMAGE}" \
    python -m vllm.entrypoints.openai.api_server \
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
    > "${LOG_DIR}/isambard_vllm_${SLURM_JOB_ID:-0}.log" 2>&1 &
VLLM_PID=$!

# GPU 1: Tool server (vision matchers + Grace CPU COLMAP workers)
CUDA_VISIBLE_DEVICES=1 apptainer exec --nv "${SIF_IMAGE}" \
    env PYTHONPATH="${PROJECT_ROOT}/src" \
    python "${PROJECT_ROOT}/tools_server/server.py" --port "${TOOL_PORT}" \
    > "${LOG_DIR}/isambard_tool_${SLURM_JOB_ID:-0}.log" 2>&1 &
TOOL_PID=$!

for i in $(seq 1 120); do
    if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1 && \
       curl -s "http://localhost:${TOOL_PORT}/health" > /dev/null 2>&1; then
        echo "vLLM (:${VLLM_PORT}) and tool server (:${TOOL_PORT}) ready"
        break
    fi
    sleep 2
done

# GPUs 2,3: Scene-level GRPO trainer (cuda:0 = GPU 2 trainable, cuda:1 = GPU 3 frozen KL)
RESUME_ARGS=()
if [ -d "${OUTPUT_DIR}/checkpoints/latest" ]; then
    RESUME_ARGS=(--resume "${OUTPUT_DIR}/checkpoints/latest")
fi
CUDA_VISIBLE_DEVICES=2,3 apptainer exec --nv "${SIF_IMAGE}" \
    env PYTHONPATH="${PROJECT_ROOT}/src" \
    python scripts/run_scene_grpo.py \
    --config "${CONFIG}" \
    --tool-server-url "http://localhost:${TOOL_PORT}" \
    --vllm-url "http://localhost:${VLLM_PORT}" \
    --output-dir "${OUTPUT_DIR}" \
    --max-epochs-this-run "${MAX_EPOCHS_THIS_RUN}" \
    "${RESUME_ARGS[@]}" \
    > "${LOG_DIR}/isambard_train_${SLURM_JOB_ID:-0}.log" 2>&1 &
TRAIN_PID=$!

wait ${TRAIN_PID}
TRAIN_EXIT=$?
echo "End: $(date -Iseconds) | Exit: ${TRAIN_EXIT} | Checkpoints: ${OUTPUT_DIR}/checkpoints"
exit ${TRAIN_EXIT}
