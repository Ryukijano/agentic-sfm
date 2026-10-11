#!/bin/bash
# =============================================================================
# SLURM: Phase 2 — Scene-level GRPO training for agentic SfM.
# 3x NVIDIA L40S (Ada, 48 GB) on a single AIRE GPU node.
#
# The agent orchestrates full scene reconstruction: retrieve pairs, match them,
# filter doppelgangers, run COLMAP, and inspect the result.
#
# GPU layout:
#   GPU 0: vLLM rollout server (Qwen3.5-2B inference, LoRA hot-reload)
#   GPU 1: Training (LoRA gradient updates)
#   GPU 2: Tool server (matchers + COLMAP pycolmap)
#
# Submit:
#   sbatch jobs/phase2_scene.slurm
# =============================================================================
#SBATCH --job-name=asfm-phase2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/kcwp264/logs/agentic-sfm/phase2_%j.out
#SBATCH --error=/scratch/kcwp264/logs/agentic-sfm/phase2_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

echo "=== Agentic-SFM Phase 2 Scene-Level GRPO ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)"
echo "Start: $(date)"

PROJECT_ROOT="/scratch/kcwp264/agentic-sfm"
CONFIG="${PROJECT_ROOT}/configs/phase2_scene.yaml"
# Unique per-job dir — do not overwrite outputs/phase2/checkpoints from 7975335.
OUTPUT_DIR="/scratch/kcwp264/agentic-sfm/outputs/phase2/run_${SLURM_JOB_ID}"
SCENE_SFT="${PROJECT_ROOT}/outputs/scene_sft_graph_retry2/checkpoints/epoch_5"
LOG_DIR="/scratch/kcwp264/logs/agentic-sfm"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

source /scratch/kcwp264/.aire_scratch_env.sh
source "${PROJECT_ROOT}/scripts/agentic_sfm_env.sh"
module load cuda/12.4.1 2>/dev/null || true
export CUDA_HOME="${CUDA_HOME:-/opt/apps/pkg/compilers/cuda/12.4.1}"
export NCCL_P2P_DISABLE=1
export NCCL_NET=Socket
export NCCL_IB_DISABLE=1
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1
export TRITON_CACHE_DIR="/scratch/kcwp264/.triton_cache"
export FLASHINFER_WORKSPACE_DIR="/scratch/kcwp264/.flashinfer"
export FLASHINFER_JIT_DIR="/scratch/kcwp264/.flashinfer/jit"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export XDG_CACHE_HOME="/scratch/kcwp264/.cache/xdg"
export HF_HOME="/scratch/kcwp264/.cache/huggingface"
export TORCH_HOME="/scratch/kcwp264/.cache/torch"
export VLLM_CACHE_ROOT="/scratch/kcwp264/.cache/vllm"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
mkdir -p "${TRITON_CACHE_DIR}" "${FLASHINFER_WORKSPACE_DIR}" "${FLASHINFER_JIT_DIR}" "${XDG_CACHE_HOME}" "${VLLM_CACHE_ROOT}"

cd "${PROJECT_ROOT}"

if [[ ! -f "${SCENE_SFT}/adapter_model.safetensors" ]]; then
    echo "ERROR: scene SFT adapter missing at ${SCENE_SFT}"
    exit 1
fi
echo "Scene SFT warmstart: ${SCENE_SFT}"

# COLMAP workdirs on job-local NVMe (deleted at job end). Never ./outputs/sfm_run_*.
export SFM_OUTPUT_ROOT="${TMP_SHARED:-${TMPDIR:-/tmp}}/asfm_sfm_${SLURM_JOB_ID}"
mkdir -p "${SFM_OUTPUT_ROOT}"
echo "SFM_OUTPUT_ROOT=${SFM_OUTPUT_ROOT}"

BASE_VLLM=$((39000 + SLURM_JOB_ID % 20000))
BASE_TOOL=$((19000 + SLURM_JOB_ID % 20000))
VLLM_PORT=$(find_free_port "${BASE_VLLM}")
TOOL_PORT=$(find_free_port "${BASE_TOOL}")
if [[ "${VLLM_PORT}" -eq "${TOOL_PORT}" ]]; then
    TOOL_PORT=$(find_free_port "$((TOOL_PORT + 1))")
fi
export VLLM_PORT TOOL_PORT
echo "Allocated ports -> vLLM: ${VLLM_PORT}, tool: ${TOOL_PORT}"

python - <<'PY'
from agentic_sfm.constants import assert_qwen35_runtime
assert_qwen35_runtime()
print("Policy runtime check passed")
PY

# W&B: confirm auth + scratch-local dirs before burning GPUs on a run whose
# curves we cannot see (wandb.init failure is only a warning in the trainer).
echo "W&B dir: ${WANDB_DIR:-<unset>}  cache: ${WANDB_CACHE_DIR:-<unset>}"
python - <<'PY'
import os
import wandb
try:
    user = wandb.api.viewer()["username"]
except Exception as e:
    raise SystemExit(f"ERROR: W&B auth failed ({type(e).__name__}: {e})")
print(f"W&B authenticated as {user}; project={os.environ.get('WANDB_PROJECT', 'agentic-sfm')}")
PY

cleanup() {
    echo "Cleaning up servers..."
    if [[ -n "${WATCH_PID:-}" ]]; then
        kill ${WATCH_PID} 2>/dev/null || true
    fi
    kill ${VLLM_PID:-} ${TOOL_PID:-} ${TRAIN_PID:-} 2>/dev/null || true
    if [[ -n "${TOOL_PID_FILE:-}" && -f "${TOOL_PID_FILE}" ]]; then
        kill "$(cat "${TOOL_PID_FILE}")" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Start vLLM server on GPU 0
echo "Starting vLLM server on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model "Qwen/Qwen3.5-2B" \
    --enable-lora \
    --max-loras 1 \
    --max-lora-rank 32 \
    --port "${VLLM_PORT}" \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --limit-mm-per-prompt '{"image":4,"video":0}' \
    --mm-processor-kwargs '{"max_pixels":401408,"min_pixels":3136}' \
    --mm-processor-cache-gb 0 \
    --trust-remote-code \
    --dtype bfloat16 \
    --api-key EMPTY \
    --enforce-eager &
VLLM_PID=$!

for i in $(seq 1 240); do
    if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "vLLM ready (took $((i*2))s)"
        break
    fi
    if [ $i -eq 240 ]; then echo "ERROR: vLLM failed"; exit 1; fi
    sleep 2
done

# Start tool server on GPU 2 (matchers + COLMAP). pycolmap/FAISS can SIGBUS
# inside FeatureMatcherWorker (job 7882190 @ 08:17) and kill uvicorn; a
# watchdog restarts it, and training abort if /health stays down.
echo "Starting tool server on GPU 2..."
TOOL_PID_FILE="${OUTPUT_DIR}/.tool_server_${TOOL_PORT}.pid"
mkdir -p "${OUTPUT_DIR}"
CUDA_VISIBLE_DEVICES=2 PYTHONPATH="${PROJECT_ROOT}/src" python \
    "${PROJECT_ROOT}/tools_server/server.py" --port "${TOOL_PORT}" &
TOOL_PID=$!
echo "${TOOL_PID}" > "${TOOL_PID_FILE}"

for i in $(seq 1 60); do
    if curl -s "http://localhost:${TOOL_PORT}/health" > /dev/null 2>&1; then
        echo "Tool server ready on :${TOOL_PORT} (pid ${TOOL_PID})"
        break
    fi
    if ! kill -0 ${TOOL_PID} 2>/dev/null; then
        echo "ERROR: Tool server process died before bind"
        exit 1
    fi
    if [ $i -eq 60 ]; then echo "ERROR: Tool server failed"; exit 1; fi
    sleep 2
done

# Run scene-level GRPO training on GPU 1 (background so the watchdog can
# kill it if the tool server stays unrestorable).
echo "Starting Phase 2 scene-level GRPO training..."
set +e
CUDA_VISIBLE_DEVICES=1 python scripts/run_scene_grpo.py \
    --config "${CONFIG}" \
    --tool-server-url "http://localhost:${TOOL_PORT}" \
    --vllm-url "http://localhost:${VLLM_PORT}" \
    --output-dir "${OUTPUT_DIR}" \
    > >(tee "${LOG_DIR}/phase2_train_${SLURM_JOB_ID}.log") 2>&1 &
TRAIN_PID=$!

chmod +x "${PROJECT_ROOT}/scripts/watch_tool_server.sh"
"${PROJECT_ROOT}/scripts/watch_tool_server.sh" \
    --port "${TOOL_PORT}" \
    --project-root "${PROJECT_ROOT}" \
    --protect-pid "${TRAIN_PID}" \
    --pid-file "${TOOL_PID_FILE}" \
    --gpu 2 \
    --interval 10 \
    --fail-threshold 3 \
    --max-restarts 5 \
    >> "${LOG_DIR}/phase2_${SLURM_JOB_ID}.out" 2>&1 &
WATCH_PID=$!
echo "Tool-server watchdog pid=${WATCH_PID} (interval 10s, fail after 30s, max 5 restarts)"

wait ${TRAIN_PID}
TRAIN_EXIT=$?
set -e
kill ${WATCH_PID} 2>/dev/null || true
wait ${WATCH_PID} 2>/dev/null || true
TOOL_PID="$(cat "${TOOL_PID_FILE}" 2>/dev/null || echo "${TOOL_PID}")"

cleanup
wait ${VLLM_PID} ${TOOL_PID} 2>/dev/null || true
trap - EXIT

mkdir -p "${OUTPUT_DIR}/checkpoints" "${OUTPUT_DIR}/logs"
cp -f "${LOG_DIR}/phase2_train_${SLURM_JOB_ID}.log" "${OUTPUT_DIR}/logs/" 2>/dev/null || true

echo "Training exit code: ${TRAIN_EXIT}"
echo "End: $(date)"
seff "${SLURM_JOB_ID}" 2>/dev/null || true
exit ${TRAIN_EXIT}
