#!/bin/bash
# =============================================================================
# SLURM: Phase 1 — S-GRPO smoke test (2-GPU variant for faster scheduling).
# 2x NVIDIA L40S (Ada, 48 GB) on a single AIRE GPU node.
#
# GPU layout:
#   GPU 0: vLLM rollout server + tool server (both inference, ~30GB total)
#   GPU 1: Training (LoRA gradient updates)
#
# Submit:
#   sbatch jobs/phase1_sgrpo_smoke_2gpu.slurm
# =============================================================================
#SBATCH --job-name=asfm-sgrpo-smoke2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/kcwp264/logs/agentic-sfm/sgrpo_smoke2_%j.out
#SBATCH --error=/scratch/kcwp264/logs/agentic-sfm/sgrpo_smoke2_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

echo "=== Agentic-SFM Phase 1 S-GRPO Smoke Test (2-GPU) ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)"
echo "Start: $(date)"

PROJECT_ROOT="/scratch/kcwp264/agentic-sfm"
CONFIG="${PROJECT_ROOT}/configs/phase1_sgrpo_smoke.yaml"
OUTPUT_DIR="/scratch/kcwp264/agentic-sfm/outputs/sgrpo_smoke"
LOG_DIR="/scratch/kcwp264/logs/agentic-sfm"
SUBSET_FILE="${PROJECT_ROOT}/data/hard_pairs_train_smoke50.json"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}" "${OUTPUT_DIR}/checkpoints" "${OUTPUT_DIR}/logs"

source /scratch/kcwp264/.aire_scratch_env.sh
source "${PROJECT_ROOT}/scripts/agentic_sfm_env.sh"
module load cuda/12.4.1 || true
export CUDA_HOME="${CUDA_HOME:-/opt/apps/pkg/compilers/cuda/12.4.1}"
export NCCL_P2P_DISABLE=1
export NCCL_NET=Socket
export NCCL_IB_DISABLE=1
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

VLLM_PORT=$((39000 + SLURM_JOB_ID % 20000))
TOOL_PORT=$((19000 + SLURM_JOB_ID % 20000))
export VLLM_PORT TOOL_PORT
echo "vLLM port: ${VLLM_PORT}  tool port: ${TOOL_PORT}"

# Build smoke subset (first 50 pairs from shuffled train set)
if [[ ! -f "${SUBSET_FILE}" ]]; then
    python3 -c "
import json
src = 'data/hard_pairs_train.json'
dst = 'data/hard_pairs_train_smoke50.json'
with open(src) as f:
    pairs = json.load(f)
subset = pairs[:50]
with open(dst, 'w') as f:
    json.dump(subset, f)
from collections import Counter
print(f'smoke50: {Counter(p[\"difficulty\"] for p in subset)}')
"
fi

# Runtime check
python - <<'PY'
from agentic_sfm.constants import assert_qwen35_runtime
assert_qwen35_runtime()
print("Policy runtime check passed")
PY

cleanup() {
    echo "Cleaning up servers..."
    kill ${VLLM_PID:-} ${TOOL_PID:-} 2>/dev/null || true
}
trap cleanup EXIT

# Start tool server on GPU 0 (shared with vLLM — both inference-only)
echo "Starting tool server on GPU 0..."
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="${PROJECT_ROOT}/src" python \
    "${PROJECT_ROOT}/tools_server/server.py" --port "${TOOL_PORT}" &
TOOL_PID=$!

for i in $(seq 1 60); do
    if curl -s "http://localhost:${TOOL_PORT}/health" > /dev/null 2>&1; then
        echo "Tool server ready on :${TOOL_PORT}"
        break
    fi
    if [ $i -eq 60 ]; then echo "ERROR: Tool server failed"; exit 1; fi
    sleep 2
done

# Start vLLM on GPU 0 (shared with tool server)
echo "Starting vLLM server on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model "Qwen/Qwen3.5-2B" \
    --enable-lora \
    --max-loras 1 \
    --max-lora-rank 32 \
    --port "${VLLM_PORT}" \
    --gpu-memory-utilization 0.50 \
    --max-model-len 8192 \
    --limit-mm-per-prompt '{"image":6,"video":0}' \
    --mm-processor-kwargs '{"max_pixels":1003520,"min_pixels":3136}' \
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

# Run S-GRPO smoke training on GPU 1
echo "Starting S-GRPO smoke training..."
set +e
CUDA_VISIBLE_DEVICES=1 python scripts/run_grpo.py \
    --config "${CONFIG}" \
    --tool-server-url "http://localhost:${TOOL_PORT}" \
    --vllm-url "http://localhost:${VLLM_PORT}" \
    --output-dir "${OUTPUT_DIR}" \
    2>&1 | tee "${LOG_DIR}/sgrpo_smoke2_train_${SLURM_JOB_ID}.log"
TRAIN_EXIT=${PIPESTATUS[0]}
set -e

cleanup
wait ${VLLM_PID} ${TOOL_PID} 2>/dev/null || true
trap - EXIT

mkdir -p "${OUTPUT_DIR}/checkpoints" "${OUTPUT_DIR}/logs"
cp -f "${LOG_DIR}/sgrpo_smoke2_train_${SLURM_JOB_ID}.log" "${OUTPUT_DIR}/logs/" 2>/dev/null || true
cp -f "/scratch/kcwp264/logs/agentic-sfm/sgrpo_smoke2_${SLURM_JOB_ID}.out" \
      "/scratch/kcwp264/logs/agentic-sfm/sgrpo_smoke2_${SLURM_JOB_ID}.err" \
      "${OUTPUT_DIR}/logs/" 2>/dev/null || true

echo "=== Smoke test artifacts ==="
find "${OUTPUT_DIR}" -maxdepth 3 -type f | sort || true

echo "Training exit code: ${TRAIN_EXIT}"
echo "End: $(date)"
seff "${SLURM_JOB_ID}" 2>/dev/null || true
exit ${TRAIN_EXIT}
