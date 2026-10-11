#!/bin/bash
# =============================================================================
# Honest Phase 1b pair eval: phase1_hard/best on val pairs (hard headline).
# 2x NVIDIA L40S — GPU 0 vLLM, GPU 1 tool server. No training GPU.
#
#   sbatch jobs/eval_pair.slurm
# =============================================================================
#SBATCH --job-name=asfm-pair-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=170G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/kcwp264/logs/agentic-sfm/eval_pair_%j.out
#SBATCH --error=/scratch/kcwp264/logs/agentic-sfm/eval_pair_%j.err

set -euo pipefail

echo "=== Agentic-SFM Phase 1b pair eval ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)"
echo "Start: $(date)"

PROJECT_ROOT="/scratch/kcwp264/agentic-sfm"
CONFIG="${PROJECT_ROOT}/configs/phase1_grpo_hard.yaml"
LORA_CKPT="${PROJECT_ROOT}/outputs/phase1_hard/checkpoints/best"
PAIRS="${PROJECT_ROOT}/data/hard_pairs_val.json"
BASELINE_JSON="${PROJECT_ROOT}/outputs/baselines/baseline_results_latest.json"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/eval_pair"
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

if [[ ! -f "${LORA_CKPT}/adapter_model.safetensors" ]]; then
    echo "ERROR: LoRA adapter missing at ${LORA_CKPT}"
    exit 1
fi
if [[ ! -f "${PAIRS}" ]]; then
    echo "ERROR: pair file missing at ${PAIRS}"
    exit 1
fi

python - <<'PY'
from agentic_sfm.constants import assert_qwen35_runtime
assert_qwen35_runtime()
print("Policy runtime check passed")
PY

BASE_VLLM=$((39000 + SLURM_JOB_ID % 20000))
BASE_TOOL=$((19000 + SLURM_JOB_ID % 20000))
VLLM_PORT=$(find_free_port "${BASE_VLLM}")
TOOL_SERVER_PORT=$(find_free_port "${BASE_TOOL}")
if [[ "${VLLM_PORT}" -eq "${TOOL_SERVER_PORT}" ]]; then
    TOOL_SERVER_PORT=$(find_free_port "$((TOOL_SERVER_PORT + 1))")
fi
export VLLM_PORT TOOL_SERVER_PORT
echo "Allocated ports -> vLLM: ${VLLM_PORT}, tool: ${TOOL_SERVER_PORT}"

cleanup() {
    echo "Cleaning up servers..."
    kill ${VLLM_PID:-} ${TOOL_PID:-} 2>/dev/null || true
}
trap cleanup EXIT

echo "Starting vLLM server on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model "Qwen/Qwen3.5-2B" \
    --enable-lora \
    --max-loras 1 \
    --max-lora-rank 32 \
    --port "${VLLM_PORT}" \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --limit-mm-per-prompt '{"image":6,"video":0}' \
    --mm-processor-kwargs '{"max_pixels":1003520,"min_pixels":3136}' \
    --mm-processor-cache-gb 0 \
    --trust-remote-code \
    --dtype bfloat16 \
    --api-key EMPTY \
    --enforce-eager &
VLLM_PID=$!

echo "Waiting for vLLM..."
for i in $(seq 1 240); do
    if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "vLLM ready (took $((i*2))s)"
        break
    fi
    if [ $i -eq 240 ]; then
        echo "ERROR: vLLM failed to start after 480s"
        exit 1
    fi
    if ! kill -0 ${VLLM_PID} 2>/dev/null; then
        echo "ERROR: vLLM process died before ready"
        exit 1
    fi
    sleep 2
done

echo "Starting tool server on GPU 1..."
CUDA_VISIBLE_DEVICES=1 PYTHONPATH="${PROJECT_ROOT}/src" python \
    "${PROJECT_ROOT}/tools_server/server.py" --port "${TOOL_SERVER_PORT}" &
TOOL_PID=$!

echo "Waiting for tool server..."
for i in $(seq 1 60); do
    if curl -s "http://localhost:${TOOL_SERVER_PORT}/health" > /dev/null 2>&1; then
        echo "Tool server ready on :${TOOL_SERVER_PORT}"
        break
    fi
    if ! kill -0 ${TOOL_PID} 2>/dev/null; then
        echo "ERROR: Tool server process died before bind"
        exit 1
    fi
    if [ $i -eq 60 ]; then
        echo "ERROR: Tool server failed to start"
        exit 1
    fi
    sleep 2
done

OUT_JSON="${OUTPUT_DIR}/phase1_hard_best_${SLURM_JOB_ID}.json"
echo "Running pair eval → ${OUT_JSON}"
set +e
python scripts/run_pair_eval.py \
    --config "${CONFIG}" \
    --lora-checkpoint "${LORA_CKPT}" \
    --pairs "${PAIRS}" \
    --tool-server-url "http://localhost:${TOOL_SERVER_PORT}" \
    --vllm-url "http://localhost:${VLLM_PORT}" \
    --output "${OUT_JSON}" \
    --baseline-json "${BASELINE_JSON}" \
    --difficulties hard \
    > >(tee "${LOG_DIR}/eval_pair_${SLURM_JOB_ID}.log") 2>&1
EVAL_EXIT=$?
set -e

cleanup
wait ${VLLM_PID} ${TOOL_PID} 2>/dev/null || true
trap - EXIT

if [[ ! -f "${OUT_JSON}" ]]; then
    echo "ERROR: eval JSON was not written"
    exit 1
fi
echo "Results: ${OUT_JSON}"
echo "Eval exit code: ${EVAL_EXIT}"
echo "End: $(date)"
seff "${SLURM_JOB_ID}" 2>/dev/null || true
exit ${EVAL_EXIT}
