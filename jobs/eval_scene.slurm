#!/bin/bash
# =============================================================================
# Honest Phase 2 scene eval: best_eval vs oracle vs exhaustive COLMAP.
# 2x NVIDIA L40S — GPU 0 vLLM, GPU 1 tools. No training.
#
# MegaDepth test-1500 only has 0015 + 0022 (the r5 train scenes). A ScanNet
# holdout (8 scenes, unseen in r5) is appended so this is not a 2-scene echo.
#
#   sbatch jobs/eval_scene.slurm
# =============================================================================
#SBATCH --job-name=asfm-scene-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=170G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/kcwp264/logs/agentic-sfm/eval_scene_%j.out
#SBATCH --error=/scratch/kcwp264/logs/agentic-sfm/eval_scene_%j.err

set -euo pipefail

echo "=== Agentic-SFM Phase 2 scene eval (best_eval vs oracle vs direct) ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start: $(date)"

PROJECT_ROOT="/scratch/kcwp264/agentic-sfm"
CONFIG="${PROJECT_ROOT}/configs/phase2_scene.yaml"
LORA_CKPT="${PROJECT_ROOT}/outputs/phase2/run_8073061/checkpoints/best_eval"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/scene_eval/run_${SLURM_JOB_ID}"
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
echo "LoRA: ${LORA_CKPT}"

export SFM_OUTPUT_ROOT="${TMP_SHARED:-${TMPDIR:-/tmp}}/asfm_eval_sfm_${SLURM_JOB_ID}"
mkdir -p "${SFM_OUTPUT_ROOT}"
echo "SFM_OUTPUT_ROOT=${SFM_OUTPUT_ROOT}"

python - <<'PY'
from agentic_sfm.constants import assert_qwen35_runtime
assert_qwen35_runtime()
print("Policy runtime check passed")
PY

BASE_VLLM=$((39000 + SLURM_JOB_ID % 20000))
BASE_TOOL=$((19000 + SLURM_JOB_ID % 20000))
VLLM_PORT=$(find_free_port "${BASE_VLLM}")
TOOL_PORT=$(find_free_port "${BASE_TOOL}")
if [[ "${VLLM_PORT}" -eq "${TOOL_PORT}" ]]; then
    TOOL_PORT=$(find_free_port "$((TOOL_PORT + 1))")
fi
export VLLM_PORT TOOL_PORT
echo "Allocated ports -> vLLM: ${VLLM_PORT}, tool: ${TOOL_PORT}"

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
    if ! kill -0 ${VLLM_PID} 2>/dev/null; then
        echo "ERROR: vLLM process died before ready"
        exit 1
    fi
    sleep 2
done

echo "Starting tool server on GPU 1..."
CUDA_VISIBLE_DEVICES=1 PYTHONPATH="${PROJECT_ROOT}/src" python \
    "${PROJECT_ROOT}/tools_server/server.py" --port "${TOOL_PORT}" &
TOOL_PID=$!

for i in $(seq 1 60); do
    if curl -s "http://localhost:${TOOL_PORT}/health" > /dev/null 2>&1; then
        echo "Tool server ready on :${TOOL_PORT}"
        break
    fi
    if ! kill -0 ${TOOL_PID} 2>/dev/null; then
        echo "ERROR: Tool server process died before bind"
        exit 1
    fi
    if [ $i -eq 60 ]; then echo "ERROR: Tool server failed"; exit 1; fi
    sleep 2
done

echo "Running scene eval → ${OUTPUT_DIR}"
set +e
python scripts/eval_scene.py \
    --config "${CONFIG}" \
    --mode agent oracle direct \
    --lora-checkpoint "${LORA_CKPT}" \
    --max-images-per-scene 20 \
    --max-scannet-scenes 8 \
    --scannet-root /scratch/kcwp264/data/megadepth/scannet_test_1500 \
    --tool-server-url "http://localhost:${TOOL_PORT}" \
    --vllm-url "http://localhost:${VLLM_PORT}" \
    --output-dir "${OUTPUT_DIR}" \
    > >(tee "${LOG_DIR}/eval_scene_${SLURM_JOB_ID}.log") 2>&1
EVAL_EXIT=$?
set -e

cleanup
wait ${VLLM_PID} ${TOOL_PID} 2>/dev/null || true
trap - EXIT

if [[ ! -f "${OUTPUT_DIR}/scene_eval_results.json" ]]; then
    echo "ERROR: eval JSON was not written"
    exit 1
fi
echo "Results: ${OUTPUT_DIR}/scene_eval_results.json"
echo "Eval exit code: ${EVAL_EXIT}"
echo "End: $(date)"
seff "${SLURM_JOB_ID}" 2>/dev/null || true
exit ${EVAL_EXIT}
