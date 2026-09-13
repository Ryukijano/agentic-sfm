#!/bin/bash
#SBATCH --job-name=agentic-sfm-pilot
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/kcwp264/logs/agentic-sfm/pilot_%j.out
#SBATCH --error=/scratch/kcwp264/logs/agentic-sfm/pilot_%j.err

set -euo pipefail

PROJECT_ROOT="/scratch/kcwp264/agentic-sfm"
CONFIG="${PROJECT_ROOT}/configs/phase1_grpo_pilot.yaml"
OUTPUT_DIR="/scratch/kcwp264/agentic-sfm/outputs/phase1_pilot"
LOG_DIR="/scratch/kcwp264/logs/agentic-sfm"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

source /scratch/kcwp264/.aire_scratch_env.sh
source "${PROJECT_ROOT}/scripts/agentic_sfm_env.sh"
module load cuda/12.4.1 2>/dev/null || true
export CUDA_HOME="${CUDA_HOME:-/opt/apps/pkg/compilers/cuda/12.4.1}"
export NCCL_P2P_DISABLE=1
export NCCL_NET=Socket
export NCCL_IB_DISABLE=1
export TRITON_CACHE_DIR="/scratch/kcwp264/.triton_cache"
export FLASHINFER_WORKSPACE_DIR="/scratch/kcwp264/.flashinfer"
export FLASHINFER_JIT_DIR="/scratch/kcwp264/.flashinfer/jit"
export HF_HOME="/scratch/kcwp264/.cache/huggingface"
export VLLM_CACHE_ROOT="/scratch/kcwp264/.cache/vllm"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export FLASHINFER_DISABLE_VERSION_CHECK=1
mkdir -p "${TRITON_CACHE_DIR}" "${FLASHINFER_WORKSPACE_DIR}" "${FLASHINFER_JIT_DIR}" "${VLLM_CACHE_ROOT}"

cd "${PROJECT_ROOT}"

# Shared GPU nodes: unique ports per job.
VLLM_PORT=$((39000 + SLURM_JOB_ID % 20000))
TOOL_SERVER_PORT=$((19000 + SLURM_JOB_ID % 20000))
export VLLM_PORT TOOL_SERVER_PORT
echo "vLLM port: ${VLLM_PORT}  tool port: ${TOOL_SERVER_PORT}"

if [[ ! -d outputs/sft/checkpoints/epoch_5 ]]; then
  echo "ERROR: SFT adapter missing at outputs/sft/checkpoints/epoch_5"
  echo "Run: sbatch jobs/phase1_oracle_sft.slurm && sbatch jobs/phase1_sft.slurm"
  exit 1
fi

python - <<'PY'
from agentic_sfm.constants import assert_qwen35_runtime
assert_qwen35_runtime()
import vllm
print("vllm", vllm.__version__)
PY

CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model "Qwen/Qwen3.5-2B" \
    --enable-lora --max-loras 1 --max-lora-rank 32 \
    --port "${VLLM_PORT}" --gpu-memory-utilization 0.90 --max-model-len 8192 \
    --limit-mm-per-prompt '{"image":6,"video":0}' \
    --trust-remote-code --dtype bfloat16 --api-key EMPTY --enforce-eager &
VLLM_PID=$!

for i in $(seq 1 240); do
    if ! kill -0 ${VLLM_PID} 2>/dev/null; then
        echo "ERROR: vLLM died before ready"
        exit 1
    fi
    curl -s "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1 && break
    [[ $i -eq 240 ]] && { echo "vLLM failed"; kill $VLLM_PID; exit 1; }
    sleep 2
done

CUDA_VISIBLE_DEVICES=2 PYTHONPATH="${PROJECT_ROOT}/src" python \
    tools_server/server.py --port "${TOOL_SERVER_PORT}" &
TOOL_PID=$!
for i in $(seq 1 60); do
    if ! kill -0 ${TOOL_PID} 2>/dev/null; then
        echo "ERROR: tool server died before bind"
        kill $VLLM_PID 2>/dev/null || true
        exit 1
    fi
    curl -s "http://localhost:${TOOL_SERVER_PORT}/health" >/dev/null 2>&1 && break
    [[ $i -eq 60 ]] && { echo "tool server failed"; kill $VLLM_PID $TOOL_PID; exit 1; }
    sleep 2
done

CUDA_VISIBLE_DEVICES=1 python scripts/run_grpo.py \
    --config "${CONFIG}" \
    --tool-server-url "http://localhost:${TOOL_SERVER_PORT}" \
    --vllm-url "http://localhost:${VLLM_PORT}" \
    --output-dir "${OUTPUT_DIR}" \
    2>&1 | tee "${LOG_DIR}/pilot_${SLURM_JOB_ID}.log"

TRAIN_EXIT=$?
kill $VLLM_PID $TOOL_PID 2>/dev/null || true
exit $TRAIN_EXIT
