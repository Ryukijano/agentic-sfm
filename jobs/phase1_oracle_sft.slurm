#!/bin/bash
#SBATCH --job-name=agentic-sfm-oracle-sft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=85G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/kcwp264/logs/agentic-sfm/oracle_sft_%j.out
#SBATCH --error=/scratch/kcwp264/logs/agentic-sfm/oracle_sft_%j.err

set -euo pipefail

echo "=== Agentic-SFM oracle crop SFT data ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start: $(date)"

PROJECT_ROOT="/scratch/kcwp264/agentic-sfm"
LOG_DIR="/scratch/kcwp264/logs/agentic-sfm"
mkdir -p "${LOG_DIR}" "${PROJECT_ROOT}/data/oracle_crops"

source /scratch/kcwp264/.aire_scratch_env.sh
source "${PROJECT_ROOT}/scripts/agentic_sfm_env.sh"
module load cuda/12.4.1 2>/dev/null || true
export CUDA_HOME="${CUDA_HOME:-/opt/apps/pkg/compilers/cuda/12.4.1}"
export HF_HOME="/scratch/kcwp264/.cache/huggingface"
export TORCH_HOME="/scratch/kcwp264/.cache/torch"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export FLASHINFER_DISABLE_VERSION_CHECK=1

cd "${PROJECT_ROOT}"

# Shared GPU nodes: never assume 8765 is free.
TOOL_SERVER_PORT=$((19000 + SLURM_JOB_ID % 20000))
export TOOL_SERVER_PORT
echo "Tool server port: ${TOOL_SERVER_PORT}"

echo "Starting tool server on GPU 0..."
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="${PROJECT_ROOT}/src" python \
    "${PROJECT_ROOT}/tools_server/server.py" --port "${TOOL_SERVER_PORT}" &
TOOL_PID=$!

echo "Waiting for tool server (pid ${TOOL_PID})..."
for i in $(seq 1 60); do
    if ! kill -0 ${TOOL_PID} 2>/dev/null; then
        echo "ERROR: Tool server process died before bind"
        exit 1
    fi
    if curl -s "http://localhost:${TOOL_SERVER_PORT}/health" > /dev/null 2>&1; then
        echo "Tool server ready on :${TOOL_SERVER_PORT}"
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "ERROR: Tool server failed to start on :${TOOL_SERVER_PORT}"
        kill ${TOOL_PID} 2>/dev/null || true
        exit 1
    fi
    sleep 2
done

CUDA_VISIBLE_DEVICES=0 python scripts/build_oracle_sft.py \
    --pairs data/hard_pairs_train.json \
    --output data/sft_train.jsonl \
    --crop-dir data/oracle_crops \
    --tool-server-url "http://localhost:${TOOL_SERVER_PORT}" \
    --matcher loftr \
    2>&1 | tee "${LOG_DIR}/oracle_sft_${SLURM_JOB_ID}.log"

EXIT=$?
kill ${TOOL_PID} 2>/dev/null || true
wait ${TOOL_PID} 2>/dev/null || true
echo "End: $(date)"
exit ${EXIT}
