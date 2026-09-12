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
module load cuda/12.6.2 2>/dev/null || true
export CUDA_HOME="${CUDA_HOME:-/opt/apps/pkg/compilers/cuda/12.6.2}"
export HF_HOME="/scratch/kcwp264/.cache/huggingface"
export TORCH_HOME="/scratch/kcwp264/.cache/torch"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

cd "${PROJECT_ROOT}"

echo "Starting tool server on GPU 0..."
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="${PROJECT_ROOT}/src" python \
    "${PROJECT_ROOT}/tools_server/server.py" &
TOOL_PID=$!

echo "Waiting for tool server..."
for i in $(seq 1 30); do
    if curl -s http://localhost:8765/health > /dev/null 2>&1; then
        echo "Tool server ready"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "ERROR: Tool server failed to start"
        kill ${TOOL_PID} 2>/dev/null || true
        exit 1
    fi
    sleep 2
done

CUDA_VISIBLE_DEVICES=0 python scripts/build_oracle_sft.py \
    --pairs data/hard_pairs_train.json \
    --output data/sft_train.jsonl \
    --crop-dir data/oracle_crops \
    --tool-server-url http://localhost:8765 \
    --matcher loftr \
    2>&1 | tee "${LOG_DIR}/oracle_sft_${SLURM_JOB_ID}.log"

EXIT=$?
kill ${TOOL_PID} 2>/dev/null || true
wait ${TOOL_PID} 2>/dev/null || true
echo "End: $(date)"
exit ${EXIT}
