#!/bin/bash
#SBATCH --job-name=asfm-sft-opsft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=85G
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/kcwp264/logs/agentic-sfm/sft_opsft_%j.out
#SBATCH --error=/scratch/kcwp264/logs/agentic-sfm/sft_opsft_%j.err

set -euo pipefail

echo "=== Agentic-SFM OPSFT-style constrained SFT (direction-constrained LoRA) ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start: $(date)"

PROJECT_ROOT="/scratch/kcwp264/agentic-sfm"
LOG_DIR="/scratch/kcwp264/logs/agentic-sfm"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/scene_opsft_base"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

source /scratch/kcwp264/.aire_scratch_env.sh
source "${PROJECT_ROOT}/scripts/agentic_sfm_env.sh"
module load cuda/12.4.1 2>/dev/null || true
export CUDA_HOME="${CUDA_HOME:-/opt/apps/pkg/compilers/cuda/12.4.1}"
export HF_HOME="/scratch/kcwp264/.cache/huggingface"
export TORCH_HOME="/scratch/kcwp264/.cache/torch"
export TRITON_CACHE_DIR="/scratch/kcwp264/.triton_cache"
export WANDB_DIR="/scratch/kcwp264/wandb"
export WANDB_CACHE_DIR="/scratch/kcwp264/.cache/wandb"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export FLASHINFER_DISABLE_VERSION_CHECK=1
mkdir -p "${TRITON_CACHE_DIR}" "${WANDB_DIR}" "${WANDB_CACHE_DIR}"

cd "${PROJECT_ROOT}"

python - <<'PY'
from agentic_sfm.constants import assert_qwen35_runtime
assert_qwen35_runtime()
print("Policy runtime check passed")
PY

# --- Step 1: extract the update direction artifact (mask + sign) if missing ---
DIRECTION_BASE="${DIRECTION_BASE:-outputs/scene_sft_graph_retry2/checkpoints/epoch_5}"
DIRECTION_TRAINED="${DIRECTION_TRAINED:-outputs/phase2_hybrid/latest/checkpoints/latest}"
DIRECTION_OUT="${DIRECTION_OUT:-outputs/opsft/direction_scene_grpo.pt}"
# exact_support is near-dense under AdamW LoRA (~100%); topk approximates
# the paper's sparse regime (global top-|delta| fraction).
DIRECTION_SELECTION="${DIRECTION_SELECTION:-topk}"
DIRECTION_TOPK_FRAC="${DIRECTION_TOPK_FRAC:-0.1}"

echo "DIRECTION_BASE:   ${DIRECTION_BASE}"
echo "DIRECTION_TRAINED: ${DIRECTION_TRAINED}"
echo "DIRECTION_OUT:    ${DIRECTION_OUT}"

if [ ! -s "${DIRECTION_OUT}" ]; then
    mkdir -p "$(dirname "${DIRECTION_OUT}")"
    echo "Extracting update direction artifact..."
    python scripts/extract_update_direction.py \
        --base "${DIRECTION_BASE}" \
        --trained "${DIRECTION_TRAINED}" \
        --output "${DIRECTION_OUT}" \
        --selection "${DIRECTION_SELECTION}" \
        --topk-frac "${DIRECTION_TOPK_FRAC}"
else
    echo "Direction artifact already exists; skipping extraction."
fi

# --- Step 2: constrained SFT ---
CUDA_VISIBLE_DEVICES=0 python scripts/run_sft_opsft.py \
    --config configs/phase2_sft_opsft.yaml \
    --output-dir "${OUTPUT_DIR}" \
    2>&1 | tee "${LOG_DIR}/sft_opsft_${SLURM_JOB_ID}.log"

echo "End: $(date)"
echo "Adapter: ${OUTPUT_DIR}/checkpoints/"
