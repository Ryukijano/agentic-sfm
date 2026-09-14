#!/bin/bash
#SBATCH --job-name=asfm-scene-sft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=85G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/kcwp264/logs/agentic-sfm/scene_sft_%j.out
#SBATCH --error=/scratch/kcwp264/logs/agentic-sfm/scene_sft_%j.err

set -euo pipefail

echo "=== Agentic-SFM Phase 2 scene SFT warmup (Qwen3.5-2B) ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start: $(date)"

PROJECT_ROOT="/scratch/kcwp264/agentic-sfm"
CONFIG="${PROJECT_ROOT}/configs/phase2_sft_scene.yaml"
OUTPUT_DIR="/scratch/kcwp264/agentic-sfm/outputs/scene_sft"
LOG_DIR="/scratch/kcwp264/logs/agentic-sfm"
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

if [ ! -s data/scene_sft_train.jsonl ]; then
    echo "ERROR: data/scene_sft_train.jsonl is missing or empty."
    echo "Run: python scripts/build_scene_sft.py --num-episodes 200"
    exit 1
fi

echo "Scene SFT examples: $(wc -l < data/scene_sft_train.jsonl)"
CUDA_VISIBLE_DEVICES=0 python scripts/run_sft.py \
    --config "${CONFIG}" \
    --output-dir "${OUTPUT_DIR}" \
    2>&1 | tee "${LOG_DIR}/scene_sft_${SLURM_JOB_ID}.log"

echo "End: $(date)"
echo "Scene adapter: ${OUTPUT_DIR}/checkpoints/"
echo "Point configs/phase2_scene.yaml model.sft_adapter at the last epoch, then sbatch jobs/phase2_scene.slurm"
