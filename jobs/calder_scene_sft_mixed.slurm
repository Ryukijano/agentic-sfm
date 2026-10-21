#!/bin/bash
# =============================================================================
# SLURM: scene SFT warm start on Calder (1x H200) with the mixed-specialist,
# two-domain oracle data (data/scene_sft_mixed_train.jsonl). Produces the new
# GRPO init AND KL anchor: outputs/scene_sft_mixed/checkpoints/epoch_5.
#
# Submit from the repo root:  sbatch jobs/calder_scene_sft_mixed.slurm
# =============================================================================
#SBATCH --job-name=asfm-sft-mixed
#SBATCH --partition=calder-gpu-hopper
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=logs/agentic-sfm/sft_mixed_%j.out
#SBATCH --error=logs/agentic-sfm/sft_mixed_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?submit from the repo root}}"
[[ -f "${PROJECT_ROOT}/scripts/calder_agentic_sfm_env.sh" ]] || { echo "ERROR: ${PROJECT_ROOT} is not the agentic-sfm repo root"; exit 1; }
CONFIG="${PROJECT_ROOT}/configs/phase2_sft_scene_mixed.yaml"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/scene_sft_mixed"
LOG_DIR="${PROJECT_ROOT}/logs/agentic-sfm"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

echo "=== Scene SFT (mixed specialists) — Calder 1xH200 ==="
echo "Job ID: ${SLURM_JOB_ID}  Node: ${SLURM_NODELIST}  Start: $(date)"

source "${PROJECT_ROOT}/scripts/calder_agentic_sfm_env.sh"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
cd "${PROJECT_ROOT}"

python - <<'PY'
from agentic_sfm.constants import assert_qwen35_runtime
assert_qwen35_runtime()
print("Policy runtime check passed")
PY

[[ -s data/scene_sft_mixed_train.jsonl ]] || { echo "ERROR: data/scene_sft_mixed_train.jsonl missing"; exit 1; }
echo "SFT examples: $(wc -l < data/scene_sft_mixed_train.jsonl)"

CUDA_VISIBLE_DEVICES=0 python scripts/run_sft.py \
    --config "${CONFIG}" \
    --output-dir "${OUTPUT_DIR}" \
    2>&1 | tee "${LOG_DIR}/sft_mixed_${SLURM_JOB_ID}.log"

echo "End: $(date)"
echo "Adapter: ${OUTPUT_DIR}/checkpoints/epoch_5 (use as model.sft_adapter / kl_adapter in the GRPO config)"
