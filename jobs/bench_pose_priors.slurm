#!/bin/bash
# =============================================================================
# SLURM: Pose-prior benchmark on the frozen held-out ScanNet manifest.
# 1x NVIDIA L40S (Ada, 48 GB) on a single AIRE GPU node.
#
# Benchmarks pairwise relative-pose accuracy of the tool-server matcher
# backends (loftr / vggt / mast3r) over identical fixed pair lists per scene
# from data/frozen_eval_manifest.json (frozen by
# scripts/freeze_scene_manifest.py — do NOT regenerate mid-benchmark).
#
# The benchmark imports tools_server/server.py in-process, so the matchers
# run on this job's GPU directly — no separate tool server is started.
#
# Submit:
#   sbatch jobs/bench_pose_priors.slurm
#   sbatch --export=MAX_IMAGES=20,NUM_RANDOM_PAIRS=20 jobs/bench_pose_priors.slurm
# =============================================================================
#SBATCH --job-name=asfm-bench-priors
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=85G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/kcwp264/logs/agentic-sfm/bench_pose_priors_%j.out
#SBATCH --error=/scratch/kcwp264/logs/agentic-sfm/bench_pose_priors_%j.err

set -euo pipefail

echo "=== Agentic-SFM pose-prior benchmark ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start: $(date)"

PROJECT_ROOT="/scratch/kcwp264/agentic-sfm"
LOG_DIR="/scratch/kcwp264/logs/agentic-sfm"
OUT_DIR="${PROJECT_ROOT}/outputs/bench_pose_priors"
MANIFEST="${PROJECT_ROOT}/data/frozen_eval_manifest.json"
mkdir -p "${LOG_DIR}" "${OUT_DIR}"

# ---------------------------------------------------------------------------
# 1. Environment
# ---------------------------------------------------------------------------
source /scratch/kcwp264/.aire_scratch_env.sh
module load miniforge/24.7.1 || true
module load cuda/12.4.1 || true

CONDA_BASE=$(conda info --base 2>/dev/null || echo "/opt/apps/pkg/interpreters/miniforge/24.7.1")
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate /scratch/kcwp264/.conda_envs/agentic-sfm
set -u
export CONDA_DEFAULT_ENV=agentic-sfm

cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export HF_HOME="${SCRATCH}/.cache/huggingface"
export TORCH_HOME="${SCRATCH}/.cache/torch"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export FLASHINFER_DISABLE_VERSION_CHECK=1

echo "Python: $(which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ---------------------------------------------------------------------------
# 2. Verify the frozen manifest before running
# ---------------------------------------------------------------------------
python scripts/freeze_scene_manifest.py --verify --out "${MANIFEST}"

# ---------------------------------------------------------------------------
# 3. Run the benchmark (matchers run in-process on this job's GPU)
# ---------------------------------------------------------------------------
python scripts/bench_pose_priors.py \
    --manifest "${MANIFEST}" \
    --matchers loftr vggt mast3r \
    --max-images "${MAX_IMAGES:-20}" \
    --num-random-pairs "${NUM_RANDOM_PAIRS:-20}" \
    --seed "${SEED:-0}" \
    --output-dir "${OUT_DIR}"

echo "Results: ${OUT_DIR}/run_${SLURM_JOB_ID}/"
echo "End: $(date)"

seff "${SLURM_JOB_ID}" || true
