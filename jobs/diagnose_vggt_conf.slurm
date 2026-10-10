#!/bin/bash
# =============================================================================
# SLURM: VGGT confidence diagnostic on one frozen ScanNet scene (~5 min).
# Chooses the confidence threshold for the VGGT correspondence path before
# a full pose-prior benchmark run.
#
# Submit: sbatch jobs/diagnose_vggt_conf.slurm
# =============================================================================
#SBATCH --job-name=asfm-vggt-conf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=85G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/kcwp264/logs/agentic-sfm/vggt_conf_%j.out
#SBATCH --error=/scratch/kcwp264/logs/agentic-sfm/vggt_conf_%j.err

set -euo pipefail

PROJECT_ROOT="/scratch/kcwp264/agentic-sfm"
OUT_DIR="${PROJECT_ROOT}/outputs/diagnose_vggt_conf"
SCENE="/scratch/kcwp264/data/megadepth/scannet_test_1500/scannet_test_1500/scene0715_00"
mkdir -p "${OUT_DIR}" /scratch/kcwp264/logs/agentic-sfm

source /scratch/kcwp264/.aire_scratch_env.sh
module load miniforge/24.7.1 || true
module load cuda/12.4.1 || true
CONDA_BASE=$(conda info --base 2>/dev/null || echo "/opt/apps/pkg/interpreters/miniforge/24.7.1")
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate /scratch/kcwp264/.conda_envs/agentic-sfm
set -u

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:/scratch/kcwp264/src/mast3r:/scratch/kcwp264/src/mast3r/dust3r:/scratch/kcwp264/src/mast3r/dust3r/croco:${PYTHONPATH:-}"
export HF_HOME="${SCRATCH}/.cache/huggingface"
export TORCH_HOME="${SCRATCH}/.cache/torch"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python scripts/diagnose_vggt_conf.py --scene "${SCENE}" --pairs 6 \
    --out "${OUT_DIR}/conf_${SLURM_JOB_ID}.json"

seff "${SLURM_JOB_ID}" || true
