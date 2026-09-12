#!/bin/bash
# =============================================================================
# SLURM: Phase 0 — Zero-shot crop feasibility evaluation.
# 3x NVIDIA L40S (Ada, 48 GB) on a single AIRE GPU node.
#
# GPU layout:
#   GPU 0: Qwen3-VL-2B-Instruct inference
#   GPU 1: LoFTR/MASt3R matcher (tool server)
#   GPU 2: reserved (Phase 1 will use for training)
#
# Submit:
#   sbatch jobs/phase0-eval.slurm
# =============================================================================
#SBATCH --job-name=asfm-p0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL

set -eo pipefail

# ---------------------------------------------------------------------------
# 1. Environment
# ---------------------------------------------------------------------------
source /scratch/kcwp264/.aire_scratch_env.sh
module load miniforge/24.7.1 || true
module load cuda/12.6.2 || true

CONDA_BASE=$(conda info --base 2>/dev/null || echo "/opt/apps/pkg/interpreters/miniforge/24.7.1/bin")
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate /scratch/kcwp264/.conda_envs/agentic-sfm
set -u
export CONDA_DEFAULT_ENV=agentic-sfm

# ---------------------------------------------------------------------------
# 2. Paths
# ---------------------------------------------------------------------------
REPO_ROOT="/scratch/kcwp264/agentic-sfm"
cd "${REPO_ROOT}"
mkdir -p logs outputs/phase0 data

export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}
export CUDA_VISIBLE_DEVICES=0,1,2
export HF_HOME=/scratch/kcwp264/.cache/huggingface
export WANDB_DIR=/scratch/kcwp264/wandb
export WANDB_CACHE_DIR=/scratch/kcwp264/.cache/wandb

echo "=== Phase 0: Zero-shot feasibility ==="
echo "Node:   $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-n/a}"
date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ---------------------------------------------------------------------------
# 3. Start tool server on GPU 1 (matcher inference)
# ---------------------------------------------------------------------------
echo "[INFO] Starting tool server on GPU 1..."
CUDA_VISIBLE_DEVICES=1 python tools_server/server.py &
TOOL_SERVER_PID=$!
sleep 5

# Verify tool server is healthy
for i in $(seq 1 10); do
  if curl -s http://localhost:8765/health > /dev/null 2>&1; then
    echo "[INFO] Tool server is healthy"
    break
  fi
  echo "[INFO] Waiting for tool server... ($i/10)"
  sleep 3
done

# ---------------------------------------------------------------------------
# 4. Run Phase 0 evaluation
# ---------------------------------------------------------------------------
echo "[INFO] Running Phase 0 zero-shot evaluation..."
python scripts/run_zeroshot.py \
  --config configs/phase0_zeroshot.yaml \
  --tool-server-url http://localhost:8765

# ---------------------------------------------------------------------------
# 5. Cleanup
# ---------------------------------------------------------------------------
kill ${TOOL_SERVER_PID} 2>/dev/null || true

echo ""
echo "=== Phase 0 finished ==="
date
echo "Results: outputs/phase0/"
