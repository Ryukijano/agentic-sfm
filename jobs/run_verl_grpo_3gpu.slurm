#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --job-name=verl-grpo-3gpu
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -e

source /scratch/kcwp264/.aire_scratch_env.sh

PROJECT_DIR=/scratch/kcwp264/agentic-sfm
CONDA_ENV=/scratch/kcwp264/.conda_envs/agentic-sfm

cd "$PROJECT_DIR"
mkdir -p logs checkpoints/verl_grpo_3gpu

export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export HF_HOME=/scratch/kcwp264/.cache/huggingface
export TRANSFORMERS_OFFLINE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

# ---------------------------------------------------------------------------
# Start the Agentic-SFM tool server on GPU 2 (the 3rd GPU in the allocation).
# ---------------------------------------------------------------------------
LOGFILE="logs/tool_server_${SLURM_JOB_ID}.log"
echo "Starting tool server on GPU 2; logging to ${LOGFILE}"
CUDA_VISIBLE_DEVICES=2 "${CONDA_ENV}/bin/python" tools_server/server.py > "$LOGFILE" 2>&1 &
TOOL_PID=$!

# Wait for tool server health endpoint
for i in $(seq 1 60); do
    if "${CONDA_ENV}/bin/python" - <<'PY'
import httpx, sys
try:
    httpx.get("http://localhost:8765/health", timeout=2.0).raise_for_status()
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
    then
        echo "Tool server is healthy"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "Tool server failed to start" >&2
        kill $TOOL_PID || true
        exit 1
    fi
    sleep 2
done

# ---------------------------------------------------------------------------
# Run verl GRPO on GPUs 0 and 1.
# ---------------------------------------------------------------------------
mapfile -t OVERRIDES < configs/verl/overrides_3gpu_clean.txt
echo "Launching verl GRPO training"
CUDA_VISIBLE_DEVICES=0,1 "${CONDA_ENV}/bin/python" -m verl.trainer.main_ppo_sync \
    "${OVERRIDES[@]}"

# Clean up tool server on normal exit
kill $TOOL_PID 2>/dev/null || true
