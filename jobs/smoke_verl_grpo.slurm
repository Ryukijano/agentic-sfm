#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=85G
#SBATCH --time=00:45:00
#SBATCH --job-name=verl-grpo-smoke
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -e

source /scratch/kcwp264/.aire_scratch_env.sh

PROJECT_DIR=/scratch/kcwp264/agentic-sfm
CONDA_ENV=/scratch/kcwp264/.conda_envs/agentic-sfm

cd "$PROJECT_DIR"
mkdir -p logs

export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export HF_HOME=/scratch/kcwp264/.cache/huggingface
export TRANSFORMERS_OFFLINE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
unset ROCR_VISIBLE_DEVICES

# ---------------------------------------------------------------------------
# Start the tool server on CPU (no GPU needed for SfM matching).
# ---------------------------------------------------------------------------
LOGFILE="logs/tool_server_${SLURM_JOB_ID}.log"
echo "Starting tool server on CPU; logging to ${LOGFILE}"
CUDA_VISIBLE_DEVICES= "${CONDA_ENV}/bin/python" tools_server/server.py > "$LOGFILE" 2>&1 &
TOOL_PID=$!

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
# Smoke test: rollout + reward on 10 validation prompts.
# This sets val_only=true so the job exits after validation.
# ---------------------------------------------------------------------------
mapfile -t OVERRIDES < configs/verl/overrides_smoke_1gpu.txt
echo "Launching verl GRPO smoke test (1-GPU with param_offload)"
CUDA_VISIBLE_DEVICES=0 "${CONDA_ENV}/bin/python" -m verl.trainer.main_ppo_sync \
    "${OVERRIDES[@]}" \
    trainer.val_only=true \
    data.val_max_samples=10 \
    data.val_batch_size=10 \
    data.train_max_samples=10 \
    actor_rollout_ref.rollout.n=1 \
    +actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
    actor_rollout_ref.rollout.max_model_len=4096 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192

kill $TOOL_PID 2>/dev/null || true
