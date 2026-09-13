#!/bin/bash
# =============================================================================
# SLURM: Phase 1 — GRPO RL training for agentic matching.
# 3x NVIDIA L40S (Ada, 48 GB) on a single AIRE GPU node.
#
# GPU layout:
#   GPU 0: vLLM rollout server (Qwen3.5-2B inference)
#   GPU 1: Training (LoRA gradient updates)
#   GPU 2: Matcher inference (tool server)
#
# Prerequisites:
#   - Phase 0 completed (hard_pairs.json built)
#   - vLLM >= 0.17 in agentic-sfm env (Qwen3.5-2B; 0.11 cannot load it)
#
# Submit:
#   sbatch jobs/phase1-grpo.slurm
# =============================================================================
#SBATCH --job-name=asfm-p1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL

set -eo pipefail

# ---------------------------------------------------------------------------
# 1. Environment
# ---------------------------------------------------------------------------
source /scratch/kcwp264/.aire_scratch_env.sh
module load miniforge/24.7.1 || true
module load cuda/12.4.1 || true

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
mkdir -p logs outputs/phase1/checkpoints

export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}
export HF_HOME=/scratch/kcwp264/.cache/huggingface
export WANDB_DIR=/scratch/kcwp264/wandb
export WANDB_CACHE_DIR=/scratch/kcwp264/.cache/wandb
export NCCL_P2P_DISABLE=1
export NCCL_NET=Socket
export NCCL_IB_DISABLE=1
export NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "=== Phase 1: GRPO RL training ==="
echo "Node:   $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-n/a}"
date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ---------------------------------------------------------------------------
# 3. Start tool server on GPU 2 (matcher inference)
# ---------------------------------------------------------------------------
echo "[INFO] Starting tool server on GPU 2..."
CUDA_VISIBLE_DEVICES=2 python tools_server/server.py &
TOOL_SERVER_PID=$!
sleep 5

for i in $(seq 1 10); do
  if curl -s http://localhost:8765/health > /dev/null 2>&1; then
    echo "[INFO] Tool server is healthy"
    break
  fi
  echo "[INFO] Waiting for tool server... ($i/10)"
  sleep 3
done

# ---------------------------------------------------------------------------
# 4. Start vLLM rollout server on GPU 0
# ---------------------------------------------------------------------------
echo "[INFO] Starting vLLM rollout server on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3.5-2B \
  --enable-lora \
  --max-loras 1 \
  --max-lora-rank 32 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --limit-mm-per-prompt '{"image":6,"video":0}' \
  --mm-processor-kwargs '{"max_pixels":1003520,"min_pixels":3136}' \
  --api-key EMPTY \
  --enforce-eager \
  &
VLLM_PID=$!
sleep 10

for i in $(seq 1 15); do
  if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "[INFO] vLLM server is healthy"
    break
  fi
  echo "[INFO] Waiting for vLLM... ($i/15)"
  sleep 5
done

# ---------------------------------------------------------------------------
# 5. Launch GRPO training on GPU 1
# ---------------------------------------------------------------------------
echo "[INFO] Launching GRPO training..."
CUDA_VISIBLE_DEVICES=1 python scripts/run_grpo.py \
  --config configs/phase1_grpo.yaml \
  --tool-server-url http://localhost:8765 \
  --vllm-url http://localhost:8000 \
  --output-dir outputs/phase1

# ---------------------------------------------------------------------------
# 6. Cleanup
# ---------------------------------------------------------------------------
kill ${VLLM_PID} ${TOOL_SERVER_PID} 2>/dev/null || true

echo ""
echo "=== Phase 1 finished ==="
date
echo "Checkpoints: outputs/phase1/checkpoints/"
