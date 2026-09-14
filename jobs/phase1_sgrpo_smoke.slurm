#!/bin/bash
# =============================================================================
# SLURM: Phase 1 — S-GRPO smoke test for agentic matching.
# 3x NVIDIA L40S (Ada, 48 GB) on a single AIRE GPU node.
#
# Verifies the full S-GRPO training loop end-to-end on a tiny slice of data
# (50 pairs, 2 epochs, group_size=4) with CGI oracle injection and the
# accumulative tool reward enabled — before launching the real phase1_grpo run.
#
# GPU layout:
#   GPU 0: vLLM rollout server (Qwen3.5-2B inference, LoRA hot-reload)
#   GPU 1: Training (LoRA gradient updates)
#   GPU 2: Matcher inference (tool server: crop / match / pose)
#
# Submit:
#   sbatch jobs/phase1_sgrpo_smoke.slurm
# =============================================================================
#SBATCH --job-name=asfm-sgrpo-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/kcwp264/logs/agentic-sfm/sgrpo_smoke_%j.out
#SBATCH --error=/scratch/kcwp264/logs/agentic-sfm/sgrpo_smoke_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

echo "=== Agentic-SFM Phase 1 S-GRPO Smoke Test ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)"
echo "Start: $(date)"

# ---------------------------------------------------------------------------
# 1. Environment
# ---------------------------------------------------------------------------
# AIRE scratch env first (before HF/uv/torch), then miniforge + cuda modules.
source /scratch/kcwp264/.aire_scratch_env.sh
module load miniforge/24.7.1 || true
module load cuda/12.4.1 || true

CONDA_BASE=$(conda info --base 2>/dev/null || echo "/opt/apps/pkg/interpreters/miniforge/24.7.1")
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate /scratch/kcwp264/.conda_envs/agentic-sfm
set -u
export CONDA_DEFAULT_ENV=agentic-sfm

export CUDA_HOME="${CUDA_HOME:-/opt/apps/pkg/compilers/cuda/12.4.1}"
export NCCL_P2P_DISABLE=1
export NCCL_NET=Socket
export NCCL_IB_DISABLE=1
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Redirect all caches to scratch to avoid home disk quota issues
export TRITON_CACHE_DIR="/scratch/kcwp264/.triton_cache"
export FLASHINFER_WORKSPACE_DIR="/scratch/kcwp264/.flashinfer"
export FLASHINFER_JIT_DIR="/scratch/kcwp264/.flashinfer/jit"
export XDG_CACHE_HOME="/scratch/kcwp264/.cache/xdg"
export HF_HOME="/scratch/kcwp264/.cache/huggingface"
export TORCH_HOME="/scratch/kcwp264/.cache/torch"
export VLLM_CACHE_ROOT="/scratch/kcwp264/.cache/vllm"
export WANDB_DIR=/scratch/kcwp264/wandb
export WANDB_CACHE_DIR=/scratch/kcwp264/.cache/wandb
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}
mkdir -p "${TRITON_CACHE_DIR}" "${FLASHINFER_WORKSPACE_DIR}" "${FLASHINFER_JIT_DIR}" "${XDG_CACHE_HOME}" "${VLLM_CACHE_ROOT}"

# ---------------------------------------------------------------------------
# 2. Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT="/scratch/kcwp264/agentic-sfm"
CONFIG="${PROJECT_ROOT}/configs/phase1_sgrpo_smoke.yaml"
OUTPUT_DIR="/scratch/kcwp264/agentic-sfm/outputs/sgrpo_smoke"
LOG_DIR="/scratch/kcwp264/logs/agentic-sfm"
SUBSET_FILE="${PROJECT_ROOT}/data/hard_pairs_train_smoke50.json"
TRAIN_LOG="${LOG_DIR}/sgrpo_smoke_training_${SLURM_JOB_ID}.log"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}/checkpoints" "${OUTPUT_DIR}/logs"

cd "${PROJECT_ROOT}"

# Pip-bundled CUDA libs + PYTHONPATH (agentic_sfm package under src/)
source "${PROJECT_ROOT}/scripts/agentic_sfm_env.sh"

# ---------------------------------------------------------------------------
# 3. Build the 50-pair training subset (idempotent)
# ---------------------------------------------------------------------------
if [ ! -f "${SUBSET_FILE}" ]; then
    echo "Building 50-pair smoke subset: ${SUBSET_FILE}"
    python - <<'PY'
import json
src = "/scratch/kcwp264/agentic-sfm/data/hard_pairs_train.json"
dst = "/scratch/kcwp264/agentic-sfm/data/hard_pairs_train_smoke50.json"
with open(src) as f:
    pairs = json.load(f)
subset = pairs[:50]
with open(dst, "w") as f:
    json.dump(subset, f, indent=2)
from collections import Counter
print(f"Wrote {len(subset)} pairs to {dst} | difficulties: {dict(Counter(p['difficulty'] for p in subset))}")
PY
else
    echo "Smoke subset already exists: ${SUBSET_FILE}"
fi

# Runtime check (transformers/vllm/peft/torch versions for Qwen3.5)
python - <<'PY'
from agentic_sfm.constants import assert_qwen35_runtime
assert_qwen35_runtime()
print("Policy runtime check passed")
PY

# Make sure server PIDs are always reaped, even if the script exits early
VLLM_PID=""
TOOL_PID=""
cleanup() {
    kill ${VLLM_PID} ${TOOL_PID} 2>/dev/null || true
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# 4. Start tool server on GPU 2 (LoFTR / MASt3R / LightGlue + MAGSAC pose)
#    Start before vLLM so both boot in parallel.
# ---------------------------------------------------------------------------
echo "Starting tool server on GPU 2..."
CUDA_VISIBLE_DEVICES=2 PYTHONPATH="${PROJECT_ROOT}/src" python \
    "${PROJECT_ROOT}/tools_server/server.py" &
TOOL_PID=$!

# ---------------------------------------------------------------------------
# 5. Start vLLM rollout server on GPU 0 (with LoRA support)
# ---------------------------------------------------------------------------
echo "Starting vLLM server on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model "Qwen/Qwen3.5-2B" \
    --enable-lora \
    --max-loras 1 \
    --max-lora-rank 32 \
    --port 8000 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --limit-mm-per-prompt '{"image":6,"video":0}' \
    --mm-processor-kwargs '{"max_pixels":1003520,"min_pixels":3136}' \
    --mm-processor-cache-gb 0 \
    --trust-remote-code \
    --dtype bfloat16 \
    --api-key EMPTY \
    --enforce-eager &
VLLM_PID=$!

# Wait for tool server (loads matcher weights on GPU 2)
echo "Waiting for tool server..."
for i in $(seq 1 120); do
    if curl -s http://localhost:8765/health > /dev/null 2>&1; then
        echo "Tool server ready (took ${i}x2s = $((i*2))s)"
        break
    fi
    if ! kill -0 ${TOOL_PID} 2>/dev/null; then
        echo "ERROR: Tool server process died"
        exit 1
    fi
    if [ $i -eq 120 ]; then
        echo "ERROR: Tool server failed to start after 240s"
        exit 1
    fi
    sleep 2
done

# Wait for vLLM (model load + CUDA graphs; slowest to come up)
echo "Waiting for vLLM..."
for i in $(seq 1 240); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "vLLM ready (took ${i}x2s = $((i*2))s)"
        break
    fi
    if ! kill -0 ${VLLM_PID} 2>/dev/null; then
        echo "ERROR: vLLM process died"
        exit 1
    fi
    if [ $i -eq 240 ]; then
        echo "ERROR: vLLM failed to start after 480s"
        echo "=== GPU status ==="
        nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || true
        echo "=== vLLM process ==="
        ps aux | grep vllm | grep -v grep || true
        exit 1
    fi
    if [ $((i % 30)) -eq 0 ]; then
        echo "  Still waiting for vLLM... (${i}x2s = $((i*2))s elapsed)"
    fi
    sleep 2
done

# ---------------------------------------------------------------------------
# 6. Run S-GRPO smoke training on GPU 1
#    Checkpoints land directly in ${OUTPUT_DIR}/checkpoints via --output-dir.
# ---------------------------------------------------------------------------
echo "Starting S-GRPO smoke training..."
set +e
CUDA_VISIBLE_DEVICES=1 python scripts/run_grpo.py \
    --config "${CONFIG}" \
    --tool-server-url "http://localhost:8765" \
    --vllm-url "http://localhost:8000" \
    --output-dir "${OUTPUT_DIR}" \
    2>&1 | tee "${TRAIN_LOG}"
TRAIN_EXIT=${PIPESTATUS[0]}
set -e

# ---------------------------------------------------------------------------
# 7. Cleanup servers, then copy checkpoints + logs to outputs/sgrpo_smoke/
# ---------------------------------------------------------------------------
echo "Cleaning up servers..."
cleanup
wait ${VLLM_PID} ${TOOL_PID} 2>/dev/null || true
trap - EXIT

# Checkpoints already land in ${OUTPUT_DIR}/checkpoints via --output-dir.
# Copy the training + slurm logs alongside them so everything for this smoke
# run lives under outputs/sgrpo_smoke/.
echo "Copying logs to ${OUTPUT_DIR}/logs/ ..."
mkdir -p "${OUTPUT_DIR}/checkpoints" "${OUTPUT_DIR}/logs"
cp -f "${TRAIN_LOG}" "${OUTPUT_DIR}/logs/" 2>/dev/null || true
cp -f "/scratch/kcwp264/logs/agentic-sfm/sgrpo_smoke_${SLURM_JOB_ID}.out" \
      "/scratch/kcwp264/logs/agentic-sfm/sgrpo_smoke_${SLURM_JOB_ID}.err" \
      "${OUTPUT_DIR}/logs/" 2>/dev/null || true

echo "=== Smoke test artifacts ==="
find "${OUTPUT_DIR}" -maxdepth 3 -type f | sort || true

echo "Training exit code: ${TRAIN_EXIT}"
echo "End: $(date)"
seff "${SLURM_JOB_ID}" 2>/dev/null || true
exit ${TRAIN_EXIT}
