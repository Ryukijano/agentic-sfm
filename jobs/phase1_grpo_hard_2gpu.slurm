#!/bin/bash
# =============================================================================
# Phase 1b (2-GPU variant) — GRPO on medium+hard pairs, unique output dir.
#
# Same run as jobs/phase1_grpo_hard.slurm but needs only 2 free GPUs:
#   GPU 0: vLLM rollout server + tool server (vLLM gpu-memory-utilization 0.50)
#   GPU 1: LoRA training
#
# Sibling arbitration vs. job 7924692 (the 3-GPU variant): first to start wins.
# On startup this script cancels 7924692 if it is still queued; if 7924692
# already started (or finished) this job exits cleanly so the two never share
# outputs/phase1_hard.
#
# DO NOT submit while 7894645 is running. This job snapshots best_eval/best
# into outputs/phase1_hard/warmstart and must not share that run's checkpoint
# directory.
#
#   sbatch jobs/phase1_grpo_hard_2gpu.slurm
# =============================================================================
#SBATCH --job-name=asfm-grpo-hard2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=140G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/kcwp264/logs/agentic-sfm/grpo_hard2_%j.out
#SBATCH --error=/scratch/kcwp264/logs/agentic-sfm/grpo_hard2_%j.err

set -euo pipefail

echo "=== Agentic-SFM Phase 1b GRPO (medium+hard, 2-GPU) ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)"
echo "Start: $(date)"

SIBLING_JOB=7924692

# First-to-start arbitration with the 3-GPU sibling. If it is still queued,
# cancel it; if it already started or finished, exit so we never share
# outputs/phase1_hard.
for i in $(seq 1 10); do
    sib_state="$(squeue -j ${SIBLING_JOB} -h -o '%T' 2>/dev/null | head -1 || true)"
    if [[ -z "${sib_state}" ]]; then
        break
    fi
    if [[ "${sib_state}" == "RUNNING" ]]; then
        echo "Sibling ${SIBLING_JOB} is already RUNNING — exiting cleanly."
        exit 0
    fi
    scancel "${SIBLING_JOB}" 2>/dev/null || true
    sleep 2
done
sib_sacct="$(sacct -j ${SIBLING_JOB} -n -X -o State 2>/dev/null | head -1 | tr -d ' ' || true)"
case "${sib_sacct}" in
    ""|PENDING|CANCELLED*|FAILED*|TIMEOUT*|NODE_FAIL*|OUT_OF_MEMORY*)
        echo "Sibling ${SIBLING_JOB} cancelled/absent/failed (state='${sib_sacct}') — this job proceeds."
        ;;
    RUNNING|COMPLETING)
        echo "Sibling ${SIBLING_JOB} is live (state='${sib_sacct}') — exiting cleanly."
        exit 0
        ;;
    COMPLETED)
        echo "Sibling ${SIBLING_JOB} COMPLETED and may own outputs/phase1_hard — exiting cleanly."
        exit 0
        ;;
    *)
        echo "Sibling ${SIBLING_JOB} state='${sib_sacct}' — exiting cleanly to avoid sharing outputs."
        exit 0
        ;;
esac

PROJECT_ROOT="/scratch/kcwp264/agentic-sfm"
CONFIG="${PROJECT_ROOT}/configs/phase1_grpo_hard_2gpu.yaml"
OUTPUT_DIR="/scratch/kcwp264/agentic-sfm/outputs/phase1_hard"
LOG_DIR="/scratch/kcwp264/logs/agentic-sfm"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

source /scratch/kcwp264/.aire_scratch_env.sh
source "${PROJECT_ROOT}/scripts/agentic_sfm_env.sh"
module load cuda/12.4.1 2>/dev/null || true
export CUDA_HOME="${CUDA_HOME:-/opt/apps/pkg/compilers/cuda/12.4.1}"
export NCCL_P2P_DISABLE=1
export NCCL_NET=Socket
export NCCL_IB_DISABLE=1
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export TRITON_CACHE_DIR="/scratch/kcwp264/.triton_cache"
export FLASHINFER_WORKSPACE_DIR="/scratch/kcwp264/.flashinfer"
export FLASHINFER_JIT_DIR="/scratch/kcwp264/.flashinfer/jit"
export XDG_CACHE_HOME="/scratch/kcwp264/.cache/xdg"
export HF_HOME="/scratch/kcwp264/.cache/huggingface"
export TORCH_HOME="/scratch/kcwp264/.cache/torch"
export VLLM_CACHE_ROOT="/scratch/kcwp264/.cache/vllm"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export FLASHINFER_DISABLE_VERSION_CHECK=1
mkdir -p "${TRITON_CACHE_DIR}" "${FLASHINFER_WORKSPACE_DIR}" "${FLASHINFER_JIT_DIR}" "${XDG_CACHE_HOME}" "${VLLM_CACHE_ROOT}"

cd "${PROJECT_ROOT}"

if squeue -j 7894645 -h -o '%T' 2>/dev/null | grep -Eq 'RUNNING|PENDING'; then
  echo "ERROR: job 7894645 is still live. Do not share its checkpoint dir."
  echo "Wait for it to finish, then resubmit."
  exit 1
fi

VLLM_PORT=$((39000 + SLURM_JOB_ID % 20000))
TOOL_SERVER_PORT=$((19000 + SLURM_JOB_ID % 20000))
export VLLM_PORT TOOL_SERVER_PORT
echo "vLLM port: ${VLLM_PORT}  tool port: ${TOOL_SERVER_PORT}"

# Freeze the Phase-1 adapter so training never reads a live write path.
WARM_DST="${OUTPUT_DIR}/warmstart"
WARM_SRC=""
for cand in \
  "${PROJECT_ROOT}/outputs/phase1/checkpoints/best_eval" \
  "${PROJECT_ROOT}/outputs/phase1/checkpoints/best" \
  "${PROJECT_ROOT}/outputs/phase1/checkpoints/latest"
do
  if [[ -f "${cand}/adapter_model.safetensors" ]]; then
    WARM_SRC="${cand}"
    break
  fi
done
if [[ -z "${WARM_SRC}" ]]; then
  echo "ERROR: no Phase-1 adapter under outputs/phase1/checkpoints/{best_eval,best,latest}"
  exit 1
fi
mkdir -p "${WARM_DST}"
cp -a "${WARM_SRC}/." "${WARM_DST}/"
if [[ ! -f "${WARM_DST}/adapter_model.safetensors" ]]; then
  echo "ERROR: snapshot ${WARM_DST} has no adapter_model.safetensors"
  exit 1
fi
echo "Warm-start snapshot: ${WARM_SRC} -> ${WARM_DST}"

KL_ADAPTER="$(python - <<'PY'
import yaml
print(yaml.safe_load(open("configs/phase1_grpo_hard_2gpu.yaml"))["model"].get("kl_adapter") or "")
PY
)"
if [[ -n "${KL_ADAPTER}" && ! -d "${KL_ADAPTER}" ]]; then
  echo "ERROR: kl_adapter missing at ${KL_ADAPTER}"
  exit 1
fi
echo "KL adapter: ${KL_ADAPTER:-<none>}"

echo "W&B dir: ${WANDB_DIR:-<unset>}  cache: ${WANDB_CACHE_DIR:-<unset>}"
python - <<'PY'
import os
import wandb
try:
    user = wandb.api.viewer()["username"]
except Exception as e:
    raise SystemExit(f"ERROR: W&B auth failed ({type(e).__name__}: {e})")
print(f"W&B authenticated as {user}; project={os.environ.get('WANDB_PROJECT', 'agentic-sfm')}")
PY

python - <<'PY'
from agentic_sfm.constants import assert_qwen35_runtime
assert_qwen35_runtime()
print("Policy runtime check passed")
PY

# Tool server first (shares GPU 0 with vLLM — inference only)
echo "Starting tool server on GPU 0..."
TOOL_PID_FILE="${OUTPUT_DIR}/.tool_server_${TOOL_SERVER_PORT}.pid"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="${PROJECT_ROOT}/src" python \
    "${PROJECT_ROOT}/tools_server/server.py" --port "${TOOL_SERVER_PORT}" &
TOOL_PID=$!
echo "${TOOL_PID}" > "${TOOL_PID_FILE}"

echo "Waiting for tool server..."
for i in $(seq 1 60); do
    if ! kill -0 ${TOOL_PID} 2>/dev/null; then
        echo "ERROR: Tool server process died before bind"
        exit 1
    fi
    if curl -s "http://localhost:${TOOL_SERVER_PORT}/health" > /dev/null 2>&1; then
        echo "Tool server ready on :${TOOL_SERVER_PORT}"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "ERROR: Tool server failed to start"
        kill ${TOOL_PID} 2>/dev/null || true
        exit 1
    fi
    sleep 2
done

echo "Starting vLLM server on GPU 0 (shared with tool server)..."
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model "Qwen/Qwen3.5-2B" \
    --enable-lora \
    --max-loras 1 \
    --max-lora-rank 32 \
    --port "${VLLM_PORT}" \
    --gpu-memory-utilization 0.50 \
    --max-model-len 8192 \
    --limit-mm-per-prompt '{"image":6,"video":0}' \
    --mm-processor-kwargs '{"max_pixels":1003520,"min_pixels":3136}' \
    --mm-processor-cache-gb 0 \
    --trust-remote-code \
    --dtype bfloat16 \
    --api-key EMPTY \
    --enforce-eager &
VLLM_PID=$!

echo "Waiting for vLLM..."
for i in $(seq 1 240); do
    if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "vLLM ready (took ${i}x2s = $((i*2))s)"
        break
    fi
    if [ $i -eq 240 ]; then
        echo "ERROR: vLLM failed to start after 480s"
        kill ${VLLM_PID} ${TOOL_PID} 2>/dev/null || true
        exit 1
    fi
    if [ $((i % 30)) -eq 0 ]; then
        echo "  Still waiting for vLLM... (${i}x2s = $((i*2))s elapsed)"
    fi
    sleep 2
done

echo "Starting GRPO training on GPU 1..."
set +e
CUDA_VISIBLE_DEVICES=1 python scripts/run_grpo.py \
    --config "${CONFIG}" \
    --tool-server-url "http://localhost:${TOOL_SERVER_PORT}" \
    --vllm-url "http://localhost:${VLLM_PORT}" \
    --output-dir "${OUTPUT_DIR}" \
    > >(tee "${LOG_DIR}/grpo_hard2_training_${SLURM_JOB_ID}.log") 2>&1 &
TRAIN_PID=$!

chmod +x "${PROJECT_ROOT}/scripts/watch_tool_server.sh"
"${PROJECT_ROOT}/scripts/watch_tool_server.sh" \
    --port "${TOOL_SERVER_PORT}" \
    --project-root "${PROJECT_ROOT}" \
    --protect-pid "${TRAIN_PID}" \
    --pid-file "${TOOL_PID_FILE}" \
    --gpu 0 \
    --interval 10 \
    --fail-threshold 3 \
    --max-restarts 5 \
    >> "${LOG_DIR}/grpo_hard2_${SLURM_JOB_ID}.out" 2>&1 &
WATCH_PID=$!
echo "Tool-server watchdog pid=${WATCH_PID}"

wait ${TRAIN_PID}
TRAIN_EXIT=$?
set -e
kill ${WATCH_PID} 2>/dev/null || true
wait ${WATCH_PID} 2>/dev/null || true
TOOL_PID="$(cat "${TOOL_PID_FILE}" 2>/dev/null || echo "${TOOL_PID}")"

echo "Cleaning up servers..."
kill ${VLLM_PID} ${TOOL_PID} ${WATCH_PID} 2>/dev/null || true
wait ${VLLM_PID} ${TOOL_PID} 2>/dev/null || true

echo "Training exit code: ${TRAIN_EXIT}"
echo "End: $(date)"
exit ${TRAIN_EXIT}
