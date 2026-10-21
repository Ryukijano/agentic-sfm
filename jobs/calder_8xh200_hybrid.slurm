#!/bin/bash
# =============================================================================
# SLURM: Phase 2 hybrid — scene GRPO on Calder 8x NVIDIA H200 NVL (141 GB each).
#
# Single 8-GPU allocation produces THREE results in one job:
#   * Multi-scene Hybrid GRPO policy (MegaDepth 0015/0022 + 32 scene-disjoint
#     ScanNet indoor scenes; vLLM TP=2, 16k context; frozen KL anchor on GPU)
#   * OPSFT ablation arm A: topk 10% direction  -> outputs/scene_opsft_topk10
#   * OPSFT ablation arm B: exact_support       -> outputs/scene_opsft_exact
#
# GPU layout:
#   GPU 0,1 : vLLM server (--tensor-parallel-size 2, --max-model-len 16384;
#             fixes the 129x "Input length > 8192" truncations from job 8237240)
#   GPU 2   : Tool server (LoFTR + MASt3R + VGGT-1B + COLMAP pycolmap)
#   GPU 3,4 : run_scene_grpo.py (cuda:0=GPU3 trainable LoRA, cuda:1=GPU4 frozen KL)
#   GPU 5   : OPSFT arm topk10  (run_sft_opsft.py -> outputs/scene_opsft_topk10)
#   GPU 6   : OPSFT arm exact   (run_sft_opsft.py -> outputs/scene_opsft_exact)
#   GPU 7   : Dedicated eval tool server (frozen-manifest eval + bench_pose_priors
#             run against finished checkpoints at end of job)
#
# Submit:  sbatch jobs/calder_8xh200_hybrid.slurm
# =============================================================================
#SBATCH --job-name=asfm-calder-8xh200
#SBATCH --partition=calder-gpu-hopper
#SBATCH --gres=gpu:nvidia_h200:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=1400G
#SBATCH --time=12:00:00
#SBATCH --output=logs/agentic-sfm/calder_8xh200_%j.out
#SBATCH --error=logs/agentic-sfm/calder_8xh200_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

echo "=== Agentic-SFM Phase 2 Scene GRPO — Calder 8xH200 ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)"
echo "Start: $(date)"

# Slurm runs a spooled copy of this script, so BASH_SOURCE points at
# /var/spool/slurm; resolve the repo from the submit dir (or an override).
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?submit from the repo root}}"
[[ -f "${PROJECT_ROOT}/scripts/calder_agentic_sfm_env.sh" ]] || { echo "ERROR: ${PROJECT_ROOT} is not the agentic-sfm repo root"; exit 1; }
CONFIG="${PROJECT_ROOT}/configs/phase2_calder_8xh200.yaml"
OPSFT_CONFIG="${PROJECT_ROOT}/configs/phase2_sft_opsft.yaml"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/phase2_calder/run_${SLURM_JOB_ID}"
SCENE_SFT="${PROJECT_ROOT}/outputs/scene_sft_graph_retry2/checkpoints/epoch_5"
LOG_DIR="${PROJECT_ROOT}/logs/agentic-sfm"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

source "${PROJECT_ROOT}/scripts/calder_agentic_sfm_env.sh"
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1  # required for /v1/load_lora_adapter (otherwise 404)

# ---------------------------------------------------------------------------
# Transient storage: all ephemeral work lands on $CALDER_FLASH (node-local NVMe)
# or /dev/shm — never on the 200 GB scratch quota.
# ---------------------------------------------------------------------------
FLASH="${CALDER_FLASH:-/dev/shm/${USER}}"
export SFM_OUTPUT_ROOT="${FLASH}/asfm_sfm_${SLURM_JOB_ID}"
export LORA_STAGING_DIR="${FLASH}/asfm_lora_${SLURM_JOB_ID}"
mkdir -p "${SFM_OUTPUT_ROOT}" "${LORA_STAGING_DIR}"
echo "SFM_OUTPUT_ROOT=${SFM_OUTPUT_ROOT}"
echo "LORA_STAGING_DIR=${LORA_STAGING_DIR}"

# Dynamic free ports (find_free_port is exported by calder_agentic_sfm_env.sh /
# agentic_sfm_env.sh).
VLLM_PORT=$(find_free_port $((39000 + SLURM_JOB_ID % 20000)))
TOOL_PORT=$(find_free_port $((19000 + SLURM_JOB_ID % 20000)))
EVAL_TOOL_PORT=$(find_free_port $((49000 + SLURM_JOB_ID % 20000)))
export VLLM_PORT TOOL_PORT EVAL_TOOL_PORT
echo "Ports -> vLLM:${VLLM_PORT} tool:${TOOL_PORT} eval-tool:${EVAL_TOOL_PORT}"

cd "${PROJECT_ROOT}"

if [[ ! -f "${SCENE_SFT}/adapter_model.safetensors" ]]; then
    echo "ERROR: scene SFT adapter missing at ${SCENE_SFT}"
    exit 1
fi
echo "Scene SFT warmstart: ${SCENE_SFT}"

python - <<'PY'
from agentic_sfm.constants import assert_qwen35_runtime
assert_qwen35_runtime()
print("Policy runtime check passed")
PY

python - <<'PY'
import os
import wandb
try:
    user = wandb.api.viewer()["username"]
except Exception as e:
    raise SystemExit(f"ERROR: W&B auth failed ({type(e).__name__}: {e})")
print(f"W&B authenticated as {user}; project={os.environ.get('WANDB_PROJECT', 'agentic-sfm')}")
PY

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
VLLM_PID=""; TOOL_PID=""; EVAL_TOOL_PID=""; TRAIN_PID=""
OPSFT_TOPK_PID=""; OPSFT_EXACT_PID=""; WATCH_PID=""
cleanup() {
    echo "Cleaning up servers..."
    kill ${WATCH_PID:-} ${TRAIN_PID:-} ${OPSFT_TOPK_PID:-} ${OPSFT_EXACT_PID:-} \
         ${VLLM_PID:-} ${TOOL_PID:-} ${EVAL_TOOL_PID:-} 2>/dev/null || true
    if [[ -n "${TOOL_PID_FILE:-}" && -f "${TOOL_PID_FILE}" ]]; then
        kill "$(cat "${TOOL_PID_FILE}")" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# GPUs 0,1: vLLM rollout server, TP=2, 16k context.
# ---------------------------------------------------------------------------
echo "Starting vLLM server on GPUs 0,1 (TP=2, max-model-len 16384)..."
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
    --model "Qwen/Qwen3.5-2B" \
    --tensor-parallel-size 2 \
    --enable-lora \
    --max-loras 1 \
    --max-lora-rank 32 \
    --port "${VLLM_PORT}" \
    --gpu-memory-utilization 0.90 \
    --max-model-len 16384 \
    --limit-mm-per-prompt '{"image":4,"video":0}' \
    --mm-processor-kwargs '{"max_pixels":401408,"min_pixels":3136}' \
    --mm-processor-cache-gb 0 \
    --trust-remote-code \
    --dtype bfloat16 \
    --api-key EMPTY \
    > "${LOG_DIR}/calder_vllm_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!

for i in $(seq 1 300); do
    if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "vLLM ready (took $((i*2))s)"
        break
    fi
    if ! kill -0 ${VLLM_PID} 2>/dev/null; then
        echo "ERROR: vLLM process died; see ${LOG_DIR}/calder_vllm_${SLURM_JOB_ID}.log"
        exit 1
    fi
    if [ $i -eq 300 ]; then echo "ERROR: vLLM failed to become ready"; exit 1; fi
    sleep 2
done

# ---------------------------------------------------------------------------
# GPU 2: tool server (LoFTR + MASt3R + VGGT-1B + COLMAP), with watchdog.
# ---------------------------------------------------------------------------
echo "Starting tool server on GPU 2..."
TOOL_PID_FILE="${OUTPUT_DIR}/.tool_server_${TOOL_PORT}.pid"
CUDA_VISIBLE_DEVICES=2 PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/../mast3r:${PROJECT_ROOT}/../mast3r/dust3r:${PROJECT_ROOT}/../mast3r/dust3r/croco" python \
    "${PROJECT_ROOT}/tools_server/server.py" --port "${TOOL_PORT}" &
TOOL_PID=$!
echo "${TOOL_PID}" > "${TOOL_PID_FILE}"

for i in $(seq 1 60); do
    if curl -s "http://localhost:${TOOL_PORT}/health" > /dev/null 2>&1; then
        echo "Tool server ready on :${TOOL_PORT} (pid ${TOOL_PID})"
        break
    fi
    if ! kill -0 ${TOOL_PID} 2>/dev/null; then
        echo "ERROR: Tool server process died before bind"
        exit 1
    fi
    if [ $i -eq 60 ]; then echo "ERROR: Tool server failed"; exit 1; fi
    sleep 2
done

# ---------------------------------------------------------------------------
# GPUs 5,6: OPSFT ablation arms in background (staged 8237240 direction).
# Each arm extracts its direction artifact if the staging bundle didn't ship it.
# ---------------------------------------------------------------------------
DIRECTION_BASE="outputs/scene_sft_graph_retry2/checkpoints/epoch_5"
DIRECTION_TRAINED="${DIRECTION_TRAINED:-outputs/phase2_hybrid/latest/checkpoints/latest}"

run_opsft_arm() {  # $1=gpu  $2=selection  $3=topk_frac("-" for none)  $4=outdir
    local gpu="$1" sel="$2" frac="$3" outdir="$4"
    local art="outputs/opsft/direction_scene_grpo_${sel}.pt"
    mkdir -p "${outdir}" "$(dirname "${art}")"
    if [ ! -s "${art}" ]; then
        echo "[opsft ${sel}] extracting direction artifact -> ${art}"
        local extra=()
        if [ "${frac}" != "-" ]; then extra+=(--topk-frac "${frac}"); fi
        CUDA_VISIBLE_DEVICES="${gpu}" python scripts/extract_update_direction.py \
            --base "${DIRECTION_BASE}" --trained "${DIRECTION_TRAINED}" \
            --output "${art}" --selection "${sel}" "${extra[@]}"
    fi
    echo "[opsft ${sel}] training on GPU ${gpu} -> ${outdir}"
    CUDA_VISIBLE_DEVICES="${gpu}" python scripts/run_sft_opsft.py \
        --config "${OPSFT_CONFIG}" \
        --output-dir "${outdir}" \
        --direction-artifact "${art}" \
        > "${LOG_DIR}/calder_opsft_${sel}_${SLURM_JOB_ID}.log" 2>&1
}

set +e
run_opsft_arm 5 topk 0.1 "outputs/scene_opsft_topk10" &
OPSFT_TOPK_PID=$!
run_opsft_arm 6 exact_support - "outputs/scene_opsft_exact" &
OPSFT_EXACT_PID=$!
set -e
echo "OPSFT arms launched: topk10 pid=${OPSFT_TOPK_PID} (GPU5), exact pid=${OPSFT_EXACT_PID} (GPU6)"

# ---------------------------------------------------------------------------
# GPUs 3,4: scene-level GRPO training (cuda:0 = GPU3 trainable, cuda:1 = GPU4 KL).
# ---------------------------------------------------------------------------
echo "Starting Phase 2 scene-level GRPO training on GPUs 3,4..."
set +e
CUDA_VISIBLE_DEVICES=3,4 python scripts/run_scene_grpo.py \
    --config "${CONFIG}" \
    --tool-server-url "http://localhost:${TOOL_PORT}" \
    --vllm-url "http://localhost:${VLLM_PORT}" \
    --output-dir "${OUTPUT_DIR}" \
    > >(tee "${LOG_DIR}/phase2_calder_train_${SLURM_JOB_ID}.log") 2>&1 &
TRAIN_PID=$!
set -e

chmod +x "${PROJECT_ROOT}/scripts/watch_tool_server.sh" 2>/dev/null || true
"${PROJECT_ROOT}/scripts/watch_tool_server.sh" \
    --port "${TOOL_PORT}" \
    --project-root "${PROJECT_ROOT}" \
    --protect-pid "${TRAIN_PID}" \
    --pid-file "${TOOL_PID_FILE}" \
    --gpu 2 \
    --interval 10 \
    --fail-threshold 3 \
    --max-restarts 5 \
    >> "${LOG_DIR}/calder_8xh200_${SLURM_JOB_ID}.out" 2>&1 &
WATCH_PID=$!
echo "Tool-server watchdog pid=${WATCH_PID}"

set +e
wait ${TRAIN_PID}
TRAIN_EXIT=$?
set -e
echo "Training exit code: ${TRAIN_EXIT}"

set +e
wait ${OPSFT_TOPK_PID};  OPSFT_TOPK_EXIT=$?
wait ${OPSFT_EXACT_PID}; OPSFT_EXACT_EXIT=$?
set -e
echo "OPSFT exits: topk10=${OPSFT_TOPK_EXIT} exact=${OPSFT_EXACT_EXIT}"

kill ${WATCH_PID} 2>/dev/null || true
wait ${WATCH_PID} 2>/dev/null || true

# ---------------------------------------------------------------------------
# GPU 7: dedicated tool server for end-of-job frozen-manifest evaluation.
# ---------------------------------------------------------------------------
echo "Starting eval tool server on GPU 7 :${EVAL_TOOL_PORT}..."
CUDA_VISIBLE_DEVICES=7 PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/../mast3r:${PROJECT_ROOT}/../mast3r/dust3r:${PROJECT_ROOT}/../mast3r/dust3r/croco" python \
    "${PROJECT_ROOT}/tools_server/server.py" --port "${EVAL_TOOL_PORT}" &
EVAL_TOOL_PID=$!
for i in $(seq 1 60); do
    curl -s "http://localhost:${EVAL_TOOL_PORT}/health" > /dev/null 2>&1 && break
    sleep 2
done

if curl -s "http://localhost:${EVAL_TOOL_PORT}/health" > /dev/null 2>&1; then
    BEST_CKPT="${OUTPUT_DIR}/checkpoints/best_eval"
    [ -d "${BEST_CKPT}" ] || BEST_CKPT="${OUTPUT_DIR}/checkpoints/latest"
    echo "Evaluating ${BEST_CKPT} on the frozen manifest (GPU 7 tool server)..."
    python scripts/eval_scene.py \
        --config "${CONFIG}" \
        --tool-server-url "http://localhost:${EVAL_TOOL_PORT}" \
        --vllm-url "http://localhost:${VLLM_PORT}" \
        --mode agent \
        --lora-checkpoint "${BEST_CKPT}" \
        --output-dir "${OUTPUT_DIR}/frozen_eval" \
        > "${LOG_DIR}/calder_eval_${SLURM_JOB_ID}.log" 2>&1 || true

    echo "Running bench_pose_priors (frozen manifest, all matchers)..."
    CUDA_VISIBLE_DEVICES=7 python scripts/bench_pose_priors.py \
        --output-dir "${OUTPUT_DIR}/bench_pose_priors" \
        > "${LOG_DIR}/calder_bench_${SLURM_JOB_ID}.log" 2>&1 || true
else
    echo "WARN: eval tool server did not come up; skipping end-of-job eval."
fi

cleanup
trap - EXIT

cp -f "${LOG_DIR}/phase2_calder_train_${SLURM_JOB_ID}.log" "${OUTPUT_DIR}/logs/" 2>/dev/null || true

echo "=== EPILOG ==="
echo "Training exit: ${TRAIN_EXIT}  |  OPSFT topk10: ${OPSFT_TOPK_EXIT}  exact: ${OPSFT_EXACT_EXIT}"
echo "GRPO checkpoints: ${OUTPUT_DIR}/checkpoints/"
echo "OPSFT arms: outputs/scene_opsft_topk10  outputs/scene_opsft_exact"
echo "End: $(date)"
seff "${SLURM_JOB_ID}" 2>/dev/null || true
exit ${TRAIN_EXIT}
