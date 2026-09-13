#!/bin/bash
# =============================================================================
# SLURM: Baseline evaluation — direct/random-crop/oracle-crop matchers.
# 1x NVIDIA L40S (Ada, 48 GB) on a single AIRE GPU node.
#
# GPU layout:
#   GPU 0: LoFTR/MASt3R/LightGlue tool server (matcher inference)
#   (the eval script itself is a CPU-only HTTP client — no vLLM, no policy)
#
# Submit:
#   sbatch jobs/run_baselines.slurm
#   sbatch --export=PAIRS=data/hard_pairs_train.json jobs/run_baselines.slurm
#   sbatch --export=PAIRS=data/hard_pairs_val.json,MAX_PAIRS=50 jobs/run_baselines.slurm
# =============================================================================
#SBATCH --job-name=asfm-baselines
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=85G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/kcwp264/logs/agentic-sfm/baselines_%j.out
#SBATCH --error=/scratch/kcwp264/logs/agentic-sfm/baselines_%j.err

set -euo pipefail

echo "=== Agentic-SFM baseline evaluation ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start: $(date)"

PROJECT_ROOT="/scratch/kcwp264/agentic-sfm"
LOG_DIR="/scratch/kcwp264/logs/agentic-sfm"
OUT_DIR="${PROJECT_ROOT}/outputs/baselines"
PAIRS="${PAIRS:-data/hard_pairs_val.json}"
mkdir -p "${LOG_DIR}" "${OUT_DIR}"

# ---------------------------------------------------------------------------
# 1. Environment
# ---------------------------------------------------------------------------
source /scratch/kcwp264/.aire_scratch_env.sh
module load miniforge/24.7.1 || true
module load cuda/12.6.2 || true

CONDA_BASE=$(conda info --base 2>/dev/null || echo "/opt/apps/pkg/interpreters/miniforge/24.7.1")
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate /scratch/kcwp264/.conda_envs/agentic-sfm
set -u
export CONDA_DEFAULT_ENV=agentic-sfm

cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export HF_HOME="/scratch/kcwp264/.cache/huggingface"
export TORCH_HOME="/scratch/kcwp264/.cache/torch"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "Python: $(which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ---------------------------------------------------------------------------
# 2. Start tool server on GPU 0 (LoFTR / MASt3R / LightGlue + MAGSAC pose)
# ---------------------------------------------------------------------------
echo "Starting tool server on GPU 0..."
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="${PROJECT_ROOT}/src" python \
    "${PROJECT_ROOT}/tools_server/server.py" &
TOOL_PID=$!

cleanup() {
    echo "Cleaning up tool server (PID ${TOOL_PID})..."
    kill ${TOOL_PID} 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for tool server..."
for i in $(seq 1 60); do
    if curl -s http://localhost:8765/health > /dev/null 2>&1; then
        echo "Tool server ready (took ~$((i*2))s)"
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "ERROR: Tool server failed to start after 120s"
        exit 1
    fi
    if ! kill -0 ${TOOL_PID} 2>/dev/null; then
        echo "ERROR: Tool server process died"
        exit 1
    fi
    sleep 2
done

# ---------------------------------------------------------------------------
# 3. Run baseline evaluation (CPU client; results staged then copied out)
# ---------------------------------------------------------------------------
WORK_DIR="${TMP_SHARED:-${OUT_DIR}/tmp}/${SLURM_JOB_ID}"
mkdir -p "${WORK_DIR}"

MAX_PAIRS_ARG=()
if [ -n "${MAX_PAIRS:-}" ]; then
    MAX_PAIRS_ARG=(--max-pairs "${MAX_PAIRS}")
fi

echo "Running baselines on ${PAIRS} ..."
python scripts/run_baselines.py \
    --pairs "${PAIRS}" \
    --tool-server-url http://localhost:8765 \
    --output "${WORK_DIR}/baseline_results.json" \
    "${MAX_PAIRS_ARG[@]}" \
    2>&1 | tee "${LOG_DIR}/baselines_${SLURM_JOB_ID}.log"

EVAL_EXIT=${PIPESTATUS[0]}

# ---------------------------------------------------------------------------
# 4. Copy results to outputs/baselines/
# ---------------------------------------------------------------------------
if [ -f "${WORK_DIR}/baseline_results.json" ]; then
    cp -f "${WORK_DIR}/baseline_results.json" \
        "${OUT_DIR}/baseline_results_${SLURM_JOB_ID}.json"
    cp -f "${WORK_DIR}/baseline_results.json" \
        "${OUT_DIR}/baseline_results_latest.json"
    echo "Results copied to ${OUT_DIR}/baseline_results_${SLURM_JOB_ID}.json"
else
    echo "WARNING: no results file at ${WORK_DIR}/baseline_results.json"
fi

echo "Eval exit code: ${EVAL_EXIT}"
echo "End: $(date)"
exit "${EVAL_EXIT}"
