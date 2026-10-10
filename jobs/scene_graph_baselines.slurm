#!/bin/bash
#SBATCH --job-name=asfm-scene-bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=85G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/kcwp264/logs/agentic-sfm/scene_bench_%j.out
#SBATCH --error=/scratch/kcwp264/logs/agentic-sfm/scene_bench_%j.err

set -euo pipefail

PROJECT_ROOT=/scratch/kcwp264/agentic-sfm
LOG_DIR=/scratch/kcwp264/logs/agentic-sfm
source /scratch/kcwp264/.aire_scratch_env.sh
source "${PROJECT_ROOT}/scripts/agentic_sfm_env.sh"
module load cuda/12.4.1 2>/dev/null || true
export HF_HOME=/scratch/kcwp264/.cache/huggingface
export TORCH_HOME=/scratch/kcwp264/.cache/torch
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export FLASHINFER_DISABLE_VERSION_CHECK=1

cd "${PROJECT_ROOT}"
PORT=$((20000 + SLURM_JOB_ID % 20000))
OUT_DIR="${PROJECT_ROOT}/outputs/baselines/scannet_holdout_${SLURM_JOB_ID}"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

CUDA_VISIBLE_DEVICES=0 python "${PROJECT_ROOT}/tools_server/server.py" --host 127.0.0.1 --port "${PORT}" &
TOOL_PID=$!
cleanup() {
    kill "${TOOL_PID}" 2>/dev/null || true
    wait "${TOOL_PID}" 2>/dev/null || true
}
trap cleanup EXIT

for i in $(seq 1 120); do
    if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        break
    fi
    if ! kill -0 "${TOOL_PID}" 2>/dev/null; then
        echo "Tool server died during startup" >&2
        exit 1
    fi
    if [[ "${i}" -eq 120 ]]; then
        echo "Tool server did not start within 240 seconds" >&2
        exit 1
    fi
    sleep 2
done

python scripts/run_scene_baselines.py \
    --real \
    --scannet-root /scratch/kcwp264/data/megadepth/scannet_test_1500 \
    --scene-ids scene0707_00 scene0708_00 \
    --max-images 12 \
    --k 3 \
    --tool-server-url "http://127.0.0.1:${PORT}" \
    --output-dir "${OUT_DIR}" \
    2>&1 | tee "${LOG_DIR}/scene_bench_${SLURM_JOB_ID}.log"
