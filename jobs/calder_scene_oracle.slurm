#!/bin/bash
# =============================================================================
# SLURM: scene-level oracle study on Calder (1x H200 per array task).
# One array task per matcher: 0=loftr 1=mast3r 2=vggt. No agent, no vLLM:
# tool server + scripts/run_scene_oracle.py on the 30 frozen ScanNet scenes.
#
# Submit from the repo root:  sbatch jobs/calder_scene_oracle.slurm
# Then:  python scripts/analyze_scene_oracle.py outputs/scene_oracle
# =============================================================================
#SBATCH --job-name=asfm-scene-oracle
#SBATCH --partition=calder-gpu-hopper
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/agentic-sfm/scene_oracle_%A_%a.out
#SBATCH --error=logs/agentic-sfm/scene_oracle_%A_%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?submit from the repo root}}"
[[ -f "${PROJECT_ROOT}/scripts/calder_agentic_sfm_env.sh" ]] || { echo "ERROR: ${PROJECT_ROOT} is not the agentic-sfm repo root"; exit 1; }
MATCHERS=(loftr mast3r vggt)
MATCHER="${MATCHERS[${SLURM_ARRAY_TASK_ID}]}"
SCANNET_ROOT="${SCANNET_ROOT:-/scratch-calder/users/${USER}/data/megadepth/scannet_test_1500}"
OUT_DIR="${PROJECT_ROOT}/outputs/scene_oracle"
mkdir -p "${PROJECT_ROOT}/logs/agentic-sfm" "${OUT_DIR}"

echo "=== Scene oracle study: matcher=${MATCHER} ==="
echo "Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}  Node: ${SLURM_NODELIST}  Start: $(date)"

source "${PROJECT_ROOT}/scripts/calder_agentic_sfm_env.sh"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
# Calder has no TMP_SHARED; COLMAP workdirs go to scratch, not ./outputs.
export SFM_OUTPUT_ROOT="${CALDER_SCRATCH}/sfm_runs/oracle_${SLURM_ARRAY_JOB_ID}_${MATCHER}"
mkdir -p "${SFM_OUTPUT_ROOT}"
cd "${PROJECT_ROOT}"

PORT=$((20000 + (SLURM_ARRAY_JOB_ID + SLURM_ARRAY_TASK_ID) % 20000))
CUDA_VISIBLE_DEVICES=0 python tools_server/server.py --host 127.0.0.1 --port "${PORT}" &
TOOL_PID=$!
cleanup() { kill "${TOOL_PID}" 2>/dev/null || true; wait "${TOOL_PID}" 2>/dev/null || true; }
trap cleanup EXIT

for i in $(seq 1 120); do
    curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && break
    kill -0 "${TOOL_PID}" 2>/dev/null || { echo "Tool server died during startup" >&2; exit 1; }
    [[ "${i}" -eq 120 ]] && { echo "Tool server did not start within 240 seconds" >&2; exit 1; }
    sleep 2
done

python scripts/run_scene_oracle.py \
    --matcher "${MATCHER}" \
    --scannet-root "${SCANNET_ROOT}" \
    --output-dir "${OUT_DIR}" \
    --sfm-output-base "${SFM_OUTPUT_ROOT}" \
    --tool-server-url "http://127.0.0.1:${PORT}"

echo "End: $(date)"
