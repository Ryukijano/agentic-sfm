#!/usr/bin/env bash
# Source from Slurm jobs and interactive shells before vLLM / torch GPU work.
#
#   source /scratch/kcwp264/.aire_scratch_env.sh
#   source /scratch/kcwp264/agentic-sfm/scripts/agentic_sfm_env.sh
#
# vLLM 0.28+cu129 bundles CUDA 12.9 user-space libs (libcudart.so.12 from
# nvidia-cuda-runtime-cu12==12.9.x). AIRE driver 560.35.03 reports max CUDA 12.6,
# but CUDA 12.x minor-version compatibility runs 12.9 user-space on a 12.6
# driver when the app uses the bundled pip runtime (which this script puts on
# LD_LIBRARY_PATH). Do NOT install vllm cu130 — CUDA 13 needs driver >= 580.

: "${AGENTIC_SFM_ROOT:=/scratch/kcwp264/agentic-sfm}"
: "${CONDA_ENV:=/scratch/kcwp264/.conda_envs/agentic-sfm}"

export PATH="${CONDA_ENV}/bin:/users/kcwp264/.local/bin:${PATH}"
export CONDA_PREFIX="${CONDA_ENV}"
export PYTHONPATH="${AGENTIC_SFM_ROOT}/src:${PYTHONPATH:-}"

# Prefer pip-bundled CUDA libs over (missing) system libcudart.so.13.
if [[ -d "${CONDA_ENV}/lib/python3.11/site-packages/nvidia" ]]; then
  _nv_libs=()
  while IFS= read -r -d '' libdir; do
    _nv_libs+=("${libdir}")
  done < <(find "${CONDA_ENV}/lib/python3.11/site-packages/nvidia" -type d -name lib -print0 2>/dev/null)
  if ((${#_nv_libs[@]})); then
    export LD_LIBRARY_PATH="$(IFS=:; echo "${_nv_libs[*]}"):${LD_LIBRARY_PATH:-}"
  fi
fi
