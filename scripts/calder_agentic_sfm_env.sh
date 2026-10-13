#!/usr/bin/env bash
# Calder counterpart to scripts/agentic_sfm_env.sh — source this on Leeds
# Calder (H200, sm_90) interactive shells and Slurm job scripts before
# vLLM / torch GPU work.
#
#   source "${AGENTIC_SFM_ROOT:-$HOME/agentic-sfm}/scripts/calder_agentic_sfm_env.sh"
#
# Calder defines NO $SCRATCH / $TMP_SHARED. We use:
#   CALDER_SCRATCH=/scratch-calder/users/$USER
#   CALDER_FLASH=/flash-calder/users/$USER
#
# vLLM 0.28+cu129 bundles CUDA 12.9 user-space libs; prefer pip-bundled
# nvidia libs via LD_LIBRARY_PATH (same trick as the Aire env script).

# --- Bootstrap: prefer the shared calder-setup env if present ------------
# $HOME is shared between Aire and Calder, so this file exists on both.
if [[ -f /users/kcwp264/calder-setup/bootstrap/calder_env.sh ]]; then
  # shellcheck disable=SC1091
  source /users/kcwp264/calder-setup/bootstrap/calder_env.sh
else
  export MODULEPATH=/uol/apps/calder/modules
  export CALDER_SCRATCH="${CALDER_SCRATCH:-/scratch-calder/users/$USER}"
  export CALDER_FLASH="${CALDER_FLASH:-/flash-calder/users/$USER}"
  export HF_HOME="$CALDER_SCRATCH/.cache/huggingface"
  export TORCH_HOME="$CALDER_SCRATCH/.cache/torch"
  export UV_CACHE_DIR="$CALDER_SCRATCH/.cache/uv"
  export PIP_CACHE_DIR="$CALDER_SCRATCH/.cache/pip"
  export WANDB_DIR="$CALDER_SCRATCH/.cache/wandb"
  export WANDB_CACHE_DIR="$CALDER_SCRATCH/.cache/wandb"
  export TRITON_CACHE_DIR="$CALDER_SCRATCH/.cache/triton"
fi

# --- Calder modules (best-effort; fine to run without a module system) ---
if command -v module >/dev/null 2>&1; then
  module load calder/hopper 2>/dev/null || true
  module load miniforge3/26.1.1-3 2>/dev/null || true
  module load cuda 2>/dev/null || true
fi

# --- Project root = parent dir of this script ----------------------------
_env_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
export AGENTIC_SFM_ROOT="${AGENTIC_SFM_ROOT:-$(dirname "${_env_script_dir}")}"

# --- Conda env resolution -------------------------------------------------
if [[ -z "${CONDA_ENV:-}" ]]; then
  for _cand in \
    "$HOME/miniforge3/envs/agentic-sfm" \
    "$CALDER_SCRATCH/.conda_envs/agentic-sfm" \
    "/scratch/kcwp264/.conda_envs/agentic-sfm"; do
    if [[ -x "${_cand}/bin/python" ]]; then
      CONDA_ENV="${_cand}"
      break
    fi
  done
fi
if [[ -n "${CONDA_ENV:-}" && -d "${CONDA_ENV}/bin" ]]; then
  export PATH="${CONDA_ENV}/bin:${PATH}"
  export CONDA_PREFIX="${CONDA_ENV}"
fi

# --- PYTHONPATH: project src + MASt3R / dust3r ----------------------------
_mast3r_repo="${MAST3R_REPO:-$CALDER_SCRATCH/src/mast3r}"
if [[ ! -d "${_mast3r_repo}" && -d /scratch/kcwp264/src/mast3r ]]; then
  _mast3r_repo=/scratch/kcwp264/src/mast3r   # Aire fallback (testing)
fi
export PYTHONPATH="${AGENTIC_SFM_ROOT}/src:${_mast3r_repo}:${_mast3r_repo}/dust3r:${PYTHONPATH:-}"

# vLLM 0.28 ships flashinfer-python with flashinfer-cubin; skip version check.
export FLASHINFER_DISABLE_VERSION_CHECK=1

# --- Prefer pip-bundled nvidia CUDA libs (same as Aire env script) --------
if [[ -n "${CONDA_ENV:-}" && -d "${CONDA_ENV}/lib/python3.11/site-packages/nvidia" ]]; then
  _nv_libs=()
  while IFS= read -r -d '' libdir; do
    _nv_libs+=("${libdir}")
  done < <(find "${CONDA_ENV}/lib/python3.11/site-packages/nvidia" -type d -name lib -print0 2>/dev/null)
  if ((${#_nv_libs[@]})); then
    export LD_LIBRARY_PATH="$(IFS=:; echo "${_nv_libs[*]}"):${LD_LIBRARY_PATH:-}"
  fi
fi

find_free_port() {
  local start_port="${1:-30000}"
  python3 -c "
import socket, sys
start = int(sys.argv[1])
for port in range(start, start + 3000):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('', port))
            print(port)
            sys.exit(0)
    except OSError:
        continue
sys.exit(1)
" "${start_port}"
}
export -f find_free_port 2>/dev/null || true
