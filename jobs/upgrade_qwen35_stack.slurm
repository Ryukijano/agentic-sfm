#!/bin/bash
#SBATCH --job-name=asfm-vllm-env
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=85G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/kcwp264/logs/agentic-sfm/upgrade_qwen35_%j.out
#SBATCH --error=/scratch/kcwp264/logs/agentic-sfm/upgrade_qwen35_%j.err

# In-place pip/uv upgrade of the existing conda env for the policy VLM:
# transformers>=5.2 (qwen3_5) + vllm>=0.17.
# Do not create a separate .venv — Slurm jobs already use this prefix.

set -euo pipefail

echo "=== Upgrade agentic-sfm conda env for Qwen3.5-2B ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start: $(date)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

PROJECT_ROOT="/scratch/kcwp264/agentic-sfm"
CONDA_ENV="/scratch/kcwp264/.conda_envs/agentic-sfm"
RECORD_DIR="${PROJECT_ROOT}/outputs/env-record-${SLURM_JOB_ID}"
mkdir -p "${RECORD_DIR}" /scratch/kcwp264/logs/agentic-sfm

source /scratch/kcwp264/.aire_scratch_env.sh
source "${PROJECT_ROOT}/scripts/agentic_sfm_env.sh"
module load cuda/12.4.1 2>/dev/null || true
export CUDA_HOME="${CUDA_HOME:-/opt/apps/pkg/compilers/cuda/12.4.1}"
export UV_PYTHON="${CONDA_ENV}/bin/python"
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-8}"
export CPATH="${CUDA_HOME}/include${CPATH:+:${CPATH}}"

cd "${PROJECT_ROOT}"
python -V
python -m pip freeze > "${RECORD_DIR}/pip-freeze-before.txt"

echo "=== Ensure policy stack (transformers>=5, vllm>=0.17, peft) ==="
if ! python -c "import vllm" 2>/dev/null; then
  uv pip install --python "${UV_PYTHON}" --torch-backend=auto \
      "vllm>=0.17" \
      "transformers>=5.2" \
      "peft>=0.13" \
      "accelerate>=1.0"
else
  echo "vllm already installed; skipping pip upgrade"
fi

echo "=== post-upgrade versions ==="
python - <<'PY'
import sys
import torch
import transformers
import vllm
import peft

print("python", sys.version.split()[0])
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
print("transformers", transformers.__version__)
print("vllm", vllm.__version__)
print("peft", peft.__version__)

import transformers.models as m
has = hasattr(m, "qwen3_5") or "qwen3_5" in dir(m)
print("has_qwen3_5", has)
if not has:
    raise SystemExit("transformers still missing qwen3_5")

from packaging.version import Version
if Version(transformers.__version__) < Version("5.2"):
    raise SystemExit(f"transformers too old: {transformers.__version__}")
if Version(vllm.__version__.split("+")[0]) < Version("0.17"):
    raise SystemExit(f"vllm too old: {vllm.__version__}")

from agentic_sfm.constants import assert_qwen35_runtime
assert_qwen35_runtime()
print("assert_qwen35_runtime: ok")

from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("Qwen/Qwen3.5-2B", trust_remote_code=True)
print("config_type", type(cfg).__name__, "model_type", getattr(cfg, "model_type", None))

# Smoke: vLLM can construct engine args (does not start server).
from vllm import EngineArgs
EngineArgs(model="Qwen/Qwen3.5-2B", trust_remote_code=True, dtype="bfloat16")
print("vllm EngineArgs: ok")
PY

python -m pip freeze > "${RECORD_DIR}/pip-freeze-after.txt"
echo "Freeze saved under ${RECORD_DIR}"
echo "End: $(date)"
seff "${SLURM_JOB_ID}" || true
