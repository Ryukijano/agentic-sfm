#!/usr/bin/env bash
# stage_for_calder.sh — run on AIRE to assemble the agentic-sfm staging
# bundle (~2.3 GB) for transfer to Leeds Calder, inside a 200 GB budget.
#
#   scripts/stage_for_calder.sh [--dry-run] [--stage-dir DIR]
#
# What it stages into ${STAGE_DIR}:
#   data_megadepth_slim/   MegaDepth+ScanNet subsets (gated-portal: moves
#                          Leeds-internal rsync only, NEVER to HF)
#   checkpoints/           SFT warmstart + phase2_hybrid best_eval/epoch_50
#   opsft_directions/      pre-extracted OPSFT update-direction artifacts
#   unpack_on_calder.sh    Calder-side unpacker (checksums, layout, HF fetch)
#   SHA256SUMS             integrity manifest over every staged file
#
# Then it prints the rsync command to push ${STAGE_DIR} to calder-login.

set -euo pipefail

AGENTIC_SFM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/scratch/kcwp264/data/megadepth}"
STAGE_DIR="${STAGE_DIR:-/scratch/kcwp264/calder_stage_agentic_sfm}"
PY="${AGENTIC_SFM_PY:-/scratch/kcwp264/.conda_envs/agentic-sfm/bin/python}"
DRY_RUN=0

while (($#)); do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --stage-dir) STAGE_DIR="$2"; shift 2 ;;
    -h|--help) sed -n '1,20p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

say()  { printf '[stage] %s\n' "$*"; }
size() { du -sh "$1" 2>/dev/null | cut -f1 || echo "?"; }

# --- What will be staged -------------------------------------------------
declare -a RSYNC_SPECS=(
  # src (trailing / = contents) : dest subdir under data_megadepth_slim
  "${DATA_ROOT}/megadepth_test_1500:megadepth_test_1500"
  "${DATA_ROOT}/scannet_test_1500:scannet_test_1500"
  "${DATA_ROOT}/scene_info:scene_info"
)
SCENE_INFO_FULL_SRC="${DATA_ROOT}/scene_info_full/scene_info"
SCENE_INFO_FULL_DST="scene_info_full/scene_info"

declare -a CKPT_SPECS=(
  "outputs/scene_sft_graph_retry2/checkpoints/epoch_5:scene_sft_graph_retry2_epoch_5"
  "outputs/phase2_hybrid/run_8237240/checkpoints/best_eval:phase2_hybrid_run_8237240_best_eval"
  "outputs/phase2_hybrid/run_8237240/checkpoints/epoch_50:phase2_hybrid_run_8237240_epoch_50"
  "outputs/bench_pose_priors/run_8235219/bench_pose_priors.json:bench_pose_priors_run_8235219.json"
)

say "Staging root : ${STAGE_DIR}"
say "Data root    : ${DATA_ROOT}"
say "Dry run      : ${DRY_RUN}"
echo

say "Planned contents:"
for spec in "${RSYNC_SPECS[@]}"; do
  src="${spec%%:*}"; dst="${spec##*:}"
  printf '  data_megadepth_slim/%-28s <- %s (%s)\n' "${dst}" "${src}" "$(size "${src}")"
done
printf '  data_megadepth_slim/%-28s <- %s/0015*.npz,0022*.npz (%s)\n' \
  "${SCENE_INFO_FULL_DST}" "${SCENE_INFO_FULL_SRC}" \
  "$(du -ch ${SCENE_INFO_FULL_SRC}/0015*.npz ${SCENE_INFO_FULL_SRC}/0022*.npz 2>/dev/null | tail -1 | cut -f1 || echo '?')"
for spec in "${CKPT_SPECS[@]}"; do
  src="${spec%%:*}"; dst="${spec##*:}"
  printf '  checkpoints/%-42s <- %s (%s)\n' "${dst}" "${AGENTIC_SFM_ROOT}/${src}" "$(size "${AGENTIC_SFM_ROOT}/${src}")"
done
printf '  opsft_directions/  (3 artifacts via extract_update_direction.py)\n'
printf '  unpack_on_calder.sh, SHA256SUMS\n'
echo

if ((DRY_RUN)); then
  say "Dry run only — nothing copied."
  exit 0
fi

mkdir -p "${STAGE_DIR}/data_megadepth_slim" "${STAGE_DIR}/checkpoints" "${STAGE_DIR}/opsft_directions"

# --- 1. data_megadepth_slim ----------------------------------------------
say "Syncing data_megadepth_slim ..."
for spec in "${RSYNC_SPECS[@]}"; do
  src="${spec%%:*}"; dst="${spec##*:}"
  mkdir -p "${STAGE_DIR}/data_megadepth_slim/${dst}"
  rsync -a --info=progress2 "${src}/" "${STAGE_DIR}/data_megadepth_slim/${dst}/"
done
# Only the two MegaDepth eval scenes' .npz files — skips 625 unused scenes
# and the .tar.gz/.zip archives (~39 GB saved).
mkdir -p "${STAGE_DIR}/data_megadepth_slim/${SCENE_INFO_FULL_DST}"
rsync -a --info=progress2 \
  --include='0015*.npz' --include='0022*.npz' --exclude='*' \
  "${SCENE_INFO_FULL_SRC}/" "${STAGE_DIR}/data_megadepth_slim/${SCENE_INFO_FULL_DST}/"

# --- 2. checkpoints -------------------------------------------------------
say "Syncing checkpoints ..."
for spec in "${CKPT_SPECS[@]}"; do
  src="${spec%%:*}"; dst="${spec##*:}"
  src_path="${AGENTIC_SFM_ROOT}/${src}"
  dst_path="${STAGE_DIR}/checkpoints/${dst}"
  if [[ -d "${src_path}" ]]; then
    mkdir -p "${dst_path}"
    rsync -a --info=progress2 "${src_path}/" "${dst_path}/"
  elif [[ -f "${src_path}" ]]; then
    rsync -a --info=progress2 "${src_path}" "${dst_path}"
  else
    echo "WARNING: missing checkpoint source ${src_path}" >&2
  fi
done

# --- 3. opsft_directions --------------------------------------------------
say "Extracting OPSFT update directions ..."
EXT="${AGENTIC_SFM_ROOT}/scripts/extract_update_direction.py"
BASE="${AGENTIC_SFM_ROOT}/outputs/scene_sft_graph_retry2/checkpoints/epoch_5"
HYB="${AGENTIC_SFM_ROOT}/outputs/phase2_hybrid/run_8237240/checkpoints"
OUT="${STAGE_DIR}/opsft_directions"
"${PY}" "${EXT}" --base "${BASE}" --trained "${HYB}/best_eval" \
  --selection topk --topk-frac 0.1 \
  --output "${OUT}/direction_8237240_best_eval_topk10.pt"
"${PY}" "${EXT}" --base "${BASE}" --trained "${HYB}/best_eval" \
  --selection exact_support \
  --output "${OUT}/direction_8237240_best_eval_exact.pt"
"${PY}" "${EXT}" --base "${BASE}" --trained "${HYB}/epoch_50" \
  --selection topk --topk-frac 0.1 \
  --output "${OUT}/direction_8237240_epoch50_topk10.pt"

# --- 4. unpack_on_calder.sh -----------------------------------------------
say "Writing unpack_on_calder.sh ..."
cat > "${STAGE_DIR}/unpack_on_calder.sh" <<'UNPACK_EOF'
#!/usr/bin/env bash
# unpack_on_calder.sh — run on a CALDER login node after rsyncing this
# staging directory across. Verifies checksums, lays out the slim
# MegaDepth/ScanNet tree on $CALDER_SCRATCH, restores checkpoints and OPSFT
# directions into the repo's outputs/, rewrites dataset paths (except the
# hash-verified frozen_eval_manifest.json), clones MASt3R, and prefetches
# public HF weights.
#
#   bash unpack_on_calder.sh
set -euo pipefail

STAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Calder env (paths + caches); falls back to explicit defaults.
if [[ -f "${AGENTIC_SFM_ROOT:-/nonexistent}/scripts/calder_agentic_sfm_env.sh" ]]; then
  :
fi
export CALDER_SCRATCH="${CALDER_SCRATCH:-/scratch-calder/users/$USER}"
export CALDER_FLASH="${CALDER_FLASH:-/flash-calder/users/$USER}"
export HF_HOME="${HF_HOME:-$CALDER_SCRATCH/.cache/huggingface}"

say() { printf '[unpack] %s\n' "$*"; }

# --- Locate repo root on Calder ------------------------------------------
AGENTIC_SFM_ROOT="${AGENTIC_SFM_ROOT:-}"
if [[ -z "${AGENTIC_SFM_ROOT}" ]]; then
  for cand in "${HOME}/agentic-sfm" "${CALDER_SCRATCH}/agentic-sfm" "${CALDER_SCRATCH}/repos/agentic-sfm"; do
    [[ -d "${cand}/src" ]] && AGENTIC_SFM_ROOT="${cand}" && break
  done
fi
[[ -n "${AGENTIC_SFM_ROOT}" ]] || { echo "Set AGENTIC_SFM_ROOT to the repo checkout on Calder" >&2; exit 1; }
say "Repo root    : ${AGENTIC_SFM_ROOT}"
say "Stage dir    : ${STAGE_DIR}"
say "Calder scratch: ${CALDER_SCRATCH}"

# --- 1. Verify checksums --------------------------------------------------
say "Verifying SHA256SUMS ..."
( cd "${STAGE_DIR}" && sha256sum -c SHA256SUMS )

# --- 2. Data layout --------------------------------------------------------
DATA_DST="${CALDER_SCRATCH}/data/megadepth"
mkdir -p "${DATA_DST}"
say "Installing data_megadepth_slim -> ${DATA_DST}"
rsync -a "${STAGE_DIR}/data_megadepth_slim/" "${DATA_DST}/"
mkdir -p "${AGENTIC_SFM_ROOT}/data"
ln -sfn "${DATA_DST}" "${AGENTIC_SFM_ROOT}/data/megadepth"
say "Symlinked ${AGENTIC_SFM_ROOT}/data/megadepth -> ${DATA_DST}"

# --- 3. Checkpoints + OPSFT directions ------------------------------------
say "Restoring checkpoints into outputs/ ..."
mkdir -p "${AGENTIC_SFM_ROOT}/outputs/scene_sft_graph_retry2/checkpoints" \
         "${AGENTIC_SFM_ROOT}/outputs/phase2_hybrid/run_8237240/checkpoints" \
         "${AGENTIC_SFM_ROOT}/outputs/bench_pose_priors/run_8235219" \
         "${AGENTIC_SFM_ROOT}/outputs/opsft"
rsync -a "${STAGE_DIR}/checkpoints/scene_sft_graph_retry2_epoch_5/" \
         "${AGENTIC_SFM_ROOT}/outputs/scene_sft_graph_retry2/checkpoints/epoch_5/"
rsync -a "${STAGE_DIR}/checkpoints/phase2_hybrid_run_8237240_best_eval/" \
         "${AGENTIC_SFM_ROOT}/outputs/phase2_hybrid/run_8237240/checkpoints/best_eval/"
rsync -a "${STAGE_DIR}/checkpoints/phase2_hybrid_run_8237240_epoch_50/" \
         "${AGENTIC_SFM_ROOT}/outputs/phase2_hybrid/run_8237240/checkpoints/epoch_50/"
rsync -a "${STAGE_DIR}/checkpoints/bench_pose_priors_run_8235219.json" \
         "${AGENTIC_SFM_ROOT}/outputs/bench_pose_priors/run_8235219/bench_pose_priors.json"
ln -sfn run_8237240 "${AGENTIC_SFM_ROOT}/outputs/phase2_hybrid/latest"
rsync -a "${STAGE_DIR}/opsft_directions/" "${AGENTIC_SFM_ROOT}/outputs/opsft/"
ln -sfn direction_8237240_best_eval_topk10.pt \
        "${AGENTIC_SFM_ROOT}/outputs/opsft/direction_scene_grpo.pt"

# --- 4. Rewrite dataset paths ----------------------------------------------
# CRITICAL: never touch data/frozen_eval_manifest.json — its SHA256
# canonical hash is verified at runtime by bench_pose_priors.py; callers
# pass --scannet-root on the CLI instead.
say "Rewriting /scratch/kcwp264/data/megadepth -> ${DATA_DST} in data/*.json{,l} ..."
python3 - "${AGENTIC_SFM_ROOT}" "${DATA_DST}" <<'PY'
import pathlib, sys
root = pathlib.Path(sys.argv[1]) / "data"
new = sys.argv[2]
old = "/scratch/kcwp264/data/megadepth"
for p in sorted(list(root.glob("*.json")) + list(root.glob("*.jsonl"))):
    if p.name == "frozen_eval_manifest.json":
        print(f"  SKIP {p.name} (hash-verified)")
        continue
    txt = p.read_text()
    if old in txt:
        p.with_suffix(p.suffix + ".bak").write_text(txt)
        p.write_text(txt.replace(old, new))
        print(f"  rewrote {p.name} (.bak kept)")
PY

# --- 5. MASt3R --------------------------------------------------------------
MAST3R_REPO="${MAST3R_REPO:-$CALDER_SCRATCH/src/mast3r}"
if [[ ! -d "${MAST3R_REPO}/.git" ]]; then
  say "Cloning MASt3R -> ${MAST3R_REPO}"
  mkdir -p "$(dirname "${MAST3R_REPO}")"
  git clone --recurse-submodules https://github.com/naver/mast3r.git "${MAST3R_REPO}"
else
  say "MASt3R already present at ${MAST3R_REPO}"
fi

# --- 6. Prefetch public HF weights ------------------------------------------
say "Prefetching HF weights into ${HF_HOME} ..."
python3 - <<'PY'
from huggingface_hub import snapshot_download
for repo in [
    "Qwen/Qwen3.5-2B",
    "facebook/VGGT-1B",
    "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
    "facebook/dinov2-small",
]:
    print(f"  snapshot_download({repo})")
    snapshot_download(repo)
PY

say "Done. Source ${AGENTIC_SFM_ROOT}/scripts/calder_agentic_sfm_env.sh before GPU work."
UNPACK_EOF
chmod +x "${STAGE_DIR}/unpack_on_calder.sh"

# --- 5. SHA256SUMS ----------------------------------------------------------
say "Computing SHA256SUMS ..."
( cd "${STAGE_DIR}" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS )

# --- Summary ----------------------------------------------------------------
echo
say "=== Staging complete ==="
du -sh "${STAGE_DIR}"
du -sh "${STAGE_DIR}"/*/ 2>/dev/null || true
echo
cat <<EOF
Copy to Calder (ControlMaster = one password for all connections):

  ssh -o ControlMaster=auto -o ControlPersist=10m \\
      -o ControlPath=~/.ssh/cm-%r@%h:%p ${USER}@calder-login

  rsync -a --info=progress2 --partial \\
      -e 'ssh -o ControlMaster=auto -o ControlPersist=10m -o ControlPath=~/.ssh/cm-%r@%h:%p' \\
      ${STAGE_DIR}/ ${USER}@calder-login:/scratch-calder/users/${USER}/calder_stage_agentic_sfm/

Then on Calder:
  cd /scratch-calder/users/${USER}/calder_stage_agentic_sfm
  AGENTIC_SFM_ROOT=<repo checkout> bash unpack_on_calder.sh

NOTE: MegaDepth/ScanNet are gated-portal — this bundle must only ever move
over Leeds-internal channels (aire.leeds.ac.uk <-> calder-login). NEVER
upload any part of it to Hugging Face, public or private.
EOF
