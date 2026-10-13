# Calder 8xH200 Migration & CVPR/ICCV Runbook

One-time migration of `agentic-sfm` from Aire (3xL40S) to Leeds Calder
(8xH200 NVL, 141 GB/GPU), plus the experiment matrix for CVPR 2027
(deadline Nov 16) / ICCV 2027.

The Calder scratch quota is **200 GB**. The job
(`jobs/calder_8xh200_hybrid.slurm`) stages **all** transient work —
COLMAP outputs (`SFM_OUTPUT_ROOT`), LoRA live-reload staging
(`LORA_STAGING_DIR`), Triton/FlashInfer/vLLM caches — on `$CALDER_FLASH`
(node-local NVMe, `/dev/shm` fallback), so **zero transient bytes** hit
scratch.

## 0. Storage budget

| Item | Size | Where |
|---|---|---|
| Slim staging bundle (`calder_stage_agentic_sfm`) | **~2.3 GB** | rsync'd to `$CALDER_SCRATCH` |
| — scene SFT adapter (epoch_5) + GRPO latest | ~0.5 GB | bundle |
| — OPSFT direction artifacts (topk10 + exact_support) | ~0.1 GB | bundle |
| — MegaDepth scenes 0015/0022 + scene_info | ~1.2 GB | bundle |
| — ScanNet 40 train scenes (scene-disjoint) + manifest | ~0.5 GB | bundle |
| — SHA256SUMS | <1 MB | bundle |
| Repo clone (code only, via git) | ~0.1 GB | `$CALDER_SCRATCH` |
| conda env `agentic-sfm` (torch, vLLM sm_90, pycolmap) | ~9 GB | `$CALDER_SCRATCH/.conda` |
| HF cache (Qwen3.5-2B + matcher weights) | ~14 GB | `$CALDER_SCRATCH/.cache/huggingface` |
| Misc caches (triton, wandb, xdg) | ~6 GB | `$CALDER_SCRATCH/.cache` |
| **Total Calder footprint** | **~31 GB** | leaves **~169 GB free** of 200 GB |

Transient per-job usage (COLMAP databases, sparse reconstructions, LoRA
staging) lives on `$CALDER_FLASH` / `/dev/shm` and is deleted at job end —
it does not count against the 200 GB.

## 1. On Aire — build the staging bundle

```bash
cd /scratch/kcwp264/agentic-sfm
bash scripts/stage_for_calder.sh
# Creates /scratch/kcwp264/calder_stage_agentic_sfm containing:
#   checkpoints/   outputs/opsft/direction_scene_grpo_{topk,exact_support}.pt
#   data/megadepth/{scene_info_full,megadepth_test_1500,scannet_test_1500}  (slim)
#   SHA256SUMS
sha256sum -c /scratch/kcwp264/calder_stage_agentic_sfm/SHA256SUMS   # optional sanity
```

## 2. Transfer Aire -> Calder (Leeds-internal, gated-portal / direct-only)

Calder transfers must go over the Leeds-internal SSH path; no public
endpoints. Use a ControlMaster so the multi-hour rsync reuses one
authenticated connection:

```bash
# ~/.ssh/config
# Host calder
#   HostName calder.leeds.ac.uk
#   User <you>
#   ControlMaster auto
#   ControlPath ~/.ssh/cm-%r@%h:%p
#   ControlPersist 8h

ssh -fNM calder                 # open master once (2FA happens here)
rsync -avP --partial --inplace \
    /scratch/kcwp264/calder_stage_agentic_sfm/ \
    calder:$CALDER_SCRATCH/calder_stage_agentic_sfm/
# Re-run the same rsync to resume; --partial --inplace is resumable.
```

## 3. On Calder login node — env + unpack

```bash
cd $CALDER_SCRATCH
git clone <repo> agentic-sfm && cd agentic-sfm

# Conda env — NOTE: build vLLM from the sm_90 wheel (Hopper), not the
# sm_89/Ada wheel used on Aire L40S:
#   pip install vllm --extra-index-url <wheelhouse with sm_90 build>
# then smoke: python -c "import vllm; print(vllm.__version__)"
conda env create -f environment.yaml -n agentic-sfm   # or: mamba
conda activate agentic-sfm
python -c "import torch; print(torch.cuda.get_device_capability())"  # -> (9,0)

bash unpack_on_calder.sh
# Verifies SHA256SUMS, unpacks the bundle into $CALDER_SCRATCH, and
# symlinks  data/megadepth -> $CALDER_SCRATCH/data/megadepth  so the
# config's relative paths (data/megadepth/...) resolve.
```

## 4. Submit + monitor

```bash
sbatch jobs/calder_8xh200_hybrid.slurm
squeue -j <JOBID>
tail -f logs/agentic-sfm/calder_8xh200_<JOBID>.out
tail -f logs/agentic-sfm/phase2_calder_train_<JOBID>.log
tail -f logs/agentic-sfm/calder_opsft_topk_<JOBID>.log
tail -f logs/agentic-sfm/calder_opsft_exact_support_<JOBID>.log

# All 8 GPUs at a glance:
srun --jobid <JOBID> --pty nvidia-smi
# 0,1 vLLM TP=2 | 2 tool server | 3 train LoRA | 4 frozen KL | 5,6 OPSFT arms | 7 eval
```

Outputs land in `outputs/phase2_calder/run_<JOBID>/checkpoints/` (GRPO),
`outputs/scene_opsft_topk10/`, `outputs/scene_opsft_exact/` (ablations),
and `run_<JOBID>/frozen_eval/` + `bench_pose_priors/` (end-of-job eval on
GPU 7 against `data/frozen_eval_manifest.json`).

## 5. CVPR 2027 (Nov 16) / ICCV 2027 experiment matrix

Each row = one `sbatch` of `calder_8xh200_hybrid.slurm` (12 h wall) or a
config clone with the noted overrides.

| # | Experiment | Overrides | Output |
|---|---|---|---|
| E1 | **Multi-scene Hybrid GRPO** (this job) | defaults | `outputs/phase2_calder/run_*/` |
| E2 | OPSFT ablation arms (bundled in E1) | — | `scene_opsft_{topk10,exact}` |
| E3 | Matcher ablation: LoFTR-only / MASt3R-only / VGGT-only | `tools.matchers: [X]` | 3 runs |
| E4 | Group-size scaling: G=4 vs G=6 vs G=8 | `rl.group_size` | GRPO credit-assignment curves |
| E5 | Context scaling: 8k vs 16k vs 24k | `--max-model-len` | truncation count vs score |
| E6 | KL ablation: `kl_coef` {0, 0.03, 0.1} | `training.kl_coef` | leash sensitivity |
| E7 | Curriculum ablation: no curriculum vs 3-stage | `curriculum.enabled` | sample-efficiency |
| E8 | Frozen-manifest benchmark of best ckpt + all matcher priors | end-of-job eval (GPU 7) | Table 1 numbers |
| E9 | Generalization: held-out ScanNet `val_scene_ids` eval | `--scene-ids scene077{0..7}_00` | indoor generalization |
| E10 | OPSFT vs GRPO head-to-head on frozen manifest | `eval_scene.py --lora-checkpoint` | constrained-vs-RL claim |

Suggested order for CVPR: E1+E2 (this job) -> E8/E9 (numbers) -> E3, E6
(ablations) -> E4, E5, E7 (analysis) -> E10 (OPSFT comparison). ICCV
extends with more scenes once the data pipeline supports >20 images.
