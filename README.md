# Agentic SfM

**RL-trained multimodal agents for hard 3D reconstruction.**

Train a vision-language model to orchestrate geometric tools — crop, match, doppelganger filtering, COLMAP — on difficult image pairs where direct matchers fail. Rewards are **verifiable** from ground-truth poses (MegaDepth / COLMAP), not human labels.

<p align="center">
  <img src="results/figures/result_clean_hero.png" alt="Agentic SfM qualitative results on MegaDepth" width="900"/>
</p>

---

## Motivation

Structure-from-motion pipelines break on **hard pairs**: low overlap, extreme viewpoint change, repeated textures, and doppelganger confusion. A frozen matcher cannot adapt its strategy per pair. We treat matching as a **multi-turn decision problem**: an MLLM chooses when to crop, which matcher to call, and when to stop — trained with **GRPO** on pose-accuracy rewards.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Qwen3.5-2B (policy, LoRA r=32)                                 │
│  multi-turn JSON tool calls                                     │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Tool Server (FastAPI)                                          │
│  crop · match (LoFTR/MASt3R/LightGlue) · doppelganger_check     │
│  retrieve · sfm_run · inspect                                   │
└───────────────────────────┬─────────────────────────────────────┘
                            │ correspondences + pose
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Verifiable reward                                              │
│  pose AUC @5°/10°/20° + inlier shaping − tool cost             │
└───────────────────────────┬─────────────────────────────────────┘
                            │ GRPO advantage
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Training: custom GRPO loop  OR  VeRL (volcengine/verl)         │
│  Rollout: vLLM  ·  Policy update: FSDP + LoRA                  │
└─────────────────────────────────────────────────────────────────┘
```

<p align="center">
  <img src="results/figures/agentic_sfm_phase1_architecture.png" alt="Phase 1 training architecture" width="700"/>
</p>

### GPU layout (3× NVIDIA L40S)

| GPU | Role |
|-----|------|
| 0 | vLLM rollout server |
| 1 | FSDP policy training (LoRA) |
| 2 | Matcher inference (tool server) |

## Results (included in repo)

All inference outputs live under [`results/`](results/). See [`results/README.md`](results/README.md) for details.

Trained LoRA adapters (pair SFT / Phase 1 GRPO / scene SFT) and figures:
[huggingface.co/Ryukijano/agentic-sfm-qwen35-2b](https://huggingface.co/Ryukijano/agentic-sfm-qwen35-2b)

### Phase 0 — zero-shot feasibility

Does stock Qwen3.5-2B cropping beat direct matching without RL?

<p align="center">
  <img src="results/figures/agentic_sfm_phase0_comparison.png" alt="Phase 0 comparison" width="600"/>
</p>

| Method | Mean inliers | Pose AUC |
|--------|-------------:|---------:|
| Direct LoFTR | 12.4 | 0.017 |
| Direct MASt3R | 12.4 | 0.017 |
| Zero-shot agent | 8.9 | 0.017 |

**Conclusion:** Zero-shot tool use is *worse* than direct matching on hard pairs → RL is required.

### Real MegaDepth inference (12 pairs)

Qualitative match visualizations on scenes `0015` and `0022`:

<p align="center">
  <img src="results/figures/result_clean_hero.png" alt="Match hero figure" width="800"/>
</p>

<p align="center">
  <img src="results/figures/result_matches_by_difficulty.png" alt="Matches by difficulty" width="700"/>
</p>

<p align="center">
  <img src="results/figures/result_matches_vs_overlap.png" alt="Matches vs overlap" width="500"/>
</p>

Per-pair overlays: [`results/phase0_real/visualizations/`](results/phase0_real/visualizations/)

### Phase 1 — S-GRPO training (running)

Best-match episodes logged to W&B during GRPO on real hard pairs — green inlier /
red outlier correspondences from the verifier, title reports full verified-inlier
count and episode reward. Mean reward climbed `0.374 → 0.496` over the first
30 logged steps (epoch-1 mean `0.388`).

| Step 10 — Brandenburg Gate (432 inliers, 85%) | Step 20 — Brandenburg Gate (540 inliers, 93%) | Step 30 — St. Peter's (295 inliers, 67%) |
|---|---|---|
| <img src="results/figures/train_best_match_step10.png"/> | <img src="results/figures/train_best_match_step20.png"/> | <img src="results/figures/train_best_match_step30.png"/> |

### Scene level — Phase 2 (pipeline built, training validating)

102 scene environments: 2 MegaDepth outdoor + 100 ScanNet indoor RGB-D scenes
(RGB + depth + intrinsics + poses, world-to-camera after c2w inversion).
Dense ground-truth point cloud (~200k points) rendered from ScanNet depth+pose
— shown as the **evaluation target**, not an agent output:

<p align="center">
  <img src="results/figures/scannet_gt_scene0772_00.png" alt="ScanNet dense GT cloud + camera trajectory" width="500"/>
</p>

Composite summary (hard pair, good pair, sparse COLMAP recon, ScanNet GT, reward curve):

<p align="center">
  <img src="results/figures/qualitative_final.png" alt="Composite qualitative figure" width="900"/>
</p>

## Project phases

| Phase | Status | Description |
|-------|--------|-------------|
| **0** | ✅ Done | Zero-shot crop feasibility + real MegaDepth eval (negative result → motivates RL) |
| **1** | 🟡 Running | S-GRPO on 1,535 hard pairs — reward climbing `0.374 → 0.496`, W&B media live |
| **2** | 🟡 Validating | Scene SFT done (102 scenes: MegaDepth + ScanNet); agent-matches→COLMAP importer; scene GRPO training validating |
| **3** | ⬜ Planned | Benchmark evaluation & write-up |

## Quick start

### Environment

```bash
conda create -n agentic-sfm python=3.11 -y
conda activate agentic-sfm
pip install -e ".[dev,rl]"
```

### 1. Start the tool server

```bash
python tools_server/server.py
# → http://localhost:8765/health
```

### 2. Phase 0 — zero-shot evaluation

```bash
python scripts/run_zeroshot.py --config configs/phase0_zeroshot.yaml
```

### 3. Phase 1 — SFT warmup then GRPO

Stock Qwen3.5-2B still needs **oracle crop traces** (heuristic boxes that beat full-frame inliers) so SFT can install the tool-call contract before GRPO.

```bash
# 1-GPU: LoFTR oracle crops → data/sft_train.jsonl
sbatch jobs/phase1_oracle_sft.slurm

# 1-GPU: LoRA SFT on those traces (Qwen/Qwen3.5-2B)
sbatch jobs/phase1_sft.slurm

# Point configs/phase1_grpo.yaml model.sft_adapter at outputs/sft/checkpoints/epoch_N
# then 3-GPU GRPO
sbatch jobs/phase1_grpo.slurm

# VeRL GRPO (experimental)
sbatch jobs/run_verl_grpo_3gpu.slurm
```

### Build the hard-pair dataset

```bash
python scripts/build_megadepth_pairs.py
# → data/hard_pairs_{train,val}.json  (398 train pairs, difficulty-stratified)
```

Dataset is not shipped in this repo (MegaDepth images are large). Distribution:

<p align="center">
  <img src="results/figures/agentic_sfm_hardpairs_distribution.png" alt="Hard pair difficulty distribution" width="500"/>
</p>

## Repository layout

```
agentic-sfm/
├── src/agentic_sfm/       # Core library
│   ├── agent/             # Policy, tool-call parsing
│   ├── tools/             # Tool client
│   ├── rewards/           # Pose AUC + shaping rewards
│   ├── data/              # Hard-pair dataset utilities
│   ├── rl/                # Custom GRPO / RCTP helpers
│   ├── verl_agent_loop.py # VeRL multi-turn agent loop
│   └── verl_reward.py     # VeRL reward hook
├── tools_server/          # FastAPI vision tool server
├── configs/               # Phase configs + VeRL Hydra overrides
├── scripts/               # Training & evaluation entry points
├── jobs/                  # Slurm launchers (AIRE L40S)
├── tests/
└── results/               # ✅ Bundled inference outputs & figures
```

## Key dependencies

- [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) — official 2B unified VLM / Instruct checkpoint (Apache 2.0). There is no `Qwen3.5-2B-Instruct` SKU.
- [verl](https://github.com/volcengine/verl) — distributed GRPO training
- [vLLM](https://github.com/vllm-project/vllm) — fast rollout sampling
- [MASt3R](https://github.com/naver/mast3r) / LoFTR — dense matchers
- [pycolmap](https://github.com/colmap/pycolmap) — COLMAP bindings

## Training details

- **Model:** `Qwen/Qwen3.5-2B` + LoRA (r=32, α=64; gated attention + Gated DeltaNet)
- **Algorithm:** GRPO (group size 8, clip-higher, dynamic sampling)
- **Reward:** pose AUC@{5°,10°,20°} + inlier shaping − per-tool cost
- **Curriculum:** easy/medium → hard → extreme overlap bins
- **Data:** 398 hard pairs from MegaDepth (medium 190, hard 189, extreme 11, easy 8)

## Citation

```bibtex
@misc{agentic-sfm2025,
  title  = {Agentic SfM: RL-Trained MLLM Tool Orchestration for Hard 3D Reconstruction},
  author = {Ryukijano},
  year   = {2025},
  url    = {https://github.com/Ryukijano/agentic-sfm}
}
```

## License

MIT — see [LICENSE](LICENSE).
