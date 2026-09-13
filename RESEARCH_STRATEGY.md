# Agentic SfM — State-of-the-Art Review & Strategic Plan

*Compiled from arXiv, CVPR/ICCV/ICML/ACL/NeurIPS 2025-2026 proceedings, project pages, and
primary GitHub repositories. X/LinkedIn used only as supplementary signal.*

---

## 1. The Landscape (mid-2025 → late-2026)

### 1.1 RL for VLMs: the core methods

| Method | Venue | Key idea | Relevance to us |
|--------|-------|----------|-----------------|
| **Visual-ARFT** | ICCV 2025 | GRPO + composite verifiable rewards (format + accuracy + tool executability) on Qwen2.5-VL. Works with as few as 20 annotated samples. | **Our architecture is a domain-specific Visual-ARFT.** This is the methodological backbone. |
| **Visual-RFT** | ICCV 2025 | First adaptation of DeepSeek-R1's RL to multimodal. GRPO on Qwen2-VL-2/7B for detection, grounding, classification. | Validates GRPO + verifiable rewards for visual perception on small Qwen VL models. |
| **DAPO** | NeurIPS 2025 | Clip-Higher, Dynamic Sampling, Token-Level Loss, Overlong Reward Shaping. Open-sourced on VeRL. | We already use clip-higher + dynamic sampling. Should add token-level loss + overlong shaping. |
| **S-GRPO** | arXiv 2026 | Unifies SFT and RL via Conditional Ground-Truth Trajectory Injection (CGI). When all rollouts in a group fail, inject the expert trajectory with max reward. | **Directly replaces our oracle-SFT → GRPO two-stage plan with a single unified stage.** Solves cold-start. |
| **DIVA-GRPO** | arXiv 2026 | Difficulty-adaptive variant augmentation. Dynamically adjusts difficulty distribution per problem. | Relevant to our curriculum design — a more principled version of our overlap-bin curriculum. |
| **EvolvedGRPO** | NeurIPS 2025 | Progressive instruction evolution. Trains on basic subproblems first, increases difficulty. Prefix-style process rewards. | Curriculum approach we could adopt. |
| **PEPO** | arXiv 2026 | Token-level perception-exploration policy optimization. Perception prior from hidden states + token entropy gating. | Could help separate "looking" tokens from "reasoning" tokens in our trajectories. |

### 1.2 Agentic tool-use RL for VLMs

| Method | Venue | Key idea | Relevance |
|--------|-------|----------|-----------|
| **PyVision-RL** | ICML 2026 | **Interaction collapse**: models reduce tool usage during RL. Fix: accumulative tool reward (R = R_acc + 0.1 × n_tc × 1[R_acc=1]) + oversampling-filtering-ranking rollout. | **Critical.** Our per-call tool_cost penalty may CAUSE interaction collapse. Must redesign. |
| **MGPO** | ACL 2026 | Multi-turn grounding RL for cropping. Grounding emerges from binary reward alone. Cold-start fix: multi-turn template + restrict policy loss to multi-turn outputs. | **Directly relevant to our crop action.** Shows we may not need grounding annotations. |
| **LFPC** | CVPR 2026 | "Information Gap" mechanism: downsample global image to force crop usage. Grounding loss with few bboxes. | Curriculum technique for teaching cropping. |
| **VISTA-R1** | CVPR 2026 | VISTA-Gym: scalable training env for tool-integrated visual reasoning. 8B model, +9.51-18.72% over baselines. | Template for our training environment design. |
| **NTEP** | arXiv 2026 | Necessary Tool-Evidence Path rewards. Rewards pre-call intent alignment + post-call evidence extraction. Non-repeated-goal regularizer. | **Directly applicable.** Our reward only checks final pose; NTEP adds per-tool-call process rewards. |
| **Visual-ARFT theory** | arXiv 2026 | TA-MDP formalization. GRPO convergence O(1/√T) under composite rewards. Reward Decomposition Theorem. PAC-Bayes generalization bound. | Theoretical backing for our composite reward design. |
| **LiteSearch-VL** | arXiv 2026 | 2B/4B multimodal search agents via trajectory distillation + synthetic step-DPO. SFT transfers the "agent contract"; DPO is refinement. | **Validates 2B model size.** SFT first to learn valid tool-call format, then RL. |
| **ToolOmni** | ACL 2026 | Decoupled Multi-Objective GRPO for tool retrieval + execution. SFT cold-start + RL. | Relevant to our matcher selection action. |

### 1.3 Geometric vision / matching / SfM

| Method | Venue | Key idea | Relevance |
|--------|-------|----------|-----------|
| **ReasonMatch / DCRL** | CVPR 2026 | **Closest concurrent work.** 8B MLLM + GRPO + verifiable matching rewards. 220k pairs from RGB-D/SfM. RL transfers, SFT does NOT. Dynamic curriculum +5.2 F1. | **Must differentiate.** They do direct matching; we do agentic tool orchestration. |
| **Doppelgangers++** | CVPR 2025 | Transformer classifier with MASt3R 3D features for doppelganger detection. | Our doppelganger_check tool should use this or XDG. |
| **XDG** | arXiv 2026 | Efficient doppelganger disambiguation via LoRA on Depth Anything 3. 3x faster. | More efficient doppelganger tool. |
| **Glob3R** | arXiv 2026 | Global SfM with 3D foundation models. Pi3X + dense matching head + motion averaging. | Relevant to Phase 2 scene-level SfM. |
| **VLM-3R** | CVPR 2026 | VLMs augmented with instruction-aligned 3D reconstruction. 200K QA pairs. | Shows VLMs can reason about 3D from monocular video. |
| **Reasmory** | arXiv 2026 | VLMs are brittle with free-form 3D tools. Constrained DSL for spatial memory queries. +6-18% over GPT-5-mini. | **Constrained tool interfaces may beat free-form JSON.** |

### 1.4 Infrastructure

| Resource | Status | Note |
|-----------|--------|------|
| **VeRL** v0.9.0 | Aug 2026 | Now has uni-agent framework (May 2026), VeRL-Omni, VeRL-Tinker. Dedicated multi-turn agent support. |
| **VeRL uni-agent** | May 2026 | Unified framework to build, run, and train LLM agents at scale on VeRL. |
| **SearchAgent-Zero** | May 2026 | VeRL-based multi-turn search agent RL. Reproducible recipe. Template for our agent loop. |

### 1.5 Provenance and positioning

**This project is a direct execution of Gabriele Berton's public research idea.** Berton
(Google DeepMind, ex-Amazon/ICCV author on image matching) posted the concept on LinkedIn
(July 3, 2026) as `#researchidea n11`:

> "agentic SfM - train an MLLM with RL to use tools for cropping, matching, doppelgangers, SfM,
> to get much better 3D reconstruction on difficult scenes... Should be easy to train (the task
> is verifiable and it's easy to sort the samples by difficulty to allow for curriculum
> learning)... Main problem: vision people are not good at RL, and vice versa."

We took this idea and built the infrastructure: Qwen3.5-2B policy, FastAPI tool server,
LoFTR/MASt3R/LightGlue/COLMAP tools, verifiable pose rewards, custom GRPO trainer. Phase 0
zero-shot inference results were posted publicly as Ryukijano. **The remaining work is the
training itself** — Phase 1 (GRPO) and beyond.

**Positioning:** We are executing Berton's proposed idea with specific technical contributions
drawn from the 2025-2026 RL-for-VLM literature:
- S-GRPO's CGI for cold-start (replacing the naive SFT→GRPO two-stage plan).
- PyVision-RL's accumulative tool reward (preventing interaction collapse).
- NTEP process rewards for per-tool-call evidence alignment.
- DAPO's token-level loss and dynamic sampling.
- DCRL-style dynamic curriculum over viewpoint/overlap bins.

**Concurrent work to differentiate against:**
- **ReasonMatch/DCRL** (CVPR 2026): RL for wide-baseline matching, but *direct matching* by the
  MLLM — no tool orchestration, no cropping, no doppelganger check, no SfM. 8B model, 220k pairs.
  Our contribution: agentic tool use (crop → match → verify), not end-to-end matching.
- Berton himself has not published a paper on this; it remains a LinkedIn idea. We have a window
  to be first to execute and write up, but should credit Berton's idea in the paper.

---

## 2. Critical Findings That Change Our Plan

### 2.1 SFT does not transfer; RL does (DCRL, CVPR 2026)

DCRL's ablation is striking:

| Strategy | OmniSpatial | SAT Real | ReasonMatch |
|----------|------------|----------|-------------|
| Base     | 43.6       | 70.0     | 27.5        |
| SFT      | 42.6       | 41.3     | 51.0        |
| DCRL     | 48.9       | 75.3     | 70.5        |

SFT *hurts* transfer (SAT Real drops from 70 → 41.3). RL *helps* transfer across the board.

**Implication:** Our oracle-SFT → GRPO two-stage plan is risky. SFT may lock the model into
a single expert trajectory and hurt generalization.

**Fix:** Adopt **S-GRPO** instead. It unifies SFT and RL: when all rollouts in a group fail
(cold-start), inject the ground-truth trajectory with max reward. This provides the SFT signal
*inside* the RL loop, without a separate SFT stage that causes catastrophic forgetting.

### 2.2 Interaction collapse is the key failure mode (PyVision-RL, ICML 2026)

Models trained with RL tend to **reduce tool usage** over time, converging to short,
low-interaction behaviors. This is exactly what our per-call `tool_cost: 0.02` penalty would
incentivize.

**Fix:** Replace the per-call penalty with PyVision-RL's **accumulative tool reward**:

```
R = R_outcome + 0.1 * n_tool_calls * 1[R_outcome = max]
```

Tool calls are rewarded *only when the final outcome is correct*. This incentivizes productive
multi-turn interaction without rewarding trivial loops.

### 2.3 Grounding can emerge from RL alone (MGPO, ACL 2026)

MGPO trains the model to predict crop coordinates with **only binary answer correctness** as
reward — no grounding annotations needed. The cold-start problem (model won't spontaneously
produce coordinates) is solved by:
1. A multi-turn conversational template.
2. Restricting policy loss to model outputs across dialogue rounds (not the injected crop results).

**Implication:** We may not need oracle crop boxes for training. The model can learn to crop
from the pose reward alone, as long as we handle cold-start.

### 2.4 The 2B model needs to learn the "agent contract" first (LiteSearch-VL)

LiteSearch-VL shows that for 2B models, full-trajectory SFT transfers the "agent contract" —
the ability to produce valid tool-call format. Without it, the 2B model "almost never emits a
usable answer." DPO/RL on top is a refinement, not a phase transition.

**Implication:** For the 2B Qwen3-VL, we *do* need a format-learning stage, but it should be
minimal (just learning valid JSON tool-call syntax), not full trajectory imitation. S-GRPO's
CGI mechanism handles this naturally.

### 2.5 Data scale matters enormously (DCRL: 220k vs our 398)

DCRL uses 220k pairs harvested from RGB-D/SfM. We have 398 MegaDepth pairs. This is a **550x
gap**. With 398 pairs and group_size=8, we have ~3184 rollouts per epoch — far too few for
stable GRPO.

**Fix:** Scale data harvesting. MegaDepth has ~100 scenes with thousands of image pairs each.
We should harvest 5k-10k pairs with verifiable pose supervision, stratified by overlap/difficulty.
DCRL's pipeline (RGB-D + SfM → verified correspondences) is a template.

### 2.6 Constrained tool interfaces beat free-form JSON (Reasmory)

Reasmory shows VLMs are brittle with free-form tool calls — they invoke tools incorrectly, skip
required transformations, or misuse results. A constrained DSL with parsed/validated programs
gives +6-18% over free-form tool use.

**Implication:** Our JSON tool-call format should be as constrained as possible. Consider a
fixed action template with slots, not free-form JSON generation.

---

## 3. What to Keep, Simplify, Remove, Redesign

### Keep
- **Qwen3.5-2B** as the policy model (official Instruct/post-trained 2B VLM; LiteSearch-VL still validates 2B agents + contract SFT).
- **3× L40S GPU layout** (rollout / training / tool server).
- **Verifiable pose-AUC + inlier rewards** (core reward signal, validated by DCRL).
- **Curriculum over difficulty bins** (validated by DCRL: +5.2 F1 over uniform).
- **LoRA adapters** (parameter-efficient, fits 2B on L40S).
- **FastAPI tool server** with LoFTR / MASt3R / LightGlue / COLMAP.
- **Dynamic sampling** (DAPO, already implemented).
- **Clip-higher** (DAPO, already implemented).

### Simplify
- **Action space:** Start with single-step `crop_and_match` only. Remove `retrieve`, `sfm_run`,
  `inspect`, `doppelganger_check` from Phase 1. Add them in Phase 2. MGPO and LFPC both show
  single-step crop → reason works.
- **Multi-turn → bounded depth:** Limit to 1-3 tool calls per episode initially. Visual-ARFT
  theory (TA-MDP) uses bounded-depth tool calls. Scale depth later.
- **Tool-call format:** Move from free-form JSON to a constrained template with slots.

### Remove
- **Per-call tool_cost penalty.** Replace with accumulative tool reward (PyVision-RL).
- **Separate oracle-SFT stage** (or make it minimal). Replace with S-GRPO's CGI.
- **FISSION recovery module** for now. It adds complexity before the basic loop works.
  Revisit if cold-start remains a problem after S-GRPO.

### Redesign
- **Reward function:**
  - Replace `tool_cost` with accumulative tool reward.
  - Add NTEP-style process reward: reward pre-call intent alignment with evidence-seeking goal.
  - Add non-repeated-goal regularizer (penalize redundant crop of same region).
  - Keep pose AUC + inlier + format + invalid penalty.
  - Add token-level loss aggregation (DAPO).
  - Add overlong reward shaping (DAPO) for trajectories exceeding max turns.

- **Training pipeline:**
  - Stage 0: Minimal format SFT (just valid JSON tool-call syntax, ~100 examples, 1 epoch).
  - Stage 1: S-GRPO with CGI (inject oracle trajectory when all rollouts fail).
  - Stage 2: Standard GRPO once the model reliably produces valid tool calls.
  - Stage 3: Add multi-turn depth, doppelganger_check, retrieve, sfm_run.

- **Data pipeline:**
  - Harvest 5k-10k pairs from MegaDepth with verifiable pose supervision.
  - Stratify by overlap (using SfM pose distances) into difficulty bins.
  - Add ScanNet / RE10K / CO3D if MegaDepth is insufficient (following DCRL).
  - Generate ReasonMatch-style region correspondence supervision.

- **Baselines (must strengthen):**
  1. Direct LoFTR (full frame).
  2. Direct MASt3R (full frame).
  3. Random crop + LoFTR (ablation: does crop location matter?).
  4. Oracle crop + LoFTR (upper bound: what if crop is perfect?).
  5. DCRL-style direct MLLM matching (no tools).
  6. Our agentic policy (crop → match → done).
  7. Our agentic policy (multi-turn, Phase 2+).

---

## 4. Staged Implementation Plan

### Phase 1a: Data & Baselines (2-3 weeks)

**Goal:** Scale data and establish strong baselines.

1. **Harvest 5k+ pairs from MegaDepth** with pose supervision.
   - Use existing MegaDepth SfM reconstructions.
   - Compute relative pose for all image pairs within each scene.
   - Stratify by overlap (rotational/translation distance).
   - Store as parquet with image paths, pose, intrinsics, difficulty bin.

2. **Run all baselines** on the expanded dataset:
   - Direct LoFTR, MASt3R, LightGlue on full frame.
   - Random crop + match (multiple crop sizes).
   - Oracle crop + match (using SfM overlap region).
   - Zero-shot Qwen3-VL-2B (no tools, just "describe correspondences").

3. **Commit the Qwen3-VL-2B migration** (currently uncommitted).

### Phase 1b: Minimal Format SFT (1 week)

**Goal:** Teach the 2B model to produce valid tool-call JSON.

- Generate ~200-500 examples of valid `crop_and_match` calls.
- 1 epoch SFT, LoRA rank 32.
- Verify: model produces valid JSON ≥ 90% of the time on held-out pairs.
- This is the "agent contract" from LiteSearch-VL.

### Phase 1c: S-GRPO Training (3-4 weeks)

**Goal:** Train the policy to choose crop regions and matchers.

- Implement S-GRPO's CGI: when all N rollouts in a group fail (zero inliers or
  invalid pose), inject the oracle `crop_and_match` trajectory with max reward.
- Use accumulative tool reward (PyVision-RL).
- Curriculum: medium → hard → extreme (following DCRL's dynamic scheduling).
- Group size 8, dynamic sampling, clip-higher 0.28.
- Token-level loss (DAPO).
- Evaluate every 5 epochs on val set.
- Compare against all baselines from Phase 1a.

### Phase 2: Multi-Turn & Doppelganger (4-6 weeks)

**Goal:** Extend to multi-turn tool use with doppelganger checking.

- Add `doppelganger_check` tool (using XDG or Doppelgangers++).
- Add `match` (without crop) and `crop_and_match` as separate actions.
- Allow 2-5 tool calls per episode.
- Add NTEP process rewards for per-tool-call evidence alignment.
- Add non-repeated-goal regularizer.
- Evaluate: does multi-turn help over single-step?

### Phase 3: Scene-Level SfM (6-8 weeks)

**Goal:** Full agentic SfM on multi-image scenes.

- Add `retrieve` (image retrieval) and `sfm_run` (COLMAP) tools.
- Scene-level reward: reconstruction completeness + accuracy.
- Use Glob3R or MASt3R-SfM as the SfM backbone.
- Compare against COLMAP + LoFTR, COLMAP + MASt3R, VGGT, Glob3R.

### Phase 4: Benchmark & Write-Up (2-3 weeks)

- Evaluate on held-out MegaDepth scenes + external datasets (ScanNet, LaMAR).
- Ablation study: single-step vs multi-turn, with/without doppelganger, with/without
  process rewards, S-GRPO vs standard GRPO, accumulative vs per-call tool cost.
- Statistical significance tests (paired bootstrap).
- Write paper positioning against DCRL (direct matching) and Berton's idea (agentic SfM).

---

## 5. Key Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| 2B model too small for multi-turn reasoning | LiteSearch-VL shows 2B works with SFT contract. Start single-step. |
| Interaction collapse (model stops using tools) | Accumulative tool reward (PyVision-RL). |
| Cold-start (model never produces valid tool calls) | S-GRPO CGI + minimal format SFT. |
| Data too small (398 pairs) | Scale to 5k+ from MegaDepth. |
| DCRL scoops us | Differentiate: agentic tool orchestration vs direct matching. |
| Berton/DeepMind publishes first | Credit Berton's idea in our paper. Our specific technical contributions (S-GRPO CGI, accumulative tool reward, NTEP process rewards, agentic crop→match→verify loop) are distinct. Execute fast. |
| VeRL multi-turn agent complexity | Start with custom GRPO (already built). Migrate to VeRL uni-agent in Phase 2. |
| Reward hacking (trivial tool loops) | Non-repeated-goal regularizer (NTEP). Accumulative reward only on correct outcome. |
| Matcher quality confounds policy quality | Always compare against oracle crop + match upper bound. |

---

## 6. Recommended Near-Term Actions (next 2 weeks)

1. **Commit the Qwen3-VL-2B migration** (uncommitted in working tree).
2. **Scale MegaDepth data harvesting** to 5k+ pairs with pose supervision.
3. **Run all baselines** (direct match, random crop, oracle crop) on expanded data.
4. **Implement S-GRPO's CGI** in `scripts/run_grpo.py`.
5. **Replace tool_cost with accumulative tool reward** in `rewards/pose_rewards.py`.
6. **Add token-level loss aggregation** (DAPO) to the GRPO trainer.
7. **Generate minimal format SFT data** (~200 valid `crop_and_match` examples).
8. **Run a smoke test** of S-GRPO on 100 pairs to validate the loop.

---

## 7. References

1. Visual-ARFT — Liu et al., ICCV 2025. arXiv:2505.14246
2. Visual-RFT — Liu et al., ICCV 2025. github.com/Liuziyu77/Visual-RFT
3. DAPO — Yu et al., NeurIPS 2025. arXiv:2503.14476
4. S-GRPO — arXiv:2604.16557
5. DIVA-GRPO — arXiv:2603.01106
6. EvolvedGRPO — NeurIPS 2025. github.com/SHENZHEBEI/EvolvedGRPO
7. PEPO — arXiv:2603.22847
8. PyVision-RL — ICML 2026. arXiv:2602.20739
9. MGPO — ACL 2026. arXiv:2507.05920. github.com/EvolvingLMMs-Lab/MGPO
10. LFPC — CVPR 2026. arXiv:2603.27494
11. VISTA-R1 — CVPR 2026. github.com/Lucanyc/VISTA-Gym
12. NTEP — arXiv:2609.03493
13. TA-MDP theory — arXiv:2604.19857
14. LiteSearch-VL — arXiv:2608.29357
15. ToolOmni — ACL 2026
16. ReasonMatch/DCRL — CVPR 2026. arXiv:2606.03577. github.com/aim-uofa/ReasonMatch
17. Doppelgangers++ — CVPR 2025
18. XDG — arXiv:2608.29733
19. Glob3R — arXiv:2607.09225
20. VLM-3R — CVPR 2026
21. Reasmory — arXiv:2606.00963
22. VeRL — github.com/vERL-project/vERL (v0.9.0, Aug 2026)
23. VeRL uni-agent — github.com/verl-project/uni-agent (May 2026)
24. SearchAgent-Zero — github.com/NLPJCL/SearchAgent-Zero
25. Berton's agentic SfM idea — LinkedIn, July 3 2026 (origin of this project's concept)
