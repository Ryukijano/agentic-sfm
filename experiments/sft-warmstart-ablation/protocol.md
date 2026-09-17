# Experiment: sft-warmstart-ablation

Pre-registered per `experiment-protocol` skill. Date: 2026-09-16.

## Hypothesis & Prediction

**H1** (primary): For this task (Qwen3.5-2B VLM, strict tool grammar, sparse
verifiable-geometric reward), GRPO warm-started from the *full* SFT adapter
(epoch_5) reaches a higher held-out eval reward than GRPO from an earlier SFT
epoch — i.e., the "paradox of perfection" / over-peaked-SFT effect reported
for tool-use RL (RC-GRPO arXiv:2602.03025; ICLR 2026 "optimal SFT ckpt ≠ best
RL start"; PEAR arXiv:2602.01058) is *small* here because our SFT is short
(5 epochs, 1.4k examples).

**H2** (secondary): Pure RL from base (no SFT) + S-GRPO oracle injection is
competitive but sample-inefficient — consistent with Tool-Zero (2511.01934)
and Tool-R0 (2602.21320) on pure-RL tool use, vs. the cold-start-collapse
failure mode our base S-GRPO paper (2604.16557) predicts for sparse-reward
vision.

**H3** (control arm): The reward-hacked adapter (run-1 epoch-1 `latest`,
preserved on HF) does *not* recover under strict crop validation within the
same step budget — the malformed-bbox habit is entrenched, not transient.

## Design

Four arms, identical GRPO config (`configs/phase1_grpo.yaml`, fixed tool
server post-`e6189e3` strict crop validation), differing only in warm-start:

| ID | Warm start | Tests |
|----|-----------|-------|
| A | `outputs/sft/checkpoints/epoch_5` | H1 baseline (current recipe) |
| B | `outputs/sft/checkpoints/epoch_2` | H1 (less-peaked SFT → more headroom?) |
| C | none (base Qwen3.5-2B) | H2 (pure RL + oracle injection) |
| D | HF `phase1_grpo/latest` (0.388, post-drift) | H3 (recovery control) |

Screening ablation — 1 seed per arm, fixed `seed` in config, ~2 epochs each.
This is a directional screen, NOT a powered comparison; follow-up arms get
3 seeds on any arm within noise of the winner.

## Primary Outcome

**Held-out eval mean reward** (`hard_pairs_val.json`, 153 pairs) at the
same global_step budget per arm.

## Secondary Outcomes (exploratory — labelled as such)

- `train/tool_error_rate`, `train/done_rate`, `train/format_reward_mean`
  (newly instrumented — reward-hack visibility)
- Fraction of zero-variance (DAPO-filtered) groups per step
- Reward std within groups (exploration diversity proxy)
- `inlier_reward` / `pose_reward` component means

## Controls & Ablations

- Same dataset: `hard_pairs_train.json` (1382 pairs, sha256:90bda9de),
  `hard_pairs_val.json` (153 pairs, sha256:fff31567)
- Same tool server image: strict `/crop` (commit `e6189e3`)
- Same hardware: 3× L40S via `jobs/phase1_grpo.slurm`
- Leakage check: val pairs disjoint from train (verified at dataset build)

## Data

MegaDepth-derived hard pairs; ScanNet not used at pair level.

## Analysis Plan

Compare primary outcome across arms at equal global_step. Report per-arm
secondary metrics as a table. No significance testing at n=1 seed — arms
differing by < eval-reward noise floor (~1 SE of batch means, to be
estimated from run A) are declared a screen tie and promoted to seeded runs.

## Stopping Rules

- Stop an arm early if `tool_error_rate` > 0.5 for 3 consecutive logged
  steps (environment failure, not learning) or eval reward diverges.
- Do not extend an arm past its budget to "rescue" a result — amend instead.

## Pre-registration Status

- [x] Checklist complete (2026-09-16): primary outcome fixed, conditions
      fixed, data snapshot hashed, stopping rules set.

## Findings update (2026-09-17) — the trained adapters were degenerate all along

Offline verification (HF+PEFT, real images, rollout sampling settings) proved:

- `outputs/phase1/checkpoints/best` (the "0.399" epoch-1 adapter): **6/6 samples
  degenerate** — tiny/degenerate bboxes, duplicated JSON keys, leaked args,
  chat-template markers, token repetition.
- `outputs/sft/checkpoints/epoch_5`: **clean tool JSON with valid bboxes**.

Mechanism (4 independent literature reviews + code audit):
1. The GRPO importance ratio was vacuous (`old`/`new` logprobs both computed
   under the current model at the same instant -> ratio ~= 1 -> clip dead).
2. Rollouts lagged the policy by a full epoch (vLLM reloaded only at epoch
   end), so the epoch-1 "0.399" measured the SFT policy, not the trained one.
3. No KL anchor let the policy drift into degenerate/repetitive modes.
4. "Least-bad-wins": degenerate tokens were exclusive to the least-bad group
   winner, while clean-box tokens had their gradient cancelled by appearing
   in both failed episodes and the CGI oracle.

Consequently the SFT-vs-RL ablation is now framed as: **SFT epoch_5 is the
policy anchor (KL reference + warm start)**, and the ablation's remaining
question is how fast/stable RL can move away from it with the new stabilizers
(on-policy reload, KL coef, off-support veto, first-call validity gate,
binary format reward, NTEP redundancy penalty, min-p/top-k sampling).
