# OPSFT-style constrained SFT

OPSFT (arXiv 2609.36659) restricts SFT to the exact update support and
direction that an on-policy RL run produced, turning SFT into a constrained
optimization inside the "footprint" of the RL update. Per training step it
enforces two constraints:

- **Stage 1 (gradient mask):** `g <- g * M`, and zero any on-support
  component whose gradient opposes the reference direction, i.e.
  `g_i = 0` where `M_i & g_i * D_i > 0`.
- **Stage 2 (parameter projection):** off-support parameters are pinned to
  the reference, `theta_i = theta0_i` where `~M_i`; on-support parameters
  that cross back over theta0 are restored, i.e. `theta_i <- theta0_i`
  where `M_i & D_i != 0 & (theta_i - theta0_i) * D_i < 0`.

**Caveat — LoRA scoping.** The paper constrains full-parameter updates. We
apply the constraints to the LoRA adapter parameters only (canonicalized
`lora_A`/`lora_B` names, adapter name stripped). The paper's claim therefore
does **not** transfer directly — this is an adaptation experiment.

## Mapping to this repo

| OPSFT concept | Here |
|---|---|
| `theta_base` | adapter that GRPO started from |
| `theta_on` | trained GRPO adapter (RL endpoint) |
| `M` | support mask: `{i : theta_base_i != theta_on_i}` for `exact_support`; global top-`topk_frac` |delta| for `topk` |
| `D` | `sign(theta_on - theta_base)` (int8 in {-1, 0, +1}, float32 diff) |

**Dense-support caveat (measured here):** under AdamW, LoRA updates are
near-dense — the phase-2 GRPO adapter differs from its SFT init in
~99.997% of elements after ~1 h, so `exact_support` leaves only the sign
constraint. `--selection topk --topk-frac 0.1` keeps the largest-magnitude
10% and approximates the paper's sparse-update regime; the Slurm job
defaults to `topk` via `DIRECTION_SELECTION`.

`src/agentic_sfm/opsft.py` provides `extract_direction` (base vs. trained
canonical state dicts -> `{"mask", "sign", "meta"}`), `save_direction` /
`load_direction`, `mask_gradients_` (stage 1), and `project_parameters_`
(stage 2).

## Experiment arms

1. **from-base** — `init_adapter = theta_base` adapter; run constrained SFT
   on oracle data under the extracted `(M, D)`. Tests whether constrained
   SFT can reproduce the on-policy improvement (analog of paper §3.2).
2. **continue** — `init_adapter = a GRPO checkpoint`; same direction.
   Continue SFT past the RL endpoint without reversing it (analog of
   paper §4.2).

## Commands

Extract the direction artifact:

```bash
python scripts/extract_update_direction.py \
    --base outputs/scene_sft_graph_retry2/checkpoints/epoch_5 \
    --trained outputs/phase2_hybrid/latest/checkpoints/latest \
    --output outputs/opsft/direction_scene_grpo.pt
```

Launch constrained SFT (the job auto-extracts if the artifact is missing):

```bash
sbatch jobs/sft_opsft.slurm
# or with overrides:
DIRECTION_BASE=<base_ckpt> DIRECTION_TRAINED=<trained_ckpt> \
    DIRECTION_OUT=<out.pt> sbatch jobs/sft_opsft.slurm
```

Training entry point: `scripts/run_sft_opsft.py --config
configs/phase2_sft_opsft.yaml --output-dir outputs/scene_opsft_base`.

## Known differences vs. the paper

- AdamW lr `2e-5` vs. the paper's `1e-7`.
- bf16 training vs. FP32 — `M`/`D` are extracted from checkpoint tensors,
  so boundary ties can differ from the paper's FP32 setting.
- Mask is defined over LoRA adapter parameters, not the full parameter
  space.
- Stage-1 + stage-2 enforcement is retained per step (as in the paper).

No results are reported yet; this document describes setup only.
