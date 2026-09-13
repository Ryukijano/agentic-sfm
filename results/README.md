# Inference Results

Pre-computed outputs from Phase 0 zero-shot feasibility and real MegaDepth evaluation.

## Layout

```
results/
├── phase0/                    # Synthetic / small-scale feasibility (Jul 2025)
│   ├── comparison.json        # Aggregated metrics: direct LoFTR/MASt3R vs zero-shot agent
│   ├── direct_loftr.json
│   ├── direct_mast3r.json
│   └── agent_zeroshot.json
├── phase0_real/
│   ├── real_results.json      # 12 pairs from MegaDepth scenes 0015 & 0022
│   └── visualizations/        # Per-pair match overlays (direct vs agent)
└── figures/                   # Publication-style summary plots
```

## Phase 0 summary (`phase0/comparison.json`)

| Method | Mean inliers | Mean inlier ratio | Mean pose AUC |
|--------|-------------:|------------------:|--------------:|
| Direct LoFTR | 12.4 | 0.109 | 0.017 |
| Direct MASt3R | 12.4 | 0.109 | 0.017 |
| Zero-shot agent | 8.9 | 0.081 | 0.017 |

**Takeaway:** Stock Qwen3.5-2B / prior Qwen3-VL zero-shot cropping does not yet beat direct matching on hard pairs — motivating Phase 1 GRPO training.

## Real MegaDepth eval (`phase0_real/real_results.json`)

12 image pairs across easy / medium / hard / extreme difficulty bins from MegaDepth scenes `0015` and `0022`. On these pairs the zero-shot agent defaults to full-frame crops, so agent and direct matching produce identical correspondence counts.

| Difficulty | Pairs | Mean direct inliers | Mean agent inliers |
|------------|------:|--------------------:|-------------------:|
| easy | 3 | 1,257 | 1,257 |
| medium | 3 | 864 | 864 |
| hard | 3 | 1,048 | 1,048 |
| extreme | 3 | 616 | 616 |

## Figures

| File | Description |
|------|-------------|
| `figures/result_clean_hero.png` | Match visualizations + bar chart (best overview figure) |
| `figures/result_qualitative_matches.png` | Stacked grid across difficulty levels |
| `figures/result_matches_by_difficulty.png` | Mean matches/inliers by difficulty |
| `figures/result_matches_vs_overlap.png` | Scatter vs overlap score |
| `figures/result_summary_table.png` | Per-pair summary table |
| `figures/agentic_sfm_phase0_comparison.png` | Phase 0 method comparison |
| `figures/agentic_sfm_phase1_architecture.png` | Training architecture diagram |

Per-pair overlays live in `phase0_real/visualizations/` (`*_direct.png` vs `*_agent.png`).
