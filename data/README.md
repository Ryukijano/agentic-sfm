# Dataset

Hard image pairs are **not** committed to this repository (MegaDepth images are multi-GB).

## Build locally

```bash
# Requires MegaDepth with COLMAP reconstructions under:
#   /path/to/megadepth/

python scripts/build_megadepth_pairs.py
python scripts/generate_training_pairs.py   # optional: expand pair pool
```

This produces:

- `hard_pairs_train.json` — 398 pairs (medium/hard/extreme/easy)
- `hard_pairs_val.json` — held-out validation split

Each entry includes image paths, overlap score, difficulty bin, and ground-truth relative pose from COLMAP.

## Difficulty bins

| Bin | Overlap | Count (train) |
|-----|---------|--------------:|
| easy | > 0.5 | 8 |
| medium | 0.3–0.5 | 190 |
| hard | 0.1–0.3 | 189 |
| extreme | < 0.1 | 11 |

See `results/figures/agentic_sfm_hardpairs_distribution.png` for the full distribution plot.
