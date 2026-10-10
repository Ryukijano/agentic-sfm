"""Diagnose VGGT point-map confidence and how many correspondences survive each threshold.

Run on a GPU node. For a handful of consecutive ScanNet pairs it runs one
VGGT forward per pair, then reports the confidence distribution and the
number of reciprocal-NN correspondences that remain at each candidate
threshold. Use the output to choose AGENTIC_SFM_VGGT_CONF_THRESHOLD before
a full benchmark run.

    python scripts/diagnose_vggt_conf.py --scene /path/to/scene0715_00 --pairs 6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from agentic_sfm.pose_prior import reciprocal_nn_matches  # noqa: E402

THRESHOLDS = [1.0, 1.5, 2.0, 3.0, 5.0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--pairs", type=int, default=6)
    ap.add_argument("--max-size", type=int, default=518)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from tools_server import server

    color = Path(args.scene) / "color"
    frames = sorted(color.glob("*.jpg"), key=lambda p: int(p.stem))[: 2 * args.pairs]
    rows = []
    for i in range(0, len(frames) - 1, 2):
        a = np.asarray(Image.open(frames[i]).convert("RGB"))
        b = np.asarray(Image.open(frames[i + 1]).convert("RGB"))
        out = server._run_vggt_pair(a, b, args.max_size)
        row = {"pair": [frames[i].stem, frames[i + 1].stem]}
        for name, conf in (("a", out["conf_a"]), ("b", out["conf_b"])):
            row[f"conf_{name}_pct"] = {
                str(q): round(float(np.percentile(conf, q)), 3) for q in (5, 50, 95)
            }
        for t in THRESHOLDS:
            pa, pb = reciprocal_nn_matches(
                out["points_a"], out["points_b"], out["conf_a"], out["conf_b"],
                conf_threshold=t,
            )
            row[f"matches_t{t}"] = int(len(pa))
        rows.append(row)
        print(json.dumps(row), flush=True)

    summary = {
        t: {
            "median_matches": float(np.median([r[f"matches_t{t}"] for r in rows])),
            "pairs_with_zero": int(sum(r[f"matches_t{t}"] == 0 for r in rows)),
        }
        for t in THRESHOLDS
    }
    print("SUMMARY", json.dumps(summary, indent=1), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps({"rows": rows, "summary": summary}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
