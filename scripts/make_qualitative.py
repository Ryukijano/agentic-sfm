#!/usr/bin/env python
"""Build a publication-quality qualitative figure for the Agentic-SfM agent.

Composes a multi-panel figure good enough for an X post / paper teaser:

    [ image pair A + inlier matches ]  [ 3D sparse reconstruction ]
    [ image pair B + inlier matches ]  [ reward curve over training ]

Everything is computed OFFLINE — no vLLM / tool server needed. It runs real
LoFTR matches on MegaDepth pairs and renders a persisted COLMAP model.

Usage:
    python scripts/make_qualitative.py \
        --scene 0022 \
        --pairs 0 1  5 20 \
        --recon outputs/phase0_real/scene_0022_colmap/sparse/0 \
        --rewards-log /scratch/kcwp264/logs/agentic-sfm/sgrpo_smoke2_7869472.out \
        --output results/figures/qualitative.png
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools_server"))

from agentic_sfm.eval.viz import (  # noqa: E402
    FIG_BG, C_TEXT, C_ACCENT, draw_match_pair, render_point_cloud,
    render_gt_pointcloud, plot_training_curves,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _scene_images(root: Path, scene: str) -> list[Path]:
    d = root / scene / "images"
    return sorted(d.glob("*.jpg")) or sorted(d.glob("*.png"))


def _run_loftr(img_a: Path, img_b: Path, matcher: str = "loftr") -> dict:
    """Run a real match via the tool functions (no server)."""
    import server as TS  # tools_server/server.py
    TS._IMAGE_STORE["a"] = str(img_a)
    TS._IMAGE_STORE["b"] = str(img_b)
    return TS.tool_match("a", "b", matcher)


def _scannet_scene_dir(root: Path, scene_id: str) -> Path:
    """Resolve a ScanNet scene dir from 'scene0772_00' / '0772_00' / '772'."""
    root = Path(root)
    if not list(root.glob("scene*_*")) and (root / "scannet_test_1500").is_dir():
        root = root / "scannet_test_1500"
    sid = scene_id if scene_id.startswith("scene") else f"scene{scene_id}"
    cand = root / sid
    if cand.is_dir():
        return cand
    stem = sid[len("scene"):]
    matches = [d for d in root.glob("scene*_*")
               if d.is_dir() and (d.name == f"scene{stem}"
                                  or d.name.split("_")[0] == f"scene{stem}"
                                  or d.name.split("_")[0] == f"scene{stem.zfill(4)}")]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"ScanNet scene '{scene_id}' not found under {root}")


def _parse_rewards(log_path: Path) -> list[float]:
    """Pull 'reward=X.XXX' values out of a trainer log."""
    if not log_path or not log_path.exists():
        return []
    out = []
    rx = re.compile(r"reward=([0-9.]+)")
    for line in log_path.read_text(errors="ignore").splitlines():
        m = rx.search(line)
        if m and "Step" in line:
            out.append(float(m.group(1)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-root", default="/scratch/kcwp264/data/megadepth/megadepth_test_1500/Undistorted_SfM")
    ap.add_argument("--scene", default="0015")
    ap.add_argument("--pairs", nargs="*", type=int, default=[0, 1, 2, 30],
                    help="flat list of image indices forming pairs")
    ap.add_argument("--matcher", default="loftr")
    ap.add_argument("--recon", default="outputs/phase0_real/scene_0022_colmap/sparse/0")
    ap.add_argument("--rewards-log", default=None)
    ap.add_argument("--scannet-scene", default=None,
                    help="ScanNet scene id (e.g. scene0772_00 or 0772_00) — "
                         "adds a dense GT point-cloud panel")
    ap.add_argument("--scannet-root",
                    default="/scratch/kcwp264/data/megadepth/scannet_test_1500/scannet_test_1500")
    ap.add_argument("--scannet-max-points", type=int, default=200000)
    ap.add_argument("--output", default="results/figures/qualitative.png")
    args = ap.parse_args()

    root = Path(args.image_root)
    imgs = _scene_images(root, args.scene)
    if len(imgs) < 2:
        raise SystemExit(f"Not enough images in {root}/{args.scene}")

    # Resolve the requested pairs into (ia, ib) index tuples.
    idx = args.pairs
    pair_idx = [(idx[i], idx[i + 1]) for i in range(0, len(idx) - 1, 2)]
    pair_idx = [(a % len(imgs), b % len(imgs)) for a, b in pair_idx][:2]

    # Right column panels: sparse recon, ScanNet dense GT cloud, reward curve.
    rewards = _parse_rewards(Path(args.rewards_log)) if args.rewards_log else []
    n_right = sum([bool(args.recon), bool(args.scannet_scene), bool(rewards)])
    n_rows = max(2, n_right)
    n_panels = len(pair_idx) + n_right
    fig = plt.figure(figsize=(7.2 * max(2, n_panels - 1), 6.4 * n_rows / 2), dpi=150)
    fig.patch.set_facecolor(FIG_BG)

    # grid: left column = match panels (stacked), right = recon/GT/curve
    gs = fig.add_gridspec(n_rows, 2, width_ratios=[1.15, 1.0],
                          hspace=0.18, wspace=0.08)
    right_row = 0  # next free slot in the right column

    # --- match panels (left column) ---
    for row, (a, b) in enumerate(pair_idx[:2]):
        res = _run_loftr(imgs[a], imgs[b], args.matcher)
        ka = np.asarray(res.get("keypoints_a") or [], dtype=float)
        kb = np.asarray(res.get("keypoints_b") or [], dtype=float)
        raw_mask = res.get("inlier_mask") or []
        mask = np.asarray(raw_mask, dtype=bool) if len(raw_mask) else None
        sub = draw_match_pair(
            imgs[a], imgs[b], ka, kb, inlier_mask=mask,
            title=f"{args.matcher} · {res.get('num_inliers', 0)} inliers "
                  f"({res.get('inlier_ratio', 0.0):.0%})",
        )
        # paste the sub-figure as an image into the composite
        buf = __import__("io").BytesIO()
        sub.savefig(buf, format="png", facecolor=FIG_BG, bbox_inches="tight")
        buf.seek(0)
        ax = fig.add_subplot(gs[row, 0])
        ax.imshow(np.asarray(Image.open(buf)))
        ax.axis("off")
        plt.close(sub)

    # --- reconstruction (top right) ---
    if args.recon:
        try:
            import pycolmap
            recon = pycolmap.Reconstruction(args.recon)
            rf = render_point_cloud(reconstruction=recon,
                                    title=f"Scene recon · {len(recon.images)} imgs")
            buf = __import__("io").BytesIO()
            rf.savefig(buf, format="png", facecolor=FIG_BG, bbox_inches="tight")
            buf.seek(0)
            ax = fig.add_subplot(gs[right_row, 1]); right_row += 1
            ax.imshow(np.asarray(Image.open(buf)))
            ax.axis("off")
            plt.close(rf)
        except Exception as e:
            logger.warning(f"recon panel failed: {e}")

    # --- ScanNet dense GT point cloud ---
    if args.scannet_scene:
        try:
            sn_dir = _scannet_scene_dir(Path(args.scannet_root), args.scannet_scene)
            gtf = render_gt_pointcloud(
                depth_dir=sn_dir / "depth",
                color_dir=sn_dir / "color",
                pose_dir=sn_dir / "pose",
                intrinsic_path=sn_dir / "intrinsic" / "intrinsic_depth.txt",
                scene_id=sn_dir.name,
                max_points=args.scannet_max_points,
            )
            buf = __import__("io").BytesIO()
            gtf.savefig(buf, format="png", facecolor=FIG_BG, bbox_inches="tight")
            buf.seek(0)
            ax = fig.add_subplot(gs[right_row, 1]); right_row += 1
            ax.imshow(np.asarray(Image.open(buf)))
            ax.axis("off")
            plt.close(gtf)
        except Exception as e:
            logger.warning(f"scannet GT cloud panel failed: {e}")

    # --- reward curve ---
    if rewards:
        cf = plot_training_curves(rewards, title="S-GRPO reward")
        buf = __import__("io").BytesIO()
        cf.savefig(buf, format="png", facecolor=FIG_BG, bbox_inches="tight")
        buf.seek(0)
        ax = fig.add_subplot(gs[right_row, 1]); right_row += 1
        ax.imshow(np.asarray(Image.open(buf)))
        ax.axis("off")
        plt.close(cf)

    fig.suptitle(
        "Agentic-SfM — RL-trained multimodal agent for hard 3D matching",
        color=C_TEXT, fontsize=16, fontweight="bold", y=0.99,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=FIG_BG, bbox_inches="tight")
    logger.info(f"Saved qualitative figure -> {out}")


if __name__ == "__main__":
    main()
