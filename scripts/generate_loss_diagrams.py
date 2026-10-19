"""Generate the three publication-quality reconstruction-loss diagrams.

Outputs (dark ``#0d1117`` theme matching ``agentic_sfm.eval.viz``):

1. ``docs/figures/reconstruction_pipeline_architecture.png``
   Closed-loop Agentic-SfM architecture + the 4-Pillar Reconstruction Losses
   feeding the GRPO / S-GRPO advantage update.

2. ``docs/figures/mathematical_vs_visual_losses_demo.png``
   3x4 grid computed with the *real* functions in
   ``agentic_sfm.rewards.reconstruction_loss`` on a synthetic L-shaped room:
   accurate reconstruction vs. doppelganger symmetry fold vs. abstention.

3. ``docs/figures/loss_response_curves.png``
   2x2 response surfaces of the loss functions.

Self-contained: inserts ``<repo>/src`` into ``sys.path`` so it runs with any
python that has numpy/scipy/matplotlib/Pillow. Deterministic (fixed seeds).

Usage:
    python scripts/generate_loss_diagrams.py
"""

from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ---------------------------------------------------------------------------
# Resolve repo + import the real loss functions
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentic_sfm.rewards.reconstruction_loss import (  # noqa: E402
    align_poses_and_points_sim3,
    compute_all_pairs_pose_maa,
    compute_graph_rigidity_metrics,
    compute_intrinsic_ba_metrics,
    compute_multiview_patch_consistency,
    compute_orthographic_3d_render_loss,
    compute_pointcloud_f1,
    compute_projected_depth_loss,
    compute_splat_render_loss,
    render_pointcloud_splat,
)

FIG_BG = "#0d1117"
PANEL_BG = "#161b22"
EDGE = "#30363d"
C_TEXT = "#ecf0f1"
C_MUTED = "#8b949e"
C_GREEN = "#2ecc71"
C_RED = "#e74c3c"
C_BLUE = "#3498db"
C_ORANGE = "#f39c12"
C_PURPLE = "#a371f7"
C_PINK = "#db61a2"
C_TEAL = "#39c5cf"

OUT_DIR = REPO_ROOT / "docs" / "figures"
RNG = np.random.default_rng(7)

plt.rcParams.update({
    "font.family": "sans-serif",
    "text.color": C_TEXT,
    "axes.edgecolor": EDGE,
    "axes.labelcolor": C_TEXT,
    "axes.facecolor": PANEL_BG,
    "xtick.color": C_TEXT,
    "ytick.color": C_TEXT,
    "figure.facecolor": FIG_BG,
    "savefig.facecolor": FIG_BG,
})


# ===========================================================================
# Synthetic L-shaped room (z-up world; BEV axes (x, y, z) matches the
# reconstruction_loss ortho-view convention where depth axis = z = height)
# ===========================================================================
WALL_H = 2.5
# L footprint polygon (x, y): leg A = [0,4]x[0,2.4], wing B = [0,1.6]x[2.4,5]
FLOOR_POLY = np.array([[0, 0], [4, 0], [4, 2.4], [1.6, 2.4], [1.6, 5], [0, 5]])
PIVOT = np.array([1.6, 2.4])  # hinge where wing B joins leg A

# wall segments: (p0, p1, base_color, fold_group)
WALLS = [
    (np.array([0.0, 0.0]), np.array([4.0, 0.0]), np.array([0.75, 0.35, 0.25]), "A"),   # wall A (south)
    (np.array([4.0, 0.0]), np.array([4.0, 2.4]), np.array([0.30, 0.55, 0.80]), "A"),   # east
    (np.array([4.0, 2.4]), np.array([1.6, 2.4]), np.array([0.55, 0.45, 0.75]), "A"),   # stub
    (np.array([1.6, 2.4]), np.array([1.6, 5.0]), np.array([0.85, 0.65, 0.25]), "B"),   # wing east
    (np.array([1.6, 5.0]), np.array([0.0, 5.0]), np.array([0.40, 0.70, 0.45]), "B"),   # wing north
    (np.array([0.0, 5.0]), np.array([0.0, 0.0]), np.array([0.70, 0.30, 0.50]), "AB"),  # west (spans both)
]


def _wall_points() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample the wall surfaces + floor; returns pts, cols, fold_mask."""
    pts, cols, fold = [], [], []
    for p0, p1, base, grp in WALLS:
        n = int(np.linalg.norm(p1 - p0) * 28)
        for t in np.linspace(0, 1, n):
            xy = p0 + t * (p1 - p0)
            for z in np.linspace(0.04, WALL_H - 0.04, 22):
                # striped wallpaper + noise texture
                stripe = 0.10 * np.sin(14 * t * np.linalg.norm(p1 - p0))
                noise = RNG.normal(0, 0.04, 3)
                pts.append([xy[0], xy[1], z])
                cols.append(np.clip(base + stripe + noise, 0, 1))
                fold.append(grp in ("B", "AB") and xy[1] >= PIVOT[1] - 1e-6)
    # floor (checkerboard)
    for x in np.arange(0.06, 4.0, 0.14):
        for y in np.arange(0.06, 5.0, 0.14):
            inside_a = (0 <= x <= 4) and (0 <= y <= 2.4)
            inside_b = (0 <= x <= 1.6) and (2.4 <= y <= 5)
            if not (inside_a or inside_b):
                continue
            c = 0.30 + 0.12 * ((int(x / 0.55) + int(y / 0.55)) % 2)
            pts.append([x, y, 0.0])
            cols.append(np.clip(np.array([c, c * 0.95, c * 0.85]) + RNG.normal(0, 0.02, 3), 0, 1))
            fold.append(bool(inside_b and y >= PIVOT[1]))
    return np.asarray(pts), np.asarray(cols), np.asarray(fold, dtype=bool)


def _lookat_cw(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """4x4 cam-from-world, z-forward convention, world z-up."""
    eye = np.asarray(eye, dtype=np.float64)
    f = np.asarray(target, dtype=np.float64) - eye
    f /= np.linalg.norm(f)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(f, up)
    right /= np.linalg.norm(right) + 1e-9
    down = np.cross(right, f)
    R = np.stack([right, -down, f])  # rows: cam x(right), y(up in img), z(fwd)
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = -R @ eye
    return M


GT_PTS, GT_COLS, FOLD_MASK = _wall_points()

# 8 cameras: 5 in leg A, 3 inside wing B.  Cam index -> fold group.
_CAM_SPECS = [  # (eye xyz, target xyz, fold_group)
    ((0.7, 1.9, 1.5), (3.5, 0.4, 1.2), "A"),
    ((3.3, 1.8, 1.4), (0.6, 0.3, 1.1), "A"),
    ((2.0, 0.5, 1.6), (1.5, 2.3, 1.2), "A"),
    ((3.5, 0.6, 1.3), (2.0, 2.2, 1.3), "A"),
    ((0.5, 0.5, 1.4), (3.8, 2.2, 1.3), "A"),
    ((0.5, 4.5, 1.5), (1.5, 2.6, 1.3), "B"),
    ((1.1, 2.9, 1.4), (0.2, 4.8, 1.2), "B"),
    ((0.4, 3.0, 1.6), (1.5, 4.9, 1.3), "B"),
]
GT_POSES = {i: _lookat_cw(np.array(e), np.array(t)) for i, (e, t, _) in enumerate(_CAM_SPECS)}
CAM_FOLD = np.array([g == "B" for *_, g in _CAM_SPECS])
N_CAMS = len(GT_POSES)

K_MAT = np.array([[110.0, 0.0, 64.0], [0.0, 110.0, 64.0], [0.0, 0.0, 1.0]])
IMG_SIZE = 128


def _rotz(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def fold_scene(pts: np.ndarray, poses: dict[int, np.ndarray],
               theta: float) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Rotate wing-B points (and their cameras) by ``theta`` about PIVOT (z axis)."""
    Rz = _rotz(theta)
    p3 = np.array([PIVOT[0], PIVOT[1], 0.0])
    out_pts = pts.copy()
    m = FOLD_MASK
    out_pts[m] = (Rz @ (pts[m] - p3).T).T + p3
    out_poses = {}
    for i, M in poses.items():
        if not CAM_FOLD[i]:
            out_poses[i] = M.copy()
            continue
        c = -M[:3, :3].T @ M[:3, 3]          # camera centre (world)
        c2 = Rz @ (c - p3) + p3
        R_wc = M[:3, :3].T
        R_wc2 = Rz @ R_wc
        M2 = np.eye(4)
        M2[:3, :3] = R_wc2.T
        M2[:3, 3] = -R_wc2.T @ c2
        out_poses[i] = M2
    return out_pts, out_poses


def _gt_depth_maps() -> dict[int, np.ndarray]:
    """Z-buffered GT depth maps from the GT cloud (for depth_consistency)."""
    depths = {}
    for i, M in GT_POSES.items():
        out = render_pointcloud_splat(GT_PTS, GT_COLS, K_MAT, M,
                                      width=IMG_SIZE, height=IMG_SIZE,
                                      splat_radius_px=6)
        depths[i] = out["depth"]
    return depths


def _gt_images() -> dict[int, np.ndarray]:
    """'Observed' camera frames = GT point-splat renders (float RGB 0-1)."""
    imgs = {}
    for i, M in GT_POSES.items():
        out = render_pointcloud_splat(GT_PTS, GT_COLS, K_MAT, M,
                                      width=IMG_SIZE, height=IMG_SIZE,
                                      splat_radius_px=3)
        imgs[i] = out["rgb"]
    return imgs


GT_IMAGES = _gt_images()
GT_DEPTHS = _gt_depth_maps()


def _graph_edges(registered: list[int], cycle_error_deg: float = 0.0):
    """Ring+chords view-graph over registered cams with consistent relative
    rotations; inject ``cycle_error_deg`` so every 3-cycle reports it."""
    regs = sorted(registered)
    edges = []
    for a, i in enumerate(regs):
        for b, j in enumerate(regs):
            if j <= i:
                continue
            # connect neighbours and next-nearest neighbours in sorted order
            if (b - a) not in (1, 2):
                continue
            R_ij = GT_POSES[j][:3, :3] @ GT_POSES[i][:3, :3].T
            if cycle_error_deg > 0:
                th = math.radians(cycle_error_deg)
                E = _rotz(th)
                R_ij = E @ R_ij
            edges.append((i, j, R_ij))
    return edges


def _intrinsic(row: int) -> dict[str, float]:
    """Synthetic BA reprojection/track stats per scenario row."""
    if row == 0:    # accurate: dense long tracks, sub-pixel BA
        errs = RNG.normal(0.7, 0.15, 400)
        tlens = RNG.integers(4, 8, 400)
    elif row == 1:  # doppelganger fold: low 2-view error on bad pair, thin tracks
        errs = np.concatenate([RNG.normal(0.8, 0.2, 200), RNG.normal(1.1, 0.3, 200)])
        tlens = np.concatenate([RNG.integers(4, 7, 200), np.full(200, 2)])
    else:           # abstention: easy wall only, short clean tracks
        errs = RNG.normal(0.6, 0.1, 120)
        tlens = np.full(120, 2)
    return compute_intrinsic_ba_metrics(np.abs(errs), tlens)


def evaluate_prediction(pred_pts, pred_cols, pred_poses,
                        cycle_err_deg: float, row: int) -> dict[str, float]:
    """Run the real loss stack on a predicted reconstruction."""
    out: dict[str, float] = {}
    intr = _intrinsic(row)
    out.update({f"p1_{k}": v for k, v in intr.items()})
    rig = compute_graph_rigidity_metrics(
        N_CAMS, _graph_edges(sorted(pred_poses), cycle_err_deg))
    out.update({f"p1_{k}": v for k, v in rig.items()})

    maa = compute_all_pairs_pose_maa(pred_poses, GT_POSES, num_total_images=N_CAMS)
    out.update({f"p2_{k}": v for k, v in maa.items()})
    al = align_poses_and_points_sim3(pred_poses, GT_POSES, pred_pts)
    out["p2_ate_reward"] = al["ate_reward"]
    f1 = compute_pointcloud_f1(al["aligned_points3d"], GT_PTS)
    out.update({f"p2_{k}": v for k, v in f1.items()})
    dep = compute_projected_depth_loss(al["aligned_points3d"], GT_POSES,
                                       {i: K_MAT for i in GT_POSES}, GT_DEPTHS)
    out["p2_depth_reward"] = dep["depth_consistency_reward"]

    ortho = compute_orthographic_3d_render_loss(pred_pts, pred_cols, GT_PTS, GT_COLS)
    out.update({f"p3_{k}": v for k, v in ortho.items()})
    spl = compute_splat_render_loss(pred_pts, pred_cols, GT_POSES,
                                    {i: K_MAT for i in GT_POSES}, GT_IMAGES)
    out.update({f"p3_{k}": v for k, v in spl.items()})
    pat = compute_multiview_patch_consistency(
        pred_pts, {i: GT_POSES[i] for i in sorted(pred_poses)},
        {i: K_MAT for i in pred_poses},
        {i: GT_IMAGES[i] for i in pred_poses})
    out.update({f"p3_{k}": v for k, v in pat.items()})

    p1 = 0.5 * intr["intrinsic_ba_reward"] + 0.5 * rig["graph_rigidity_reward"]
    p2 = (0.4 * maa["all_pairs_maa"] + 0.3 * al["ate_reward"]
          + 0.3 * f1["f1_3d"])
    p3 = (ortho["ortho_visual_reward"] + spl["splat_visual_reward"]
          + pat["patch_consistency_reward"]) / 3.0
    r_recon = (p1 + p2 + p3) / 3.0
    efficiency = [0.92, 0.78, 0.95][row]          # tool-call efficiency per row
    total = r_recon * (0.5 + 0.5 * efficiency)  # P4 multiplicative gate
    # Legacy proxy: registration fraction + reproj-only term (no 3D truth)
    legacy = 0.5 * (len(pred_poses) / N_CAMS) + 0.5 * math.exp(
        -intr["huber_reproj_error"] / 2.0)
    out.update({"pillar1": p1, "pillar2": p2, "pillar3": p3,
                "r_recon": r_recon, "new_reward": total, "legacy": legacy})
    return out


# ===========================================================================
# Figure 1 — architecture diagram
# ===========================================================================
def _box(ax, x, y, w, h, label, fc, fontsize=10.5, ec=EDGE, sub=None,
         sub_fs=8.0, lw=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                              boxstyle="round,pad=0.012,rounding_size=0.015",
                              facecolor=fc, edgecolor=ec, lw=lw,
                              mutation_aspect=1.0))
    ty = y + h * (0.62 if sub else 0.5)
    ax.text(x + w / 2, ty, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=C_TEXT)
    if sub:
        ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center",
                fontsize=sub_fs, color=C_MUTED)


def _arrow(ax, p0, p1, color=C_MUTED, lw=2.0, style="-|>", shrinkA=2,
           shrinkB=2, connectionstyle="arc3,rad=0", label=None, lfs=8.5,
           lcolor=None, loffset=(0, 0)):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, color=color, lw=lw,
                                 shrinkA=shrinkA, shrinkB=shrinkB,
                                 mutation_scale=14,
                                 connectionstyle=connectionstyle))
    if label:
        mx, my = (p0[0] + p1[0]) / 2 + loffset[0], (p0[1] + p1[1]) / 2 + loffset[1]
        ax.text(mx, my, label, fontsize=lfs, color=lcolor or color,
                ha="center", va="center", fontstyle="italic")


def fig1_architecture(path: Path):
    fig, ax = plt.subplots(figsize=(19.5, 10.5))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    fig.patch.set_facecolor(FIG_BG)

    ax.text(2, 96.5, "Agentic SfM — Closed-Loop Reconstruction & 4-Pillar Loss Stack",
            fontsize=20, fontweight="bold", color=C_TEXT)
    ax.text(2, 92.5, "VLM policy drives SfM tools; the sparse 3D model is scored by intrinsic, "
            "extrinsic, visual and efficiency-gated losses that shape the GRPO advantage.",
            fontsize=11, color=C_MUTED)

    # ---- left pipeline ----
    _box(ax, 2, 62, 13, 14, "Unordered\nScene Images", "#21262d", sub="N views, no poses")
    _box(ax, 19, 62, 15, 14, "VLM Policy", "#1f3a5f",
         sub="Qwen/Qwen3.5-2B\n(GRPO / S-GRPO)", ec=C_BLUE, lw=2.2)
    _box(ax, 38, 58, 17, 22, "Tool Server", "#2c2116",
         sub="build_initial_graph\nverify_doppelganger\nprune_edges\n"
             "bridge_components\nsfm_run", ec=C_ORANGE, lw=2.0)
    ax.text(46.5, 55.5, "LoFTR · MASt3R · VGGT · MapAnything",
            fontsize=8, color=C_ORANGE, ha="center")
    _box(ax, 59, 62, 15, 14, "pycolmap\nSparse 3D Model", "#16321f",
         sub="sparse/0\npoints X_k · tracks T_k\nposes R_i, t_i", ec=C_GREEN, lw=2.0)

    _arrow(ax, (15.2, 69), (18.8, 69), color=C_TEXT)
    _arrow(ax, (34.2, 69), (37.8, 69), color=C_TEXT, label="tool calls",
           loffset=(0, 2.2))
    _arrow(ax, (55.2, 69), (58.8, 69), color=C_TEXT)

    # ---- visual loop closure (top feedback) ----
    _box(ax, 24, 82, 30, 9, "render_diagnostic_4panel", "#2d1f3d",
         sub="BEV Floorplan + 3D Isometric + Reproj Heatmap + Splat View",
         ec=C_PURPLE, lw=2.0)
    _arrow(ax, (66, 76.2), (46, 81.8), color=C_PURPLE,
           connectionstyle="arc3,rad=-0.25")
    _arrow(ax, (32, 81.8), (28, 76.4), color=C_PURPLE,
           label="Visual Loop Closure (inspect)", loffset=(-9.5, -0.5),
           lcolor=C_PURPLE)

    # ---- right: four pillars ----
    px, pw = 78, 20
    _box(ax, px, 76, pw, 12, "Pillar 1 · Intrinsic Math", "#12261b",
         sub="Huber BA Reproj + Track Rigidity\n(|T_k| ≥ 3) + Fiedler λ₂ & 3-Cycle Error",
         ec=C_GREEN, fontsize=10)
    ax.text(px + 1.2, 75.2, "Zero-GT", fontsize=8, color=C_GREEN, fontweight="bold")

    _box(ax, px, 59, pw, 12, "Pillar 2 · Extrinsic 3D Math", "#12243a",
         sub="All-Pairs Rel-Pose mAA + Normalized\nCamera ATE + Voxel 3D F1@10cm +\n"
             "Z-Buffered Depth & Gradient Loss",
         ec=C_BLUE, fontsize=10)
    ax.text(px + 1.2, 58.2, "Sim(3)-Aligned", fontsize=8, color=C_BLUE, fontweight="bold")

    _box(ax, px, 42, pw, 12, "Pillar 3 · Visual 3D Losses", "#33182c",
         sub="3-View Ortho (BEV/Front/Side) Occ.\nIoU & RGB/Depth + Point-Splat Render\n"
             "(Masked SSIM + L1) + Patch ZNCC",
         ec=C_PINK, fontsize=10)
    ax.text(px + 1.2, 41.2, "Render-and-Compare", fontsize=8, color=C_PINK,
            fontweight="bold")

    _box(ax, px, 25, pw, 12, "Pillar 4 · Gated Efficiency", "#33240f",
         sub="Multiplicative efficiency & doppel-\nganger F1 gated by R_recon —\n"
             "no raw tool-call inflation",
         ec=C_ORANGE, fontsize=10)
    ax.text(px + 1.2, 24.2, "Tie-Breaker", fontsize=8, color=C_ORANGE, fontweight="bold")

    _arrow(ax, (74.2, 69), (77.8, 82), color=C_GREEN, lw=1.8,
           connectionstyle="arc3,rad=-0.12")
    _arrow(ax, (74.2, 69), (77.8, 65), color=C_BLUE, lw=1.8)
    _arrow(ax, (74.2, 69), (77.8, 48), color=C_PINK, lw=1.8,
           connectionstyle="arc3,rad=0.12")
    ax.text(72.5, 79.5, "scores 3D model", fontsize=8.5, color=C_MUTED,
            ha="center", fontstyle="italic")

    _box(ax, 46, 12, 23, 13, "GRPO / S-GRPO\nCGI Advantage", "#3d1620",
         sub="A = R_recon · eff_gate − baseline", ec=C_RED, lw=2.0)
    for yy, col in ((82, C_GREEN), (65, C_BLUE), (48, C_PINK), (31, C_ORANGE)):
        _arrow(ax, (px, yy), (69.2, 19.5), color=col, lw=1.5,
               connectionstyle="arc3,rad=0.16")
    _arrow(ax, (45.8, 18.5), (26.5, 61.6), color=C_RED, lw=2.2,
           connectionstyle="arc3,rad=-0.22",
           label="policy gradient update", loffset=(-6, -3.0), lcolor=C_RED)

    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=FIG_BG)
    plt.close(fig)


# ===========================================================================
# Figure 2 — 3x4 demo grid on the synthetic L-room
# ===========================================================================
def _draw_bev(ax, pred_pts, metrics, title_prefix):
    ax.set_facecolor(PANEL_BG)
    # GT footprint outline
    poly = np.vstack([FLOOR_POLY, FLOOR_POLY[0]])
    ax.plot(poly[:, 0], poly[:, 1], "-", color=C_TEXT, lw=1.4, alpha=0.85,
            label="GT footprint")
    if len(pred_pts):
        ax.scatter(pred_pts[:, 0], pred_pts[:, 1], s=2.5, c="#9ecbff",
                   alpha=0.5, linewidths=0)
    ax.set_aspect("equal")
    ax.set_xlim(-0.8, 4.8); ax.set_ylim(-0.9, 5.8)
    ax.set_title(f"{title_prefix}\n$BEV$ occupancy IoU = "
                 f"{metrics['p3_ortho_bev_iou']:.2f}", fontsize=9.5,
                 fontweight="bold")
    ax.tick_params(labelsize=6.5)
    ax.legend(loc="upper right", fontsize=6.5, facecolor=PANEL_BG,
              edgecolor=EDGE, labelcolor=C_TEXT)


def _draw_3d(ax, pred_pts, pred_cols, pred_poses, metrics):
    ax.set_facecolor(PANEL_BG)
    if len(pred_pts):
        ax.scatter(pred_pts[:, 0], pred_pts[:, 1], pred_pts[:, 2],
                   c=np.clip(pred_cols, 0, 1), s=1.5, alpha=0.85, linewidths=0)
    for i, M in pred_poses.items():
        c = -M[:3, :3].T @ M[:3, 3]
        ax.scatter(*c, c=[C_ORANGE], s=45, marker="^", edgecolors="k",
                   linewidths=0.3)
        f = M[:3, :3].T[:, 2]
        ax.plot([c[0], c[0] + 0.7 * f[0]], [c[1], c[1] + 0.7 * f[1]],
                [c[2], c[2] + 0.7 * f[2]], color=C_ORANGE, lw=0.9, alpha=0.8)
    ax.set_title(f"3D cloud + frustums\n$F1_{{3D}}$ = {metrics['p2_f1_3d']:.2f} · "
                 f"mAA = {metrics['p2_all_pairs_maa']:.2f}", fontsize=9.5,
                 fontweight="bold")
    ax.view_init(elev=28, azim=-58)
    ax.set_xlim(-0.8, 4.8); ax.set_ylim(-0.9, 5.8); ax.set_zlim(0, 3.2)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(PANEL_BG)
        axis.pane.set_edgecolor(EDGE)
        axis._axinfo["grid"]["color"] = (0.3, 0.3, 0.35, 0.3)
    ax.tick_params(labelsize=6)


def _draw_splat(ax, pred_pts, pred_cols, view_idx, metrics):
    ax.set_facecolor(PANEL_BG)
    out_p = render_pointcloud_splat(pred_pts, pred_cols, K_MAT, GT_POSES[view_idx],
                                    width=IMG_SIZE, height=IMG_SIZE,
                                    splat_radius_px=3)
    gt = GT_IMAGES[view_idx]
    combo = np.concatenate([out_p["rgb"], gt], axis=1)
    ax.imshow(np.clip(combo, 0, 1))
    ax.axvline(IMG_SIZE - 0.5, color=C_TEXT, lw=0.8, alpha=0.6)
    ax.text(IMG_SIZE * 0.5, IMG_SIZE - 8, "Splat render", ha="center", va="bottom",
            fontsize=7.5, fontweight="bold", color=C_TEXT,
            bbox={"facecolor": FIG_BG, "alpha": 0.75, "pad": 1.5, "edgecolor": "none"})
    ax.text(IMG_SIZE * 1.5, IMG_SIZE - 8, "GT view", ha="center", va="bottom",
            fontsize=7.5, fontweight="bold", color=C_TEXT,
            bbox={"facecolor": FIG_BG, "alpha": 0.75, "pad": 1.5, "edgecolor": "none"})
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Point-splat render (view {view_idx})\n"
                 f"SSIM = {metrics['p3_splat_ssim']:.2f} · ZNCC reward = "
                 f"{metrics['p3_patch_consistency_reward']:.2f}",
                 fontsize=9.5, fontweight="bold")


def _draw_bars(ax, metrics):
    ax.set_facecolor(PANEL_BG)
    vals = [metrics["legacy"], metrics["new_reward"]]
    bars = ax.barh([1, 0], vals, height=0.55,
                   color=[C_MUTED, C_TEAL], edgecolor=EDGE)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["Legacy Proxy\nReward", "4-Pillar\nRecon Reward"],
                       fontsize=7.5, fontweight="bold")
    for b, v in zip(bars, vals):
        ax.text(v + 0.02, b.get_y() + b.get_height() / 2, f"{v:.2f}",
                va="center", fontsize=8.5, fontweight="bold", color=C_TEXT)
    ax.set_xlim(0, 1.12)
    ax.set_title("Reward verdict", fontsize=9.5, fontweight="bold")
    ax.tick_params(labelsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig2_demo(path: Path):
    scenarios = []
    # Row 1: accurate — tiny noise, all 8 cameras
    p1_pts = GT_PTS + RNG.normal(0, 0.01, GT_PTS.shape)
    scenarios.append(("Row 1 — Accurate Reconstruction (8/8 cams)",
                      p1_pts, GT_COLS, dict(GT_POSES), 2.0, 0))
    # Row 2: doppelganger symmetry fold — wing B + its cams rotated 180°
    p2_pts, p2_poses = fold_scene(GT_PTS, GT_POSES, math.pi)
    scenarios.append(("Row 2 — Doppelganger Symmetry Fold (8/8 cams, wall B folded 180°)",
                      p2_pts, GT_COLS, p2_poses, 38.0, 1))
    # Row 3: abstention — only the 3 cameras on wall A region + nearby points
    keep_cams = [0, 1, 3]
    # only the patch of leg-A wall/floor visible to the 3 easy cameras —
    # ~62% of the room is simply never triangulated
    vis_mask = ~FOLD_MASK & (GT_PTS[:, 1] < 1.3) & (GT_PTS[:, 0] < 3.2)
    scenarios.append(("Row 3 — Abstention Loophole (3/8 cams, 0° pose error)",
                      GT_PTS[vis_mask], GT_COLS[vis_mask],
                      {i: GT_POSES[i] for i in keep_cams}, 1.0, 2))

    fig = plt.figure(figsize=(22, 15.5))
    gs = fig.add_gridspec(3, 4, wspace=0.22, hspace=0.42)
    axes = np.empty((3, 4), dtype=object)
    for r in range(3):
        for c in range(4):
            kw = {"projection": "3d"} if c == 1 else {}
            axes[r, c] = fig.add_subplot(gs[r, c], **kw)
    view_idx = 2  # held-out-style registered camera view for splat column

    for r, (row_title, pts, cols, poses, cyc, row) in enumerate(scenarios):
        m = evaluate_prediction(pts, cols, poses, cyc, row)
        _draw_bev(axes[r, 0], pts, m, row_title.split("—")[1].strip()
                  if "—" in row_title else row_title)
        _draw_3d(axes[r, 1], pts, cols, poses, m)
        _draw_splat(axes[r, 2], pts, cols, view_idx if view_idx in poses else 0, m)
        _draw_bars(axes[r, 3], m)
        # row label on far left
        axes[r, 0].set_ylabel(row_title.split("—")[0].strip(), fontsize=10,
                              fontweight="bold", color=C_ORANGE)
        print(f"  {row_title}: legacy={m['legacy']:.3f} new={m['new_reward']:.3f} "
              f"(P1={m['pillar1']:.2f} P2={m['pillar2']:.2f} P3={m['pillar3']:.2f})")

    fig.suptitle("Mathematical & Visual Losses Catch What Legacy Proxies Miss\n"
                 "Legacy = registered-fraction + reprojection proxy ·  New = 4-Pillar R_recon "
                 "(Intrinsic + Extrinsic 3D + Visual) × efficiency gate",
                 fontsize=15, fontweight="bold", color=C_TEXT, y=0.995)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=FIG_BG)
    plt.close(fig)


# ===========================================================================
# Figure 3 — response curves
# ===========================================================================
def _style_ax(ax):
    ax.set_facecolor(PANEL_BG)
    ax.grid(True, color=EDGE, alpha=0.55, lw=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    leg = ax.legend(facecolor=PANEL_BG, edgecolor=EDGE, labelcolor=C_TEXT,
                    fontsize=9, prop={"weight": "bold"})
    if leg:
        leg.get_frame().set_alpha(0.9)


def _rigidity_curve(cycle_errs_deg, edges_builder):
    return [compute_graph_rigidity_metrics(
        N_CAMS + 2, edges_builder(e))["graph_rigidity_reward"]
        for e in cycle_errs_deg]


def _edges_with_l2(n_nodes, kind, err_deg):
    """Build a triangulated view graph on n_nodes with topology ``kind`` whose
    measured Fiedler λ₂ spans disconnected -> weak bridge -> ring -> dense mesh,
    and inject ``err_deg`` on chord edges so every 3-cycle has residual ``err_deg``."""
    pairs = []
    chords = set()
    if kind == "disconnected":
        pairs = [(0, 1), (1, 2), (0, 2), (4, 5), (5, 6), (4, 6)]
        chords = {(0, 2), (4, 6)}
    elif kind == "hinged-chain":
        pairs = [(i, i + 1) for i in range(n_nodes - 1)] + [(0, 2), (n_nodes - 3, n_nodes - 1)]
        chords = {(0, 2), (n_nodes - 3, n_nodes - 1)}
    elif kind == "triangulated-chain":
        pairs = [(i, i + 1) for i in range(n_nodes - 1)] + [(i, i + 2) for i in range(n_nodes - 2)]
        chords = {(i, i + 2) for i in range(n_nodes - 2)}
    elif kind == "triangulated-ring":
        pairs = (
            [(i, (i + 1) % n_nodes) for i in range(n_nodes)]
            + [(i, (i + 2) % n_nodes) for i in range(n_nodes)]
        )
        chords = {(i, (i + 2) % n_nodes) for i in range(n_nodes)}
    edges = []
    for u, v in pairs:
        R = _rotz(math.radians(err_deg)) if (u, v) in chords and err_deg > 0 else np.eye(3)
        edges.append((u, v, R))
    return edges


def fig3_curves(path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 11),
                             gridspec_kw={"wspace": 0.24, "hspace": 0.34})

    # Panel A — intrinsic_ba_reward vs reproj error for track lengths
    ax = axes[0, 0]
    e_px = np.linspace(0, 6, 121)
    for tl, col in zip([2, 3, 4, 6, 8],
                       [C_MUTED, C_BLUE, C_TEAL, C_ORANGE, C_GREEN]):
        r = [compute_intrinsic_ba_metrics(np.full(200, e),
             np.full(200, tl))["intrinsic_ba_reward"] for e in e_px]
        ax.plot(e_px, r, color=col, lw=2.4, label=f"$|T_k|$ = {tl}")
    ax.set_title("A · intrinsic_ba_reward vs. reprojection error $e_{px}$\n"
                 "(2-view tracks carry zero rigidity weight)",
                 fontsize=11.5, fontweight="bold")
    ax.set_xlabel("mean reprojection error (px)", fontweight="bold")
    ax.set_ylabel("reward", fontweight="bold")
    ax.set_ylim(-0.02, 1.05)
    _style_ax(ax)

    # Panel B — graph_rigidity_reward vs 3-cycle residual for λ2 values
    ax = axes[0, 1]
    cyc = np.linspace(0, 45, 91)
    for kind, n, col in [
        ("disconnected", 8, C_MUTED),
        ("hinged-chain", 12, C_RED),
        ("triangulated-chain", 11, C_BLUE),
        ("triangulated-ring", 9, C_GREEN),
    ]:
        l2 = compute_graph_rigidity_metrics(
            n, _edges_with_l2(n, kind, 0.0))["fiedler_lambda2"]
        r = [compute_graph_rigidity_metrics(
            n, _edges_with_l2(n, kind, e))["graph_rigidity_reward"]
            for e in cyc]
        ax.plot(cyc, r, color=col, lw=2.4,
                label=f"$\\lambda_2$ = {l2:.2f} ({kind})")
    ax.set_title("B · graph_rigidity_reward vs. 3-cycle rotation residual\n"
                 "(Fiedler connectivity gate × exp(−L_cycle/15°))",
                 fontsize=11.5, fontweight="bold")
    ax.set_xlabel("mean 3-cycle residual $L_{cycle}$ (deg)", fontweight="bold")
    ax.set_ylabel("reward", fontweight="bold")
    ax.set_ylim(-0.02, 1.05)
    _style_ax(ax)

    # Panel C — abstention penalty: mAA & ATE vs registered fraction
    ax = axes[1, 0]
    ks = np.arange(2, N_CAMS + 1)
    maa_v, ate_v, legacy_v = [], [], []
    for k in ks:
        sub = {i: GT_POSES[i] for i in range(k)}
        maa_v.append(compute_all_pairs_pose_maa(
            sub, GT_POSES, num_total_images=N_CAMS)["all_pairs_maa"])
        if k >= 3:
            ate_v.append(align_poses_and_points_sim3(
                sub, GT_POSES, GT_PTS)["ate_reward"])
        else:
            ate_v.append(0.0)
        # legacy: accuracy over registered pairs only (abstention hidden)
        maa_reg = compute_all_pairs_pose_maa(sub, {i: GT_POSES[i] for i in sub},
                                             num_total_images=k)["all_pairs_maa"]
        legacy_v.append(maa_reg)
    ax.plot(ks / N_CAMS, maa_v, "o-", color=C_BLUE, lw=2.6,
            label="all_pairs_maa (unreg → 180°)")
    ax.plot(ks / N_CAMS, ate_v, "s-", color=C_GREEN, lw=2.6,
            label="ate_reward (coverage × e$^{-ATE}$)")
    ax.plot(ks / N_CAMS, legacy_v, "^--", color=C_MUTED, lw=2.2,
            label="legacy registered-only accuracy")
    ax.set_title("C · Abstention penalty vs. registered fraction $k/N$\n"
                 "(all-pairs scoring makes dropping hard views quadratic-costly)",
                 fontsize=11.5, fontweight="bold")
    ax.set_xlabel("registered cameras $k/N$", fontweight="bold")
    ax.set_ylabel("reward", fontweight="bold")
    ax.set_ylim(-0.02, 1.05)
    _style_ax(ax)

    # Panel D — visual & 3D geometric sensitivity vs fold angle
    ax = axes[1, 1]
    thetas = np.linspace(0, 90, 10)
    f1_v, iou_v, dep_v, spl_v = [], [], [], []
    for th in thetas:
        pts_f, _ = fold_scene(GT_PTS, GT_POSES, math.radians(th))
        f1_v.append(compute_pointcloud_f1(pts_f, GT_PTS)["f1_3d"])
        iou_v.append(compute_orthographic_3d_render_loss(
            pts_f, GT_COLS, GT_PTS, GT_COLS)["ortho_bev_iou"])
        dep_v.append(compute_projected_depth_loss(
            pts_f, GT_POSES, {i: K_MAT for i in GT_POSES},
            GT_DEPTHS)["depth_consistency_reward"])
        spl_v.append(compute_splat_render_loss(
            pts_f, GT_COLS, GT_POSES, {i: K_MAT for i in GT_POSES},
            GT_IMAGES)["splat_visual_reward"])
    ax.plot(thetas, f1_v, "o-", color=C_GREEN, lw=2.5, label="$F1_{3D}$@10cm")
    ax.plot(thetas, iou_v, "s-", color=C_BLUE, lw=2.5, label="BEV occupancy IoU")
    ax.plot(thetas, dep_v, "d-", color=C_ORANGE, lw=2.5,
            label="depth_consistency_reward")
    ax.plot(thetas, spl_v, "^-", color=C_PINK, lw=2.5, label="splat_visual_reward")
    ax.set_title("D · Visual & 3D losses vs. doppelganger fold angle $\\theta_{fold}$\n"
                 "(wing B rotated about its hinge — real metric values)",
                 fontsize=11.5, fontweight="bold")
    ax.set_xlabel("fold angle $\\theta_{fold}$ (deg)", fontweight="bold")
    ax.set_ylabel("reward", fontweight="bold")
    ax.set_ylim(-0.02, 1.05)
    _style_ax(ax)

    fig.suptitle("Reconstruction-Loss Response Surfaces "
                 "(computed with agentic_sfm.rewards.reconstruction_loss)",
                 fontsize=15, fontweight="bold", color=C_TEXT)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=FIG_BG)
    plt.close(fig)


# ===========================================================================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"GT cloud: {len(GT_PTS)} pts | {N_CAMS} cams | "
          f"fold pts: {int(FOLD_MASK.sum())}")
    print("[1/3] architecture diagram …")
    fig1_architecture(OUT_DIR / "reconstruction_pipeline_architecture.png")
    print("[2/3] mathematical vs visual losses demo …")
    fig2_demo(OUT_DIR / "mathematical_vs_visual_losses_demo.png")
    print("[3/3] loss response curves …")
    fig3_curves(OUT_DIR / "loss_response_curves.png")
    for name in ("reconstruction_pipeline_architecture.png",
                 "mathematical_vs_visual_losses_demo.png",
                 "loss_response_curves.png"):
        p = OUT_DIR / name
        print(f"  wrote {p} ({p.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
