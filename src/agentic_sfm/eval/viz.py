"""Publication-quality visualisations for the Agentic-SfM agent.

Produces the figures you can screenshot for an X post / paper figure:

- ``draw_match_pair`` — two images side-by-side with inlier (green) and
  outlier (red) correspondence lines drawn between them.
- ``render_point_cloud`` — matplotlib 3D render of a COLMAP sparse
  reconstruction (coloured points + camera frusta).
- ``plot_reward_breakdown`` — horizontal bar chart of reward components.
- ``plot_training_curves`` — reward / loss curves over training steps.
- ``episode_figure`` — one combined figure for a single episode.

Everything returns a matplotlib ``Figure`` (Agg backend, headless-safe) so
callers can ``wandb.Image(fig)`` or ``fig.savefig(path)``.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless — no display on compute nodes
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: F401  (3d axes side-effect)

from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# colour palette (matches the "RL / agent" aesthetic)
# ---------------------------------------------------------------------------
C_INLIER = "#2ecc71"   # green
C_OUTLIER = "#e74c3c"  # red
C_ACCENT = "#3498db"   # blue
C_CAMERA = "#f39c12"   # orange
C_TEXT = "#ecf0f1"
FIG_BG = "#0d1117"     # dark github-style background


def _to_pil(img: Any) -> Image.Image:
    """Accept PIL.Image / np.ndarray / path / base64 and return RGB PIL."""
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, np.ndarray):
        return Image.fromarray(img.astype(np.uint8)).convert("RGB")
    if isinstance(img, (str, Path)):
        return Image.open(img).convert("RGB")
    if isinstance(img, (bytes, bytearray)):
        return Image.open(io.BytesIO(img)).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(img)}")


def _decode_b64(b64: str) -> Image.Image:
    import base64
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


# ---------------------------------------------------------------------------
# Match visualisation
# ---------------------------------------------------------------------------
def draw_match_pair(
    img_a: Any,
    img_b: Any,
    kpts_a: np.ndarray,
    kpts_b: np.ndarray,
    inlier_mask: np.ndarray | None = None,
    title: str = "",
    max_draw: int = 80,
) -> plt.Figure:
    """Two images side-by-side with correspondence lines.

    ``inlier_mask`` (bool, len N) colours lines green (inlier) / red (outlier).
    If None, all lines are drawn accent-blue. Keypoints are in *image* pixels.
    """
    ia, ib = _to_pil(img_a), _to_pil(img_b)
    wa, ha = ia.size
    wb, hb = ib.size
    H, W = max(ha, hb), wa + wb

    fig, ax = plt.subplots(figsize=(12, 6), dpi=130)
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(FIG_BG)

    canvas = Image.new("RGB", (W, H))
    canvas.paste(ia, (0, 0))
    canvas.paste(ib, (wa, 0))
    ax.imshow(np.asarray(canvas))
    ax.axis("off")

    kpts_a = np.asarray(kpts_a, dtype=float)
    kpts_b = np.asarray(kpts_b, dtype=float)
    n = min(len(kpts_a), len(kpts_b), max_draw)
    if n > 0:
        # subsample for readability
        idx = np.linspace(0, min(len(kpts_a), len(kpts_b)) - 1, n).astype(int)
        for j, k in enumerate(idx):
            xa, ya = kpts_a[k]
            xb, yb = kpts_b[k] + np.array([wa, 0.0])
            if inlier_mask is not None and len(inlier_mask) > k:
                col = C_INLIER if inlier_mask[k] else C_OUTLIER
                alpha = 0.9 if inlier_mask[k] else 0.35
            else:
                col, alpha = C_ACCENT, 0.75
            ax.plot([xa, xb], [ya, yb], "-", color=col, lw=1.0, alpha=alpha)
            ax.plot(xa, ya, "o", color=col, ms=2.5, alpha=alpha)
            ax.plot(xb, yb, "o", color=col, ms=2.5, alpha=alpha)

    n_inl = int(inlier_mask.sum()) if inlier_mask is not None else len(kpts_a)
    legend = [
        Line2D([0], [0], color=C_INLIER, lw=2, label=f"inliers ({n_inl})"),
        Line2D([0], [0], color=C_OUTLIER, lw=2, label="outliers"),
    ]
    leg = ax.legend(handles=legend, loc="lower right", facecolor="#161b22",
                    edgecolor="#30363d", labelcolor=C_TEXT, fontsize=10,
                    prop={"weight": "bold"})
    if title:
        ax.set_title(title, color=C_TEXT, fontsize=13, fontweight="bold", pad=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Reconstruction (point cloud + cameras)
# ---------------------------------------------------------------------------
def render_point_cloud(
    model_dir: str | Path | None = None,
    reconstruction: Any = None,
    max_points: int = 40000,
    title: str = "Sparse reconstruction",
    elev: float = 20.0,
    azim: float = -60.0,
) -> plt.Figure:
    """Render a COLMAP/pycolmap sparse model: coloured 3D points + camera frusta.

    Pass either ``model_dir`` (a ``sparse/0`` dir or a ``Reconstruction`` via
    ``reconstruction=``). Falls back to a graceful placeholder on failure.
    """
    import pycolmap

    recon = reconstruction
    if recon is None:
        if model_dir is None:
            raise ValueError("Need model_dir or reconstruction")
        recon = pycolmap.Reconstruction(str(model_dir))

    # --- points ---
    pts, cols = [], []
    for _pid, p in recon.points3D.items():
        pts.append(p.xyz)
        cols.append(np.asarray(p.color) / 255.0)
    pts = np.asarray(pts) if pts else np.zeros((0, 3))
    cols = np.asarray(cols) if cols else np.zeros((0, 3))

    # --- camera centres + viewing directions ---
    cam_centers, cam_fwds = [], []
    for _iid, im in recon.images.items():
        try:
            c = np.asarray(im.projection_center())  # world-space centre
            cam_centers.append(c)
            # forward (viewing) direction: rotation's +Z axis in world coords
            R = np.asarray(im.cam_from_world.rotation.matrix())
            cam_fwds.append(R[2])  # 3rd row = world-space forward
        except Exception:
            cam_fwds.append(np.zeros(3))

    fig = plt.figure(figsize=(9, 8), dpi=140)
    fig.patch.set_facecolor(FIG_BG)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(FIG_BG)

    if len(pts):
        if len(pts) > max_points:
            sel = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
            pts_p, cols_p = pts[sel], cols[sel]
        else:
            pts_p, cols_p = pts, cols
        # depth-shade + slightly larger points read better in a sparse cloud
        ax.scatter(pts_p[:, 0], pts_p[:, 2], -pts_p[:, 1],
                   c=np.clip(cols_p, 0, 1), s=5.5, alpha=0.95,
                   linewidths=0, depthshade=True)

    if cam_centers:
        cc = np.asarray(cam_centers)
        # camera frustum cones: small pyramid from centre along forward dir
        scale = max(np.ptp(pts, axis=0).max() * 0.06, 0.5) if len(pts) else 1.0
        segs = []
        for c, f in zip(cam_centers, cam_fwds):
            f = f / (np.linalg.norm(f) + 1e-9)
            tip = c + f * scale
            # 4 base corners of a tiny pyramid around the tip
            up = np.array([0, 0, 1.0])
            right = np.cross(f, up); right /= (np.linalg.norm(right) + 1e-9)
            up2 = np.cross(right, f); up2 /= (np.linalg.norm(up2) + 1e-9)
            r = scale * 0.35
            base = [tip + right*r + up2*r, tip + right*r - up2*r,
                    tip - right*r - up2*r, tip - right*r + up2*r]
            for b in base:
                segs.append([c, b])
            segs += [[base[i], base[(i+1) % 4]] for i in range(4)]
        if segs:
            segs = np.asarray(segs)
            lc = Line3DCollection(
                [[ [s[0][0], s[0][2], -s[0][1]], [s[1][0], s[1][2], -s[1][1]] ]
                 for s in segs],
                colors=C_CAMERA, linewidths=1.1, alpha=0.9)
            ax.add_collection3d(lc)
        ax.scatter(cc[:, 0], cc[:, 2], -cc[:, 1],
                   c=C_CAMERA, s=55, marker="^", edgecolors="k",
                   linewidths=0.4, label="cameras", depthshade=False)

    npts = len(pts)
    ax.set_title(
        f"{title}\n{len(recon.images)} registered images · {npts:,} 3D points",
        color=C_TEXT, fontsize=13, fontweight="bold",
    )
    # Auto-frame around the point cloud ( tighter "wow" shot )
    if len(pts):
        mid = np.median(pts, axis=0)
        rng = np.ptp(pts, axis=0).max()
        ax.set_xlim(mid[0] - rng * 0.6, mid[0] + rng * 0.6)
        ax.set_ylim(mid[2] - rng * 0.6, mid[2] + rng * 0.6)
        ax.set_zlim(-mid[1] - rng * 0.6, -mid[1] + rng * 0.6)
    ax.view_init(elev=elev, azim=azim)

    # tidy dark 3d panes
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(FIG_BG)
        axis.pane.set_edgecolor("#30363d")
        axis.label.set_color(C_TEXT)
        axis._axinfo["grid"]["color"] = (0.3, 0.3, 0.35, 0.3)
    ax.tick_params(colors=C_TEXT, labelsize=7)
    ax.set_xlabel("X", color=C_TEXT); ax.set_ylabel("Z", color=C_TEXT)
    ax.set_zlabel("-Y", color=C_TEXT)
    try:
        if len(pts):
            ax.set_box_aspect((np.ptp(pts[:,0])+1e-6, np.ptp(pts[:,2])+1e-6,
                               np.ptp(pts[:,1])+1e-6))
    except Exception:
        pass
    if cam_centers:
        ax.legend(facecolor="#161b22", edgecolor="#30363d",
                  labelcolor=C_TEXT, fontsize=10, prop={"weight": "bold"})
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Reward breakdown + training curves
# ---------------------------------------------------------------------------
def plot_reward_breakdown(components: dict[str, float], title: str = "Reward breakdown") -> plt.Figure:
    """Horizontal bar chart of per-component reward contributions."""
    items = [(k, v) for k, v in components.items()
             if isinstance(v, (int, float)) and k != "total_reward" and abs(v) > 1e-9]
    items.sort(key=lambda kv: kv[1])
    labels = [k.replace("_", " ") for k, _ in items]
    vals = [v for _, v in items]
    colors = [C_INLIER if v >= 0 else C_OUTLIER for v in vals]

    fig, ax = plt.subplots(figsize=(7.5, max(2.5, 0.5 * len(items) + 1)), dpi=140)
    fig.patch.set_facecolor(FIG_BG); ax.set_facecolor(FIG_BG)
    ax.barh(labels, vals, color=colors, edgecolor="#30363d")
    for i, v in enumerate(vals):
        ax.text(v + (0.02 if v >= 0 else -0.02), i, f"{v:+.2f}",
                va="center", ha="left" if v >= 0 else "right",
                color=C_TEXT, fontsize=9, fontweight="bold")
    ax.axvline(0, color="#30363d", lw=1)
    ax.set_title(title, color=C_TEXT, fontsize=12, fontweight="bold")
    ax.tick_params(colors=C_TEXT)
    for s in ax.spines.values():
        s.set_color("#30363d")
    fig.tight_layout()
    return fig


def plot_training_curves(
    rewards: list[float],
    losses: list[float] | None = None,
    title: str = "GRPO training",
) -> plt.Figure:
    """Reward (+ optional loss) curve over optimisation steps."""
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=140)
    fig.patch.set_facecolor(FIG_BG); ax.set_facecolor(FIG_BG)
    xs = np.arange(1, len(rewards) + 1)
    ax.plot(xs, rewards, "-o", color=C_ACCENT, lw=2, ms=4, label="reward")
    if len(rewards) > 3:
        # light moving-average trendline
        k = max(2, len(rewards) // 8)
        ma = np.convolve(rewards, np.ones(k) / k, mode="valid")
        ax.plot(np.arange(k, len(rewards) + 1), ma, "-", color=C_INLIER,
                lw=2.5, alpha=0.9, label="trend")
    ax.set_xlabel("update step", color=C_TEXT)
    ax.set_ylabel("mean reward", color=C_ACCENT)
    ax.tick_params(colors=C_TEXT)
    for s in ax.spines.values():
        s.set_color("#30363d")
    if losses:
        ax2 = ax.twinx()
        ax2.plot(xs[: len(losses)], losses, "-s", color=C_CAMERA, lw=1.4,
                 ms=3, alpha=0.7, label="loss")
        ax2.set_ylabel("loss", color=C_CAMERA)
        ax2.tick_params(colors=C_TEXT)
    ax.set_title(title, color=C_TEXT, fontsize=13, fontweight="bold")
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor=C_TEXT,
              prop={"weight": "bold"})
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Convenience: decode an episode's stored b64 images
# ---------------------------------------------------------------------------
def episode_images(ep: Any) -> list[Image.Image]:
    """Decode the b64 image list stored on a rollout episode."""
    out = []
    for item in getattr(ep, "images", []) or []:
        try:
            out.append(_decode_b64(item) if isinstance(item, str) else _to_pil(item))
        except Exception:
            continue
    return out


def match_result_figure(ep: Any, match: dict[str, Any], title: str | None = None) -> plt.Figure | None:
    """Build a match-pair figure from a rollout episode + one match result.

    Uses the episode's first two stored images as the pair and the result's
    ``keypoints_a/b`` + ``inlier_mask``. Returns None if not drawable.
    """
    imgs = episode_images(ep)
    if len(imgs) < 2:
        return None
    ka = np.asarray(match.get("keypoints_a") or [], dtype=float)
    kb = np.asarray(match.get("keypoints_b") or [], dtype=float)
    if len(ka) == 0 or len(kb) == 0:
        return None
    mask = match.get("inlier_mask")
    mask = np.asarray(mask, dtype=bool) if mask else None
    n_inl = match.get("num_inliers", int(mask.sum()) if mask is not None else len(ka))
    t = title or (
        f"{getattr(ep, 'pair_id', getattr(ep, 'scene_id', 'episode'))} · "
        f"{match.get('matcher', 'matcher')} · {n_inl} inliers "
        f"({match.get('inlier_ratio', 0.0):.0%}) · reward {getattr(ep, 'reward', 0.0):.2f}"
    )
    return draw_match_pair(imgs[0], imgs[1], ka, kb, inlier_mask=mask, title=t)


def episode_qualitative_media(
    ep: Any,
    prefix: str = "qual",
) -> dict[str, Any]:
    """Build a {wandb_key: figure} dict for one episode's best match.

    Picks ``ep.final_match`` (or the result with most inliers) and renders the
    correspondence figure. Returns {} if nothing is drawable.
    """
    match = getattr(ep, "final_match", None)
    if not match:
        cands = [r for r in (getattr(ep, "results", []) or [])
                 if isinstance(r, dict) and (r.get("keypoints_a") or r.get("matches"))]
        if cands:
            match = max(cands, key=lambda r: r.get("num_inliers", 0))
    if not match:
        return {}
    fig = match_result_figure(ep, match)
    if fig is None:
        return {}
    return {f"{prefix}/best_match": fig}

