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
# Dense GT point cloud (ScanNet-style depth + pose -> world-space cloud)
# ---------------------------------------------------------------------------
def _load_intrinsic(path: Path) -> np.ndarray:
    """Read a 3x3 or 4x4 whitespace-separated intrinsics matrix -> 3x3 K."""
    K = np.loadtxt(str(path), dtype=np.float64)
    if K.shape == (4, 4):
        K = K[:3, :3]
    return K


def gt_pointcloud_arrays(
    depth_dir: str | Path,
    color_dir: str | Path,
    pose_dir: str | Path,
    intrinsic_path: str | Path,
    max_points: int = 200000,
    max_depth: float = 10.0,
    depth_scale: float = 1000.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Back-project RGB-D frames into a world-frame coloured point cloud.

    For every colour frame with a matching depth map + pose file:

      1. unproject valid depth pixels (``depth / depth_scale`` metres, masked
         to ``(0, max_depth]``) through the depth intrinsics into camera space
      2. transform camera-space points into world space with the frame's
         4x4 **camera-to-world** pose (the ScanNet ``pose/*.txt``
         convention)
      3. colour each point by projecting it into the colour image using
         ``intrinsic_color.txt`` + ``extrinsic_color.txt`` discovered next to
         ``intrinsic_path``; when those are absent the colour image is
         resized to depth resolution and indexed directly.

    Returns ``(points (N,3) float64, colours (N,3) float64 in [0,1],
    camera_centres (M,3))``.  ``points`` is randomly subsampled to
    ``max_points``; frames are sub-strided so the budget is spread evenly.
    """
    depth_dir = Path(depth_dir)
    color_dir = Path(color_dir)
    pose_dir = Path(pose_dir)
    intrinsic_path = Path(intrinsic_path)

    K_depth = _load_intrinsic(intrinsic_path)

    # Optional colour calibration discovered alongside the depth intrinsics
    # (``intrinsic/`` dir in the ScanNet layout).
    K_color = None
    T_color_from_depth = None
    for cand in (intrinsic_path.parent / "intrinsic_color.txt",):
        if cand.exists():
            try:
                K_color = _load_intrinsic(cand)
            except Exception:
                K_color = None
    e_path = intrinsic_path.parent / "extrinsic_color.txt"
    if e_path.exists():
        try:
            # ScanNet convention: extrinsic_color maps *depth* camera
            # coordinates to *color* camera coordinates, so a depth-camera
            # point projects into the colour image as
            # ``K_color @ (E @ p_depth)``.
            T_color_from_depth = np.loadtxt(str(e_path), dtype=np.float64)
            if T_color_from_depth.shape != (4, 4):
                T_color_from_depth = None
        except Exception:
            T_color_from_depth = None

    frames = []
    for img in sorted(color_dir.glob("*.jpg"),
                      key=lambda p: int(p.stem) if p.stem.isdigit() else 0):
        stem = img.stem
        d_path, p_path = depth_dir / f"{stem}.png", pose_dir / f"{stem}.txt"
        if d_path.exists() and p_path.exists():
            frames.append((img, d_path, p_path))
    if not frames:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3))

    # Per-frame pixel stride so the total lands near max_points.
    try:
        n_px = int(np.asarray(
            Image.open(frames[0][1])).size) or 480 * 640
    except Exception:
        n_px = 480 * 640
    stride = max(1, int(np.sqrt(n_px * len(frames) / max(max_points, 1))))

    all_pts, all_cols, cam_centers = [], [], []
    for img_path, d_path, p_path in frames:
        try:
            depth = np.asarray(Image.open(d_path), dtype=np.float64) / depth_scale
            c2w = np.loadtxt(str(p_path), dtype=np.float64).reshape(4, 4)
        except Exception as e:
            logger.warning(f"gt cloud: skipping {d_path.name}: {e}")
            continue
        if not np.isfinite(c2w).all():
            continue
        cam_centers.append(c2w[:3, 3])

        H, W = depth.shape
        color_img = np.asarray(_to_pil(img_path), dtype=np.float64) / 255.0

        ys, xs = np.mgrid[0:H:stride, 0:W:stride]
        z = depth[ys, xs]
        mask = (z > 0) & (z <= max_depth) & np.isfinite(z)
        if not mask.any():
            continue
        xs, ys, z = xs[mask].astype(np.float64), ys[mask].astype(np.float64), z[mask]

        fx, fy, cx, cy = K_depth[0, 0], K_depth[1, 1], K_depth[0, 2], K_depth[1, 2]
        x_cam = (xs - cx) * z / fx
        y_cam = (ys - cy) * z / fy
        pts_cam = np.stack([x_cam, y_cam, z], axis=0)          # (3, M)
        pts_w = (c2w[:3, :3] @ pts_cam).T + c2w[:3, 3]         # (M, 3)

        # --- colours ---
        if K_color is not None and T_color_from_depth is not None:
            pts_col = (T_color_from_depth[:3, :3] @ pts_cam).T + \
                T_color_from_depth[:3, 3]
            zc = np.clip(pts_col[:, 2], 1e-6, None)
            u = K_color[0, 0] * pts_col[:, 0] / zc + K_color[0, 2]
            v = K_color[1, 1] * pts_col[:, 1] / zc + K_color[1, 2]
        else:
            # No colour calibration: rescale colour to depth resolution.
            ch, cw = color_img.shape[:2]
            u = xs * (cw / W)
            v = ys * (ch / H)
        ui = np.clip(np.round(u).astype(int), 0, color_img.shape[1] - 1)
        vi = np.clip(np.round(v).astype(int), 0, color_img.shape[0] - 1)
        all_pts.append(pts_w)
        all_cols.append(color_img[vi, ui])

    if not all_pts:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.asarray(cam_centers)
    pts = np.concatenate(all_pts, axis=0)
    cols = np.concatenate(all_cols, axis=0)
    cams = np.asarray(cam_centers) if cam_centers else np.zeros((0, 3))

    if len(pts) > max_points:
        sel = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
        pts, cols = pts[sel], cols[sel]
    return pts, cols, cams


def render_gt_pointcloud(
    depth_dir: str | Path,
    color_dir: str | Path,
    pose_dir: str | Path,
    intrinsic_path: str | Path,
    scene_id: str = "scene",
    max_points: int = 200000,
    title: str | None = None,
    elev: float = 20.0,
    azim: float = -60.0,
) -> plt.Figure:
    """Render a dense GT point cloud from depth maps + camera-to-world poses.

    Same dark 3D style as ``render_point_cloud`` (``FIG_BG`` background,
    ``C_TEXT`` labels, orange camera markers).  This is the "ground truth
    reconstruction" figure — far denser than a COLMAP sparse model.

    ``intrinsic_path`` should point at ``intrinsic_depth.txt``;
    ``intrinsic_color.txt``/``extrinsic_color.txt`` are auto-discovered in
    the same directory for colour projection.
    """
    pts, cols, cam_centers = gt_pointcloud_arrays(
        depth_dir, color_dir, pose_dir, intrinsic_path,
        max_points=max_points,
    )

    fig = plt.figure(figsize=(9, 8), dpi=140)
    fig.patch.set_facecolor(FIG_BG)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(FIG_BG)

    if len(pts):
        ax.scatter(pts[:, 0], pts[:, 2], -pts[:, 1],
                   c=np.clip(cols, 0, 1), s=1.2, alpha=0.9,
                   linewidths=0, depthshade=True)

    if len(cam_centers):
        ax.scatter(cam_centers[:, 0], cam_centers[:, 2], -cam_centers[:, 1],
                   c=C_CAMERA, s=45, marker="^", edgecolors="k",
                   linewidths=0.4, label="cameras", depthshade=False)

    n_frames = len(cam_centers)
    ax.set_title(
        f"{title or f'{scene_id} — dense GT reconstruction'}\n"
        f"{n_frames} frames · {len(pts):,} points",
        color=C_TEXT, fontsize=13, fontweight="bold",
    )
    if len(pts):
        mid = np.median(pts, axis=0)
        rng = np.ptp(pts, axis=0).max()
        ax.set_xlim(mid[0] - rng * 0.6, mid[0] + rng * 0.6)
        ax.set_ylim(mid[2] - rng * 0.6, mid[2] + rng * 0.6)
        ax.set_zlim(-mid[1] - rng * 0.6, -mid[1] + rng * 0.6)
    ax.view_init(elev=elev, azim=azim)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(FIG_BG)
        axis.pane.set_edgecolor("#30363d")
        axis.label.set_color(C_TEXT)
        axis._axinfo["grid"]["color"] = (0.3, 0.3, 0.35, 0.3)
    ax.tick_params(colors=C_TEXT, labelsize=7)
    ax.set_xlabel("X", color=C_TEXT)
    ax.set_ylabel("Z", color=C_TEXT)
    ax.set_zlabel("-Y", color=C_TEXT)
    try:
        if len(pts):
            ax.set_box_aspect((np.ptp(pts[:, 0]) + 1e-6,
                               np.ptp(pts[:, 2]) + 1e-6,
                               np.ptp(pts[:, 1]) + 1e-6))
    except Exception:
        pass
    if len(cam_centers):
        ax.legend(facecolor="#161b22", edgecolor="#30363d",
                  labelcolor=C_TEXT, fontsize=10, prop={"weight": "bold"})
    fig.tight_layout()
    # Handy for callers/tests that need the actual cloud size.
    fig._gt_point_count = int(len(pts))  # type: ignore[attr-defined]
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

