"""Mathematical and visual 3D reconstruction losses for Agentic SfM.

Implements the four-pillar reconstruction loss & reward stack described in
``docs/reconstruction_losses.md``:

1. **Intrinsic Mathematical Losses** (no GT required):
   - Track-conditioned robust Huber bundle-adjustment reprojection loss
   - Track length & multi-view rigidity statistics (|T_k| >= 3)
   - View-graph 3-cycle geodesic rotation consistency & Fiedler algebraic
     connectivity (lambda_2 of the normalized graph Laplacian)

2. **Extrinsic Mathematical 3D Losses** (against GT poses, depth, & 3D cloud):
   - All-pairs relative pose mAA@{5°,10°,20°} (unregistered cameras penalized
     as 180° error so abstention cannot game AUC)
   - Sim(3)-aligned camera trajectory normalized Huber ATE
   - Projected 3D-point-to-sensor-depth L1 + VGGT-style spatial depth-gradient
     consistency loss
   - Bidirectional truncated Chamfer distance & voxel-normalized 3D F-score
     (F1@tau) against dense RGB-D ground-truth point clouds

3. **Visual Reconstruction Losses from the 3D Model**:
   - Multi-view patch photometric & ZNCC reprojection consistency across tracks
   - Differentiable soft-point / isotropic Gaussian splatting render loss
     (masked L1 + SSIM on rendered vs. observed camera views)
   - Canonical 3-view orthographic 3D render loss (Top-Down BEV floorplan,
     Front, Side occupancy IoU + RGB appearance + depth consistency)

4. **Visual Loop Closure (`render_diagnostic_4panel`)**:
   - 2x2 matplotlib figure (BEV floorplan + 3D isometric + 3D reprojection
     error heatmap + splat-rendered camera view vs. GT frame) for VLM
     inspection and qualitative logging.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree

from agentic_sfm.eval.scene_eval import (
    _apply_sim3_to_pose,
    _as_cam_from_world,
    _camera_center,
    _direction_err_deg,
    _rot_geodesic_deg,
    _rotation_gauge,
    _umeyama,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Robust kernels & elementary helpers
# ---------------------------------------------------------------------------


def huber_loss(residual: np.ndarray | float, delta: float = 1.0) -> np.ndarray:
    """Elementwise Huber loss (quadratic below ``delta``, linear above)."""
    r = np.abs(np.asarray(residual, dtype=np.float64))
    quad = np.minimum(r, delta)
    lin = r - quad
    return 0.5 * quad * quad + delta * lin


def _windowed_ssim(
    img1: np.ndarray,
    img2: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Fast global/masked SSIM in [0, 1] for float images in [0, 1]."""
    a = np.asarray(img1, dtype=np.float64)
    b = np.asarray(img2, dtype=np.float64)
    if a.shape != b.shape or a.size == 0:
        return 0.0
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.ndim == 2 and a.ndim == 3:
            m = np.broadcast_to(m[..., None], a.shape)
        if not m.any():
            return 0.0
        a_vals = a[m]
        b_vals = b[m]
    else:
        a_vals = a.reshape(-1)
        b_vals = b.reshape(-1)
    if a_vals.size < 4:
        return float(np.clip(1.0 - np.mean(np.abs(a_vals - b_vals)), 0.0, 1.0))

    c1 = 0.01**2
    c2 = 0.03**2
    mu_a = float(np.mean(a_vals))
    mu_b = float(np.mean(b_vals))
    var_a = float(np.var(a_vals))
    var_b = float(np.var(b_vals))
    cov = float(np.mean((a_vals - mu_a) * (b_vals - mu_b)))
    num = (2.0 * mu_a * mu_b + c1) * (2.0 * cov + c2)
    den = (mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2)
    ssim = num / max(den, 1e-12)
    return float(np.clip((ssim + 1.0) * 0.5 if ssim < 0 else ssim, 0.0, 1.0))


# ---------------------------------------------------------------------------
# 1. Intrinsic Mathematical Losses (BA reprojection, tracks, graph rigidity)
# ---------------------------------------------------------------------------


def compute_intrinsic_ba_metrics(
    reproj_errors_px: np.ndarray | list[float],
    track_lengths: np.ndarray | list[int],
    sigma_px: float = 1.5,
) -> dict[str, float]:
    """Compute track-conditioned robust BA quality metrics (no GT required).

    Args:
        reproj_errors_px: Per-3D-point mean reprojection error in pixels.
        track_lengths: Per-3D-point track length (number of observing views).
        sigma_px: Huber transition threshold in pixels.

    Returns:
        Dict with ``huber_reproj_error``, ``mean_track_length``,
        ``frac_tracks_ge3``, ``track_rigidity_score``, ``intrinsic_ba_reward``.
    """
    errs = np.asarray(reproj_errors_px, dtype=np.float64).reshape(-1)
    tlen = np.asarray(track_lengths, dtype=np.float64).reshape(-1)
    valid = np.isfinite(errs) & np.isfinite(tlen) & (tlen >= 2)
    errs = errs[valid]
    tlen = tlen[valid]

    if errs.size == 0:
        return {
            "huber_reproj_error": float("inf"),
            "mean_track_length": 0.0,
            "frac_tracks_ge3": 0.0,
            "track_rigidity_score": 0.0,
            "intrinsic_ba_reward": 0.0,
        }

    # Weight each 3D point by its track support so longer tracks carry more weight
    weights = np.clip(tlen, 2.0, 20.0)
    huber_vals = huber_loss(errs / max(sigma_px, 1e-6), delta=1.0)
    huber_mean = float(np.average(huber_vals, weights=weights))
    mean_tlen = float(np.mean(tlen))
    frac_ge3 = float(np.mean(tlen >= 3.0))

    # 2-view tracks have 0 redundancy; rigidity saturates around mean length 6
    length_factor = float(np.clip((mean_tlen - 2.0) / 4.0, 0.0, 1.0))
    rigidity = length_factor * frac_ge3
    reproj_factor = float(math.exp(-huber_mean / 2.0))
    ba_reward = reproj_factor * rigidity

    return {
        "huber_reproj_error": huber_mean,
        "mean_track_length": mean_tlen,
        "frac_tracks_ge3": frac_ge3,
        "track_rigidity_score": rigidity,
        "intrinsic_ba_reward": ba_reward,
    }


def extract_colmap_point_arrays(
    recon_or_dir: Any,
) -> dict[str, Any]:
    """Extract 3D points, RGB colors, reprojection errors, track lengths, and
    per-image 2D observations from a ``pycolmap.Reconstruction`` or directory.
    """
    recon = recon_or_dir
    if isinstance(recon_or_dir, (str, Path)):
        import pycolmap

        recon = pycolmap.Reconstruction(str(recon_or_dir))

    pts, cols, errs, tlens = [], [], [], []
    observations: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}

    # Pre-index 2D keypoints per image if available
    img_kpts: dict[int, np.ndarray] = {}
    for iid, im in recon.images.items():
        try:
            p2d = [ kp.xy for kp in im.points2D ]
            img_kpts[int(iid)] = np.asarray(p2d, dtype=np.float64) if p2d else np.zeros((0, 2))
        except Exception:
            img_kpts[int(iid)] = np.zeros((0, 2))

    for _pid, p in recon.points3D.items():
        xyz = np.asarray(p.xyz, dtype=np.float64)
        col = np.asarray(p.color, dtype=np.float64) / 255.0
        err = float(getattr(p, "error", 0.0))
        elems = getattr(getattr(p, "track", None), "elements", [])
        tlen = len(elems)
        pts.append(xyz)
        cols.append(col)
        errs.append(err)
        tlens.append(tlen)
        for el in elems:
            iid = int(el.image_id)
            kp_idx = int(el.point2D_idx)
            kps = img_kpts.get(iid)
            if kps is not None and 0 <= kp_idx < len(kps):
                observations.setdefault(iid, []).append((xyz, kps[kp_idx]))

    return {
        "points3d": np.asarray(pts, dtype=np.float64) if pts else np.zeros((0, 3)),
        "colors": np.asarray(cols, dtype=np.float64) if cols else np.zeros((0, 3)),
        "reproj_errors": np.asarray(errs, dtype=np.float64) if errs else np.zeros(0),
        "track_lengths": np.asarray(tlens, dtype=np.int64) if tlens else np.zeros(0, dtype=np.int64),
        "observations": observations,
        "num_registered": len(recon.images),
    }


def compute_graph_rigidity_metrics(
    num_nodes: int,
    edges: list[tuple[int, int]] | list[tuple[int, int, np.ndarray]],
) -> dict[str, float]:
    """Compute Fiedler algebraic connectivity (lambda_2) and 3-cycle rotation
    consistency on the view graph.

    Args:
        num_nodes: Total number of scene images N.
        edges: List of ``(u, v)`` or ``(u, v, R_uv)`` where ``R_uv`` is the 3x3
            relative rotation from ``u`` to ``v`` (``R_v @ R_u^T``).
    """
    if num_nodes <= 1 or not edges:
        return {
            "fiedler_lambda2": 0.0,
            "mean_cycle_error_deg": 0.0,
            "num_3cycles": 0,
            "graph_rigidity_reward": 0.0,
        }

    adj = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    rot_map: dict[tuple[int, int], np.ndarray] = {}

    for item in edges:
        u, v = int(item[0]), int(item[1])
        if u == v or not (0 <= u < num_nodes and 0 <= v < num_nodes):
            continue
        adj[u, v] = 1.0
        adj[v, u] = 1.0
        if len(item) >= 3 and item[2] is not None:
            R_uv = np.asarray(item[2], dtype=np.float64).reshape(3, 3)
            rot_map[(u, v)] = R_uv
            rot_map[(v, u)] = R_uv.T

    deg = adj.sum(axis=1)
    with np.errstate(divide="ignore"):
        inv_sqrt_deg = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_inv_sqrt = np.diag(inv_sqrt_deg)
    L_norm = np.eye(num_nodes) - D_inv_sqrt @ adj @ D_inv_sqrt
    # Zero out isolated nodes? No — isolated nodes give 0 eigenvalues in L,
    # which properly makes lambda_2 == 0 when the graph is disconnected!
    if np.any(deg == 0):
        # For isolated nodes, L_ii = 1 if we kept I; standard convention sets
        # L_ii = 0 on isolated vertices so disconnected graphs have lambda_2 = 0.
        for idx in np.where(deg == 0)[0]:
            L_norm[idx, idx] = 0.0

    eigvals = np.linalg.eigvalsh(L_norm)
    eigvals = np.sort(np.clip(eigvals, 0.0, None))
    fiedler = float(eigvals[1]) if len(eigvals) >= 2 else 0.0

    # Enumerate 3-cycles (u < v < w)
    cycle_errs = []
    if rot_map:
        for u in range(num_nodes):
            nbrs_u = [v for v in range(u + 1, num_nodes) if adj[u, v] > 0]
            for idx_a, v in enumerate(nbrs_u):
                for w in nbrs_u[idx_a + 1 :]:
                    if adj[v, w] > 0:
                        R_uv = rot_map.get((u, v))
                        R_vw = rot_map.get((v, w))
                        R_wu = rot_map.get((w, u))
                        if R_uv is not None and R_vw is not None and R_wu is not None:
                            R_cyc = R_wu @ R_vw @ R_uv
                            cycle_errs.append(_rot_geodesic_deg(R_cyc, np.eye(3)))

    mean_cyc = float(np.mean(cycle_errs)) if cycle_errs else 0.0
    conn_score = float(np.clip(fiedler / 0.3, 0.0, 1.0))
    cyc_score = float(math.exp(-mean_cyc / 15.0)) if cycle_errs else 1.0
    return {
        "fiedler_lambda2": fiedler,
        "mean_cycle_error_deg": mean_cyc,
        "num_3cycles": len(cycle_errs),
        "graph_rigidity_reward": conn_score * cyc_score,
    }


# ---------------------------------------------------------------------------
# 2. Extrinsic Mathematical 3D Losses (mAA, ATE, depth, 3D F-score)
# ---------------------------------------------------------------------------


def compute_all_pairs_pose_maa(
    pred_poses: dict[int, np.ndarray],
    gt_poses: dict[int, np.ndarray],
    num_total_images: int | None = None,
    thresholds: tuple[float, ...] = (5.0, 10.0, 20.0),
) -> dict[str, float]:
    """Compute IMC-style relative-pose mAA over ALL scene pairs.

    Pairs where either image is unregistered receive 180° rotation and
    translation error, so abstaining on difficult views cannot inflate AUC.
    """
    all_ids = sorted(gt_poses.keys())
    n_total = max(int(num_total_images or len(all_ids)), len(all_ids))
    if n_total < 2:
        return {"all_pairs_maa": 0.0, "maa_5": 0.0, "maa_10": 0.0, "maa_20": 0.0}

    total_pairs = n_total * (n_total - 1) // 2
    pass_counts = {float(t): 0 for t in thresholds}

    for idx_a, i in enumerate(all_ids):
        for j in all_ids[idx_a + 1 :]:
            if i not in pred_poses or j not in pred_poses:
                continue
            Mp_i, Mp_j = pred_poses[i], pred_poses[j]
            Mg_i, Mg_j = gt_poses[i], gt_poses[j]

            R_rel_p = Mp_j[:3, :3] @ Mp_i[:3, :3].T
            R_rel_g = Mg_j[:3, :3] @ Mg_i[:3, :3].T
            rot_err = _rot_geodesic_deg(R_rel_p, R_rel_g)

            # Relative translation in camera j's frame: t_{j<-i} = t_j - R_{j<-i} t_i
            t_rel_p = Mp_j[:3, 3] - R_rel_p @ Mp_i[:3, 3]
            t_rel_g = Mg_j[:3, 3] - R_rel_g @ Mg_i[:3, 3]
            trans_err = _direction_err_deg(t_rel_p, t_rel_g)

            err = max(rot_err, trans_err)
            for t in thresholds:
                if err < t:
                    pass_counts[float(t)] += 1

    per_t = {float(t): pass_counts[float(t)] / max(total_pairs, 1) for t in thresholds}
    maa = float(np.mean(list(per_t.values())))
    return {
        "all_pairs_maa": maa,
        "maa_5": per_t.get(5.0, 0.0),
        "maa_10": per_t.get(10.0, 0.0),
        "maa_20": per_t.get(20.0, 0.0),
    }


def align_poses_and_points_sim3(
    pred_poses: dict[int, np.ndarray],
    gt_poses: dict[int, np.ndarray],
    points3d: np.ndarray | None = None,
) -> dict[str, Any]:
    """Align predicted cam-from-world poses and 3D points to GT via gauge-fixed
    Umeyama Sim(3), returning rotation errors, normalized ATE, and aligned
    points ``X' = s * Rs @ X + ts``.
    """
    matched = sorted(set(pred_poses) & set(gt_poses))
    pts = np.asarray(points3d, dtype=np.float64) if points3d is not None else np.zeros((0, 3))
    if len(matched) < 3:
        return {
            "matched_indices": matched,
            "scale": 1.0,
            "R_sim3": np.eye(3),
            "t_sim3": np.zeros(3),
            "mean_rot_error_deg": None,
            "mean_ate_m": None,
            "normalized_ate": None,
            "ate_reward": 0.0,
            "aligned_points3d": pts,
            "aligned_poses": {},
        }

    W = _rotation_gauge(pred_poses, gt_poses, matched)
    pred_w = {
        i: _apply_sim3_to_pose(pred_poses[i], 1.0, W, np.zeros(3))
        for i in matched
    }
    src_c = np.stack([_camera_center(pred_w[i]) for i in matched])
    dst_c = np.stack([_camera_center(gt_poses[i]) for i in matched])
    s, R_res, ts = _umeyama(src_c, dst_c)
    Rs = R_res @ W

    aligned_poses: dict[int, np.ndarray] = {}
    rot_errs, trans_errs_m = [], []
    for i, M in pred_poses.items():
        M_al = _apply_sim3_to_pose(M, s, Rs, ts)
        aligned_poses[i] = M_al
        if i in gt_poses:
            rot_errs.append(_rot_geodesic_deg(M_al[:3, :3], gt_poses[i][:3, :3]))
            c_al = _camera_center(M_al)
            c_gt = _camera_center(gt_poses[i])
            trans_errs_m.append(float(np.linalg.norm(c_al - c_gt)))

    # Normalize ATE by the GT trajectory diameter so indoor (m) and outdoor
    # scenes share a dimensionless scale.
    all_gt_c = np.stack([_camera_center(gt_poses[i]) for i in gt_poses])
    gt_diam = float(np.ptp(all_gt_c, axis=0).max()) if len(all_gt_c) >= 2 else 1.0
    gt_diam = max(gt_diam, 1e-3)

    ate_m = float(np.mean(trans_errs_m))
    norm_ate = float(np.mean(huber_loss(np.asarray(trans_errs_m) / gt_diam, delta=0.25)))
    cov = len(matched) / max(len(gt_poses), 1)
    ate_reward = float(cov * math.exp(-norm_ate / 0.15))

    if len(pts) > 0:
        aligned_pts = (s * (Rs @ pts.T)).T + ts
    else:
        aligned_pts = pts

    return {
        "matched_indices": matched,
        "scale": float(s),
        "R_sim3": Rs,
        "t_sim3": ts,
        "mean_rot_error_deg": float(np.mean(rot_errs)),
        "mean_ate_m": ate_m,
        "normalized_ate": norm_ate,
        "ate_reward": ate_reward,
        "aligned_points3d": aligned_pts,
        "aligned_poses": aligned_poses,
    }


def _voxel_downsample(
    points: np.ndarray,
    colors: np.ndarray | None = None,
    voxel_size: float = 0.05,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Deterministic voxel-grid centroid downsampling."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0 or voxel_size <= 0:
        return pts, colors
    coords = np.floor(pts / voxel_size).astype(np.int64)
    _, unique_idx = np.unique(coords, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    out_pts = pts[unique_idx]
    out_cols = colors[unique_idx] if colors is not None and len(colors) == len(pts) else None
    return out_pts, out_cols


def compute_pointcloud_f1(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    tau: float = 0.10,
    max_dist: float = 1.0,
    voxel_size: float = 0.08,
) -> dict[str, float]:
    """Compute bidirectional truncated Chamfer distance and 3D F-score (F1@tau).

    ``gt_points`` is voxel-downsampled to ``voxel_size`` before computing
    completeness recall so that sparse keypoint clouds are judged on surface
    coverage rather than pixel density.
    """
    P = np.asarray(pred_points, dtype=np.float64)
    G = np.asarray(gt_points, dtype=np.float64)
    P = P[np.isfinite(P).all(axis=1)] if P.ndim == 2 and len(P) else np.zeros((0, 3))
    G = G[np.isfinite(G).all(axis=1)] if G.ndim == 2 and len(G) else np.zeros((0, 3))

    if len(P) == 0 or len(G) == 0:
        return {
            "chamfer_acc": max_dist,
            "chamfer_comp": max_dist,
            "chamfer_l1": max_dist,
            "precision_at_tau": 0.0,
            "recall_at_tau": 0.0,
            "f1_3d": 0.0,
        }

    G_vox, _ = _voxel_downsample(G, voxel_size=voxel_size)
    if len(G_vox) == 0:
        G_vox = G

    tree_gt = cKDTree(G)
    tree_pred = cKDTree(P)

    d_p2g, _ = tree_gt.query(P, k=1)
    d_g2p, _ = tree_pred.query(G_vox, k=1)

    acc = float(np.mean(np.minimum(d_p2g, max_dist)))
    comp = float(np.mean(np.minimum(d_g2p, max_dist)))
    prec = float(np.mean(d_p2g < tau))
    rec = float(np.mean(d_g2p < tau))
    f1 = float(2.0 * prec * rec / max(prec + rec, 1e-8))

    return {
        "chamfer_acc": acc,
        "chamfer_comp": comp,
        "chamfer_l1": 0.5 * (acc + comp),
        "precision_at_tau": prec,
        "recall_at_tau": rec,
        "f1_3d": f1,
    }


def compute_projected_depth_loss(
    aligned_points3d: np.ndarray,
    gt_poses: dict[int, np.ndarray],
    intrinsics: dict[int, np.ndarray] | list[np.ndarray],
    depth_maps: dict[int, np.ndarray | str | Path],
    depth_scale: float = 1000.0,
    max_depth: float = 10.0,
    grad_weight: float = 0.5,
) -> dict[str, float]:
    """Compare Sim(3)-aligned 3D points against sensor depth maps (e.g. ScanNet)
    using scale-invariant log-depth L1 + VGGT-style spatial gradient loss.
    """
    pts = np.asarray(aligned_points3d, dtype=np.float64)
    if len(pts) == 0 or not depth_maps:
        return {
            "depth_log_l1": float("inf"),
            "depth_grad_l1": float("inf"),
            "depth_loss": float("inf"),
            "depth_consistency_reward": 0.0,
            "num_depth_samples": 0,
        }

    log_errs = []
    grad_errs = []

    for idx, d_src in depth_maps.items():
        if idx not in gt_poses:
            continue
        if isinstance(intrinsics, dict):
            K = intrinsics.get(idx)
        elif isinstance(intrinsics, (list, tuple)) and 0 <= idx < len(intrinsics):
            K = intrinsics[idx]
        else:
            K = None
        if K is None:
            continue
        K = np.asarray(K, dtype=np.float64)
        if K.shape == (4, 4):
            K = K[:3, :3]

        try:
            if isinstance(d_src, (str, Path)):
                depth = np.asarray(Image.open(d_src), dtype=np.float64) / depth_scale
            else:
                depth = np.asarray(d_src, dtype=np.float64)
        except Exception:
            continue

        if depth.ndim != 2:
            continue
        H, W = depth.shape

        M_cw = gt_poses[idx]
        R_cw, t_cw = M_cw[:3, :3], M_cw[:3, 3]
        pts_cam = (R_cw @ pts.T).T + t_cw
        z_pred = pts_cam[:, 2]
        in_front = z_pred > 0.1
        if not np.any(in_front):
            continue
        pts_c = pts_cam[in_front]
        z_p = z_pred[in_front]

        u = K[0, 0] * pts_c[:, 0] / z_p + K[0, 2]
        v = K[1, 1] * pts_c[:, 1] / z_p + K[1, 2]
        ui = np.round(u).astype(np.int64)
        vi = np.round(v).astype(np.int64)
        in_bounds = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        if not np.any(in_bounds):
            continue

        ui, vi, z_p = ui[in_bounds], vi[in_bounds], z_p[in_bounds]
        # Keep only the front-most predicted point per pixel (z-buffer occlusion)
        order = np.argsort(z_p)
        ui, vi, z_p = ui[order], vi[order], z_p[order]
        flat_px = vi * W + ui
        _, first_idx = np.unique(flat_px, return_index=True)
        first_idx = np.sort(first_idx)
        ui, vi, z_p = ui[first_idx], vi[first_idx], z_p[first_idx]

        z_gt = depth[vi, ui]
        valid = (z_gt > 0.1) & (z_gt <= max_depth) & np.isfinite(z_gt)
        if not np.any(valid):
            continue

        z_p = z_p[valid]
        z_gt = z_gt[valid]
        log_errs.append(np.abs(np.log(z_p / z_gt)))

        # VGGT-style pairwise depth-difference (spatial gradient) consistency
        if len(z_p) >= 2:
            dz_pred = np.diff(z_p)
            dz_gt = np.diff(z_gt)
            grad_errs.append(np.abs(dz_pred - dz_gt))

    if not log_errs:
        return {
            "depth_log_l1": float("inf"),
            "depth_grad_l1": float("inf"),
            "depth_loss": float("inf"),
            "depth_consistency_reward": 0.0,
            "num_depth_samples": 0,
        }

    all_log = np.concatenate(log_errs)
    mean_log = float(np.mean(all_log))
    mean_grad = float(np.mean(np.concatenate(grad_errs))) if grad_errs else 0.0
    total_loss = mean_log + grad_weight * mean_grad
    reward = float(math.exp(-total_loss / 0.35))

    return {
        "depth_log_l1": mean_log,
        "depth_grad_l1": mean_grad,
        "depth_loss": total_loss,
        "depth_consistency_reward": reward,
        "num_depth_samples": int(all_log.size),
    }


# ---------------------------------------------------------------------------
# 3. Visual Reconstruction Losses from the 3D Model
# ---------------------------------------------------------------------------


def _rasterize_orthographic_view(
    points: np.ndarray,
    colors: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    axes: tuple[int, int, int],
    grid_size: int = 128,
    dilation_radius: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rasterize a 3D colored point cloud onto an orthographic 2D grid.

    ``axes = (ax_u, ax_v, ax_depth)`` selects which world coordinates map to
    horizontal, vertical, and depth axes (e.g. ``(0, 1, 2)`` = Top-Down BEV).

    Returns ``(occupancy_mask (H,W) bool, depth_norm (H,W) float, rgb (H,W,3) float)``.
    """
    H = W = int(grid_size)
    occ = np.zeros((H, W), dtype=bool)
    depth_buf = np.full((H, W), np.inf, dtype=np.float64)
    rgb_buf = np.zeros((H, W, 3), dtype=np.float64)

    if len(points) == 0:
        return occ, np.zeros((H, W), dtype=np.float64), rgb_buf

    au, av, az = axes
    span = np.maximum(bounds_max - bounds_min, 1e-6)
    u_norm = (points[:, au] - bounds_min[au]) / span[au]
    v_norm = (points[:, av] - bounds_min[av]) / span[av]
    z_norm = (points[:, az] - bounds_min[az]) / span[az]

    in_box = (
        (u_norm >= 0.0) & (u_norm <= 1.0)
        & (v_norm >= 0.0) & (v_norm <= 1.0)
        & np.isfinite(z_norm)
    )
    if not np.any(in_box):
        return occ, np.zeros((H, W), dtype=np.float64), rgb_buf

    u_px = np.clip((u_norm[in_box] * (W - 1)).astype(np.int64), 0, W - 1)
    v_px = np.clip((v_norm[in_box] * (H - 1)).astype(np.int64), 0, H - 1)
    z_val = np.clip(z_norm[in_box], 0.0, 1.0)
    cols = (
        np.clip(colors[in_box], 0.0, 1.0)
        if len(colors) == len(points)
        else np.ones((len(u_px), 3), dtype=np.float64) * 0.7
    )

    # Sort back-to-front so nearest points overwrite in a single vectorized pass
    order = np.argsort(-z_val)
    u_px, v_px, z_val, cols = u_px[order], v_px[order], z_val[order], cols[order]
    occ[v_px, u_px] = True
    depth_buf[v_px, u_px] = z_val
    rgb_buf[v_px, u_px] = cols

    if dilation_radius > 0 and occ.any():
        struct = np.ones((2 * dilation_radius + 1, 2 * dilation_radius + 1), dtype=bool)
        occ_dil = binary_dilation(occ, structure=struct)
        # Propagate nearest valid RGB/depth into dilated neighborhood via small shifts
        for dy in range(-dilation_radius, dilation_radius + 1):
            for dx in range(-dilation_radius, dilation_radius + 1):
                if dx == 0 and dy == 0:
                    continue
                shifted_occ = np.roll(np.roll(occ, dy, axis=0), dx, axis=1)
                fill = shifted_occ & ~occ
                if fill.any():
                    rgb_buf[fill] = np.roll(np.roll(rgb_buf, dy, axis=0), dx, axis=1)[fill]
                    depth_buf[fill] = np.roll(np.roll(depth_buf, dy, axis=0), dx, axis=1)[fill]
        occ = occ_dil

    depth_out = np.where(occ, depth_buf, 0.0)
    return occ, depth_out, rgb_buf


def compute_orthographic_3d_render_loss(
    pred_points: np.ndarray,
    pred_colors: np.ndarray,
    gt_points: np.ndarray,
    gt_colors: np.ndarray,
    grid_size: int = 128,
    dilation_radius: int = 2,
) -> dict[str, float]:
    """Render both predicted and GT 3D point clouds from 3 canonical
    orthographic views (Top-Down BEV, Front, Side) on the shared GT bounding
    box and score silhouette IoU, RGB appearance, and depth agreement.

    Directly catches symmetry folding ("doppelganger" wall-doubling) where a
    corridor or wall rotates outside the true floorplan footprint.
    """
    P = np.asarray(pred_points, dtype=np.float64)
    G = np.asarray(gt_points, dtype=np.float64)
    if len(P) == 0 or len(G) == 0:
        return {
            "ortho_bev_iou": 0.0,
            "ortho_occupancy_iou": 0.0,
            "ortho_rgb_l1": 1.0,
            "ortho_depth_l1": 1.0,
            "ortho_visual_reward": 0.0,
        }

    Cp = (
        np.asarray(pred_colors, dtype=np.float64)
        if pred_colors is not None and len(pred_colors) == len(P)
        else np.full_like(P, 0.7)
    )
    Cg = (
        np.asarray(gt_colors, dtype=np.float64)
        if gt_colors is not None and len(gt_colors) == len(G)
        else np.full_like(G, 0.7)
    )

    # Shared bounding box anchored to GT with a 15% margin so out-of-footprint
    # folded walls are still captured inside the canvas.
    mid = np.median(G, axis=0)
    half_span = np.maximum(np.ptp(G, axis=0) * 0.65, 0.5)
    bmin = mid - half_span
    bmax = mid + half_span

    views = [
        ("bev", (0, 1, 2)),
        ("front", (0, 2, 1)),
        ("side", (1, 2, 0)),
    ]
    ious, rgb_l1s, depth_l1s = [], [], []
    bev_iou = 0.0

    for name, axes in views:
        occ_p, dep_p, rgb_p = _rasterize_orthographic_view(
            P, Cp, bmin, bmax, axes, grid_size=grid_size, dilation_radius=dilation_radius
        )
        occ_g, dep_g, rgb_g = _rasterize_orthographic_view(
            G, Cg, bmin, bmax, axes, grid_size=grid_size, dilation_radius=dilation_radius
        )
        inter = occ_p & occ_g
        union = occ_p | occ_g
        iou = float(inter.sum() / max(union.sum(), 1))
        ious.append(iou)
        if name == "bev":
            bev_iou = iou

        if inter.any():
            rgb_l1s.append(float(np.mean(np.abs(rgb_p[inter] - rgb_g[inter]))))
            depth_l1s.append(float(np.mean(np.abs(dep_p[inter] - dep_g[inter]))))
        else:
            rgb_l1s.append(1.0)
            depth_l1s.append(1.0)

    mean_iou = float(np.mean(ious))
    mean_rgb_l1 = float(np.mean(rgb_l1s))
    mean_dep_l1 = float(np.mean(depth_l1s))

    # Gate appearance/depth credits by occupancy overlap so empty clouds score 0
    overlap_gate = min(mean_iou / 0.15, 1.0)
    reward = float(
        0.45 * mean_iou
        + overlap_gate * (0.30 * (1.0 - mean_rgb_l1) + 0.25 * (1.0 - mean_dep_l1))
    )

    return {
        "ortho_bev_iou": bev_iou,
        "ortho_occupancy_iou": mean_iou,
        "ortho_rgb_l1": mean_rgb_l1,
        "ortho_depth_l1": mean_dep_l1,
        "ortho_visual_reward": float(np.clip(reward, 0.0, 1.0)),
    }


def render_pointcloud_splat(
    points3d: np.ndarray,
    colors: np.ndarray,
    K: np.ndarray,
    cam_from_world: np.ndarray,
    width: int = 128,
    height: int = 128,
    splat_radius_px: int = 2,
) -> dict[str, np.ndarray]:
    """Render a 3D colored point cloud from a perspective camera ``(K, T_cw)``
    via soft Gaussian splatting / alpha compositing.

    Returns ``{"rgb": (H,W,3), "opacity": (H,W), "depth": (H,W)}``.
    """
    H, W = int(height), int(width)
    rgb_out = np.zeros((H, W, 3), dtype=np.float64)
    opacity_out = np.zeros((H, W), dtype=np.float64)
    depth_out = np.zeros((H, W), dtype=np.float64)

    pts = np.asarray(points3d, dtype=np.float64)
    if len(pts) == 0:
        return {"rgb": rgb_out, "opacity": opacity_out, "depth": depth_out}

    cols = (
        np.clip(np.asarray(colors, dtype=np.float64), 0.0, 1.0)
        if colors is not None and len(colors) == len(pts)
        else np.full_like(pts, 0.7)
    )
    M = _as_cam_from_world(cam_from_world)
    if M is None:
        return {"rgb": rgb_out, "opacity": opacity_out, "depth": depth_out}

    K = np.asarray(K, dtype=np.float64)
    if K.shape == (4, 4):
        K = K[:3, :3]

    pts_cam = (M[:3, :3] @ pts.T).T + M[:3, 3]
    z = pts_cam[:, 2]
    valid = (z > 0.05) & np.isfinite(z)
    if not np.any(valid):
        return {"rgb": rgb_out, "opacity": opacity_out, "depth": depth_out}

    pts_cam = pts_cam[valid]
    cols = cols[valid]
    z = z[valid]

    u = K[0, 0] * pts_cam[:, 0] / z + K[0, 2]
    v = K[1, 1] * pts_cam[:, 1] / z + K[1, 2]

    in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not np.any(in_img):
        return {"rgb": rgb_out, "opacity": opacity_out, "depth": depth_out}

    u = u[in_img]
    v = v[in_img]
    z = z[in_img]
    cols = cols[in_img]

    # Front-to-back order for alpha compositing
    order = np.argsort(z)
    u_px = np.clip(np.round(u[order]).astype(np.int64), 0, W - 1)
    v_px = np.clip(np.round(v[order]).astype(np.int64), 0, H - 1)
    z_ord = z[order]
    cols_ord = cols[order]

    r = max(int(splat_radius_px), 0)
    sigma = max(r * 0.75, 0.8)
    transmittance = np.ones((H, W), dtype=np.float64)

    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            weight = math.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma)) * 0.85
            vy = np.clip(v_px + dy, 0, H - 1)
            ux = np.clip(u_px + dx, 0, W - 1)
            # First hit at each pixel (closest depth) dominates
            layer_alpha = np.zeros((H, W), dtype=np.float64)
            layer_rgb = np.zeros((H, W, 3), dtype=np.float64)
            layer_z = np.zeros((H, W), dtype=np.float64)
            # Reverse order so front-most overwrites
            rev = slice(None, None, -1)
            layer_alpha[vy[rev], ux[rev]] = weight
            layer_rgb[vy[rev], ux[rev]] = cols_ord[rev]
            layer_z[vy[rev], ux[rev]] = z_ord[rev]

            contrib = transmittance * layer_alpha
            rgb_out += contrib[..., None] * layer_rgb
            depth_out += contrib * layer_z
            transmittance *= 1.0 - layer_alpha

    opacity_out = 1.0 - transmittance
    with np.errstate(divide="ignore", invalid="ignore"):
        rgb_out = np.where(opacity_out[..., None] > 1e-6, rgb_out / opacity_out[..., None], 0.0)
        depth_out = np.where(opacity_out > 1e-6, depth_out / opacity_out, 0.0)

    return {
        "rgb": np.clip(rgb_out, 0.0, 1.0),
        "opacity": np.clip(opacity_out, 0.0, 1.0),
        "depth": depth_out,
    }


def compute_splat_render_loss(
    points3d: np.ndarray,
    colors: np.ndarray,
    poses: dict[int, np.ndarray],
    intrinsics: dict[int, np.ndarray] | list[np.ndarray],
    images: dict[int, np.ndarray | Image.Image | str | Path],
    render_size: int = 128,
    splat_radius_px: int = 2,
) -> dict[str, float]:
    """Render the 3D point cloud into observed camera views and compute masked
    photometric L1 + SSIM visual reconstruction reward.
    """
    if len(points3d) == 0 or not poses or not images:
        return {
            "splat_l1": 1.0,
            "splat_ssim": 0.0,
            "splat_coverage": 0.0,
            "splat_visual_reward": 0.0,
        }

    l1s, ssims, covs = [], [], []
    for idx, img_src in images.items():
        if idx not in poses:
            continue
        if isinstance(intrinsics, dict):
            K = intrinsics.get(idx)
        elif isinstance(intrinsics, (list, tuple)) and 0 <= idx < len(intrinsics):
            K = intrinsics[idx]
        else:
            K = None
        if K is None:
            continue

        try:
            if isinstance(img_src, (str, Path)):
                pil = Image.open(img_src).convert("RGB")
            elif isinstance(img_src, Image.Image):
                pil = img_src.convert("RGB")
            else:
                arr = np.asarray(img_src)
                if arr.dtype != np.uint8:
                    arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
                pil = Image.fromarray(arr).convert("RGB")
        except Exception:
            continue

        orig_w, orig_h = pil.size
        pil_res = pil.resize((render_size, render_size), Image.Resampling.BILINEAR)
        gt_rgb = np.asarray(pil_res, dtype=np.float64) / 255.0

        K_mat = np.asarray(K, dtype=np.float64).copy()
        if K_mat.shape == (4, 4):
            K_mat = K_mat[:3, :3]
        K_scaled = K_mat.copy()
        K_scaled[0, :] *= render_size / max(orig_w, 1)
        K_scaled[1, :] *= render_size / max(orig_h, 1)

        out = render_pointcloud_splat(
            points3d,
            colors,
            K_scaled,
            poses[idx],
            width=render_size,
            height=render_size,
            splat_radius_px=splat_radius_px,
        )
        mask = out["opacity"] > 0.1
        if not np.any(mask):
            continue

        cov = float(np.mean(mask))
        l1 = float(np.mean(np.abs(out["rgb"][mask] - gt_rgb[mask])))
        ssim = _windowed_ssim(out["rgb"], gt_rgb, mask=mask)
        l1s.append(l1)
        ssims.append(ssim)
        covs.append(cov)

    if not ssims:
        return {
            "splat_l1": 1.0,
            "splat_ssim": 0.0,
            "splat_coverage": 0.0,
            "splat_visual_reward": 0.0,
        }

    mean_l1 = float(np.mean(l1s))
    mean_ssim = float(np.mean(ssims))
    mean_cov = float(np.mean(covs))
    cov_gate = min(mean_cov / 0.05, 1.0)
    reward = cov_gate * (0.6 * mean_ssim + 0.4 * (1.0 - mean_l1))

    return {
        "splat_l1": mean_l1,
        "splat_ssim": mean_ssim,
        "splat_coverage": mean_cov,
        "splat_visual_reward": float(np.clip(reward, 0.0, 1.0)),
    }


def compute_multiview_patch_consistency(
    points3d: np.ndarray,
    poses: dict[int, np.ndarray],
    intrinsics: dict[int, np.ndarray] | list[np.ndarray],
    images: dict[int, np.ndarray | Image.Image | str | Path],
    patch_radius: int = 3,
    max_points: int = 256,
) -> dict[str, float]:
    """Self-supervised multi-view patch photometric consistency across projected
    3D points (zero GT required).

    Projects each 3D point into all registered views where it is in front of the
    camera, extracts a ``(2r+1)x(2r+1)`` RGB patch, and measures L1 + ZNCC
    agreement against the multi-view mean patch.
    """
    pts = np.asarray(points3d, dtype=np.float64)
    if len(pts) == 0 or len(poses) < 2 or len(images) < 2:
        return {
            "patch_reproj_loss": 1.0,
            "patch_consistency_reward": 0.0,
            "num_consistent_points": 0,
        }

    if len(pts) > max_points:
        sel = np.linspace(0, len(pts) - 1, max_points).astype(np.int64)
        pts = pts[sel]

    loaded_imgs: dict[int, np.ndarray] = {}
    Ks: dict[int, np.ndarray] = {}
    for idx in poses:
        if idx not in images:
            continue
        if isinstance(intrinsics, dict):
            K = intrinsics.get(idx)
        elif isinstance(intrinsics, (list, tuple)) and 0 <= idx < len(intrinsics):
            K = intrinsics[idx]
        else:
            K = None
        if K is None:
            continue
        K_mat = np.asarray(K, dtype=np.float64)
        if K_mat.shape == (4, 4):
            K_mat = K_mat[:3, :3]
        try:
            src = images[idx]
            if isinstance(src, (str, Path)):
                arr = np.asarray(Image.open(src).convert("RGB"), dtype=np.float64) / 255.0
            elif isinstance(src, Image.Image):
                arr = np.asarray(src.convert("RGB"), dtype=np.float64) / 255.0
            else:
                arr = np.asarray(src, dtype=np.float64)
                if arr.max() > 1.5:
                    arr = arr / 255.0
            loaded_imgs[idx] = arr
            Ks[idx] = K_mat
        except Exception:
            continue

    if len(loaded_imgs) < 2:
        return {
            "patch_reproj_loss": 1.0,
            "patch_consistency_reward": 0.0,
            "num_consistent_points": 0,
        }

    r = max(int(patch_radius), 1)
    # Pre-project all points into each loaded view
    proj_uv: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for idx, img in loaded_imgs.items():
        H, W = img.shape[:2]
        M = poses[idx]
        K_mat = Ks[idx]
        p_cam = (M[:3, :3] @ pts.T).T + M[:3, 3]
        z = p_cam[:, 2]
        u = np.round(K_mat[0, 0] * p_cam[:, 0] / np.maximum(z, 1e-6) + K_mat[0, 2]).astype(np.int64)
        v = np.round(K_mat[1, 1] * p_cam[:, 1] / np.maximum(z, 1e-6) + K_mat[1, 2]).astype(np.int64)
        valid = (z > 0.1) & (u >= r) & (u < W - r) & (v >= r) & (v < H - r)
        proj_uv[idx] = (u, v, valid)

    point_losses = []
    for p_idx in range(len(pts)):
        patches = []
        for idx, img in loaded_imgs.items():
            u, v, valid = proj_uv[idx]
            if valid[p_idx]:
                ui, vi = u[p_idx], v[p_idx]
                patches.append(img[vi - r : vi + r + 1, ui - r : ui + r + 1])
        if len(patches) < 2:
            continue
        stack = np.stack(patches, axis=0)  # (V, 2r+1, 2r+1, 3)
        mean_patch = np.mean(stack, axis=0, keepdims=True)
        l1 = float(np.mean(np.abs(stack - mean_patch)))

        # ZNCC against mean patch
        s_flat = stack.reshape(len(patches), -1)
        m_flat = mean_patch.reshape(1, -1)
        s_zero = s_flat - s_flat.mean(axis=1, keepdims=True)
        m_zero = m_flat - m_flat.mean(axis=1, keepdims=True)
        denom = np.linalg.norm(s_zero, axis=1) * np.linalg.norm(m_zero, axis=1)
        zncc = np.where(denom > 1e-6, (s_zero * m_zero).sum(axis=1) / denom, 1.0)
        zncc_loss = float(np.mean(0.5 * (1.0 - np.clip(zncc, -1.0, 1.0))))
        point_losses.append(0.5 * l1 + 0.5 * zncc_loss)

    if not point_losses:
        return {
            "patch_reproj_loss": 1.0,
            "patch_consistency_reward": 0.0,
            "num_consistent_points": 0,
        }

    loss = float(np.mean(point_losses))
    return {
        "patch_reproj_loss": loss,
        "patch_consistency_reward": float(np.clip(1.0 - loss, 0.0, 1.0)),
        "num_consistent_points": len(point_losses),
    }


# ---------------------------------------------------------------------------
# 4. Visual Loop Closure: 4-Panel 3D Diagnostic Render for VLM Inspection
# ---------------------------------------------------------------------------


def render_diagnostic_4panel(
    points3d: np.ndarray,
    colors: np.ndarray,
    cam_centers: np.ndarray | None = None,
    reproj_errors: np.ndarray | None = None,
    splat_rgb: np.ndarray | None = None,
    title: str = "3D Reconstruction Diagnostic",
):
    """Render a 2x2 diagnostic figure of the reconstructed 3D model:
      - Top-Left: Top-Down BEV floorplan (X-Y points + camera trajectory)
      - Top-Right: 3D isometric point cloud + camera markers
      - Bottom-Left: 3D reprojection error heatmap (green <= 1px -> red >= 3px)
      - Bottom-Right: Splat-rendered camera view (or track/error histogram)

    Returns a matplotlib ``Figure`` (headless Agg).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_BG = "#0d1117"
    C_TEXT = "#ecf0f1"
    C_CAM = "#f39c12"

    P = np.asarray(points3d, dtype=np.float64) if points3d is not None else np.zeros((0, 3))
    C = (
        np.clip(np.asarray(colors, dtype=np.float64), 0.0, 1.0)
        if colors is not None and len(colors) == len(P)
        else np.full_like(P, 0.7)
    )
    cams = np.asarray(cam_centers, dtype=np.float64) if cam_centers is not None and len(cam_centers) else np.zeros((0, 3))

    fig = plt.figure(figsize=(10, 9), dpi=120)
    fig.patch.set_facecolor(FIG_BG)

    # Panel 1: Top-Down BEV floorplan
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_facecolor(FIG_BG)
    if len(P):
        ax1.scatter(P[:, 0], P[:, 2], c=C, s=3.0, alpha=0.85, linewidths=0)
    if len(cams):
        ax1.plot(cams[:, 0], cams[:, 2], "-", color=C_CAM, lw=1.2, alpha=0.8)
        ax1.scatter(cams[:, 0], cams[:, 2], c=C_CAM, s=35, marker="^", zorder=5)
    ax1.set_title("Top-Down BEV Floorplan (X–Z)", color=C_TEXT, fontsize=10, fontweight="bold")
    ax1.tick_params(colors=C_TEXT, labelsize=7)
    ax1.set_aspect("equal", adjustable="datalim")

    # Panel 2: 3D Isometric view
    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    ax2.set_facecolor(FIG_BG)
    if len(P):
        ax2.scatter(P[:, 0], P[:, 2], -P[:, 1], c=C, s=3.0, alpha=0.9, linewidths=0)
    if len(cams):
        ax2.scatter(cams[:, 0], cams[:, 2], -cams[:, 1], c=C_CAM, s=40, marker="^")
    ax2.set_title("3D Isometric View", color=C_TEXT, fontsize=10, fontweight="bold")
    ax2.tick_params(colors=C_TEXT, labelsize=6)

    # Panel 3: Reprojection residual heatmap in BEV
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_facecolor(FIG_BG)
    if len(P) and reproj_errors is not None and len(reproj_errors) == len(P):
        err_norm = np.clip(np.asarray(reproj_errors, dtype=np.float64) / 3.0, 0.0, 1.0)
        heat_cols = np.stack([err_norm, 1.0 - err_norm, np.zeros_like(err_norm)], axis=1)
        ax3.scatter(P[:, 0], P[:, 2], c=heat_cols, s=3.5, alpha=0.9, linewidths=0)
    elif len(P):
        ax3.scatter(P[:, 0], P[:, 2], c="#2ecc71", s=3.0, alpha=0.8, linewidths=0)
    ax3.set_title("Reprojection Residual Heatmap (green<1px, red>3px)", color=C_TEXT, fontsize=9, fontweight="bold")
    ax3.tick_params(colors=C_TEXT, labelsize=7)
    ax3.set_aspect("equal", adjustable="datalim")

    # Panel 4: Splat-rendered novel/camera view
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_facecolor(FIG_BG)
    if splat_rgb is not None and splat_rgb.size > 0:
        ax4.imshow(np.clip(splat_rgb, 0.0, 1.0))
        ax4.axis("off")
        ax4.set_title("3D Point-Splat Camera Render", color=C_TEXT, fontsize=10, fontweight="bold")
    else:
        ax4.text(
            0.5, 0.5, f"{len(P):,} 3D pts | {len(cams)} cams",
            color=C_TEXT, ha="center", va="center", fontsize=11, transform=ax4.transAxes,
        )
        ax4.axis("off")

    fig.suptitle(title, color=C_TEXT, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig
