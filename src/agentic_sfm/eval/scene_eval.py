"""Scene-level evaluation metrics for Phase 2 agentic SfM.

Evaluates a finished ``SceneRolloutEpisode`` against ground-truth scene info
and produces a flat ``group/metric`` dict suitable for ``wandb.log`` or
Trackio.  Metric groups:

  - ``registration/``  images registered, 3D points, reprojection error
  - ``pose/``          mean/median rotation + translation errors vs GT
                       (absolute after Sim(3) alignment + pairwise relative)
  - ``completeness/``  how much of the scene was reconstructed
  - ``doppelganger/``  precision/recall/F1 of the agent's filtering vs GT
                       doppelganger pairs (when provided)
  - ``efficiency/``    tool-call counts, per-tool histogram, quality-per-call
  - ``reward/``        pass-through of ``ep.reward_components``

``gt_scene`` is flexible: pass either the full scene dict produced by
``SceneDataset.from_megadepth`` (keys ``image_paths``, ``overlap_matrix``,
``image_indices``, ``gt_recon``) or a bare ``gt_recon`` dict with
``num_images`` / ``poses`` / optional ``doppelganger_pairs``.

GT poses are cam-from-world 4x4 matrices keyed by ``str(index)`` into
``ep.image_paths`` (the convention used by ``SceneDataset``), by image path,
or given as a list aligned with ``ep.image_paths``.

Predicted poses are read, in order, from ``recon_result["poses"]`` (dict or
list), ``recon_result["registered_images"]``/``"images"`` name lists, or by
loading the COLMAP sparse model in ``recon_result["output_dir"]`` with
pycolmap (optional — silently skipped when pycolmap is absent).
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Iterable
from typing import Any

import numpy as np

from agentic_sfm.rewards.pose_rewards import _ntep_process_reward_counts
from agentic_sfm.rl.scene_episode import (
    SceneRolloutEpisode,
    _base_image_id,
    _is_flagged_doppelganger,
    _pair_key,
)

logger = logging.getLogger(__name__)

_POSE_AUC_THRESHOLDS = (5, 10, 20)


# ---------------------------------------------------------------------------
# Pose representation helpers
# ---------------------------------------------------------------------------


def _qvec_to_rotmat(qvec: Iterable[float]) -> np.ndarray:
    """COLMAP Hamilton quaternion [qw, qx, qy, qz] -> 3x3 rotation."""
    w, x, y, z = [float(v) for v in qvec]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n <= 0:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _as_cam_from_world(pose: Any) -> np.ndarray | None:
    """Normalize a pose blob to a 4x4 cam-from-world matrix, or None.

    Accepts: 4x4 / 3x4 nested lists, flat length-16/12 lists, and dicts of
    the form ``{"R": 3x3, "t": 3}`` or ``{"qvec": 4, "tvec": 3}``.
    """
    if pose is None:
        return None
    if isinstance(pose, dict):
        if "R" in pose and "t" in pose:
            R = np.asarray(pose["R"], dtype=np.float64).reshape(3, 3)
            t = np.asarray(pose["t"], dtype=np.float64).reshape(3)
        elif "qvec" in pose and "tvec" in pose:
            R = _qvec_to_rotmat(pose["qvec"])
            t = np.asarray(pose["tvec"], dtype=np.float64).reshape(3)
        elif "pose" in pose or "cam_from_world" in pose:
            return _as_cam_from_world(pose.get("cam_from_world", pose.get("pose")))
        else:
            return None
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = t
        return M

    arr = np.asarray(pose, dtype=np.float64)
    if arr.shape == (4, 4):
        return arr
    if arr.shape == (3, 4):
        M = np.eye(4)
        M[:3, :4] = arr
        return M
    flat = arr.reshape(-1)
    if flat.size == 16:
        return flat.reshape(4, 4)
    if flat.size == 12:
        M = np.eye(4)
        M[:3, :4] = flat.reshape(3, 4)
        return M
    return None


def _rot_geodesic_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    """Rotation angle of R1 @ R2.T in degrees."""
    R_rel = R1 @ R2.T
    trace = float(np.clip(np.trace(R_rel), -1.0, 3.0))
    return float(np.degrees(np.arccos((trace - 1.0) / 2.0)))


def _direction_err_deg(t1: np.ndarray, t2: np.ndarray) -> float:
    """Angle between two translation vectors in degrees."""
    n1, n2 = np.linalg.norm(t1), np.linalg.norm(t2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 180.0 if n1 != n2 else 0.0
    cos = float(np.clip(np.dot(t1, t2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def _camera_center(M: np.ndarray) -> np.ndarray:
    """World-space camera center of a cam-from-world 4x4."""
    R, t = M[:3, :3], M[:3, 3]
    return -R.T @ t


def _umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Sim(3) alignment: returns (s, R, t) with dst ~= s * R @ src + t.

    ``src``/``dst`` are (N, 3) corresponding point sets; needs N >= 3 for a
    well-constrained solution (2 points still return a Kabsch solution).
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    X = src - mu_s
    Y = dst - mu_d
    cov = (Y.T @ X) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var = float((X**2).sum() / n)
    s = float(np.trace(np.diag(D) @ S) / var) if var > 1e-12 else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def _rotation_gauge(pred_poses: dict[int, np.ndarray], gt_poses: dict[int, np.ndarray],
                    matched: list[int]) -> np.ndarray:
    """World-frame rotation W aligning pred rotations to GT.

    W ≈ mean over matched images of R_gt^T R_pred projected to SO(3)
    (chordal single-rotation averaging).  Center-based Umeyama alone cannot
    resolve the rotation about a degenerate (e.g. collinear) center set, so
    the rotational gauge is fixed from the camera rotations first.
    """
    A = np.zeros((3, 3))
    for i in matched:
        A += gt_poses[i][:3, :3].T @ pred_poses[i][:3, :3]
    A /= max(len(matched), 1)
    U, _, Vt = np.linalg.svd(A)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1.0
    return U @ S @ Vt


def _apply_sim3_to_pose(M: np.ndarray, s: float, Rs: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """Apply a world-frame Sim(3) X' = s Rs X + ts to a cam-from-world pose.

    The camera center transforms as a world point (C' = s Rs C + ts) and the
    rotation as R_c' = R_c Rs^T, so the new cam-from-world translation is
    t_c' = -R_c' C' = s t_c - R_c Rs^T ts.
    """
    R_c, t_c = M[:3, :3], M[:3, 3]
    R_new = R_c @ Rs.T
    t_new = s * t_c - R_new @ ts
    out = np.eye(4)
    out[:3, :3] = R_new
    out[:3, 3] = t_new
    return out


# ---------------------------------------------------------------------------
# Image-name resolution: predicted/GT keys -> ep.image_paths index
# ---------------------------------------------------------------------------


def _index_lookup(ep: SceneRolloutEpisode) -> dict[str, int]:
    """Map every alias for an episode image to its index.

    Aliases: positional index ("3"), server id ("img_0003"), full relative
    path, and basename — so COLMAP image names, server ids, and GT pose keys
    all resolve to the same index.
    """
    lookup: dict[str, int] = {}
    for i, p in enumerate(ep.image_paths):
        p_str = str(p)
        lookup.setdefault(str(i), i)
        lookup.setdefault(f"img_{i:04d}", i)
        lookup.setdefault(p_str, i)
        lookup.setdefault(os.path.basename(p_str), i)
    return lookup


def _resolve_image_index(key: Any, lookup: dict[str, int]) -> int | None:
    """Resolve a pose/image key (index, id, path, name) to an image index."""
    if isinstance(key, (int, np.integer)):
        return int(key) if 0 <= int(key) < 10**9 else None
    s = _base_image_id(str(key))  # strip server crop suffixes
    if s in lookup:
        return lookup[s]
    base = os.path.basename(s)
    if base in lookup:
        return lookup[base]
    # Relative path under image_root, or nested COLMAP name like "a/b/c.jpg".
    for alias, idx in lookup.items():
        if "/" in alias and (alias.endswith(s) or s.endswith(alias)):
            return idx
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Predicted poses: recon_result["poses"], name lists, or COLMAP output_dir
# ---------------------------------------------------------------------------


def _poses_from_colmap_dir(output_dir: str) -> dict[str, np.ndarray]:
    """Load per-image cam-from-world poses from a COLMAP sparse dir.

    ``output_dir`` may be the model dir itself (cameras.bin/images.bin) or a
    ``sparse`` parent containing a ``0`` subdir.  Returns {} when pycolmap is
    unavailable or the load fails — this path is best-effort.
    """
    try:
        import pycolmap  # noqa: F401
    except ImportError:
        return {}

    candidates = [output_dir]
    sub = os.path.join(output_dir, "0")
    if os.path.isdir(sub):
        candidates.insert(0, sub)

    for cand in candidates:
        try:
            recon = pycolmap.Reconstruction(cand)
        except Exception:
            continue
        poses: dict[str, np.ndarray] = {}
        try:
            images = recon.images
        except Exception:
            continue
        for _, img in images.items():
            M = None
            cfw = getattr(img, "cam_from_world", None)
            if cfw is not None:
                try:
                    obj = cfw() if callable(cfw) else cfw
                    m = getattr(obj, "matrix", None)
                    arr = m() if callable(m) else (m if m is not None else obj)
                    M = _as_cam_from_world(np.asarray(arr))
                except Exception:
                    M = None
            if M is None and hasattr(img, "qvec") and hasattr(img, "tvec"):
                M = _as_cam_from_world({"qvec": img.qvec, "tvec": img.tvec})
            if M is not None:
                poses[str(getattr(img, "name", f"image_{len(poses)}"))] = M
        if poses:
            return poses
    return {}


def _extract_pred_poses(
    recon_result: dict[str, Any],
    lookup: dict[str, int],
) -> dict[int, np.ndarray]:
    """Predicted cam-from-world poses keyed by ep.image_paths index."""
    if not isinstance(recon_result, dict) or recon_result.get("error"):
        return {}

    raw = recon_result.get("poses")
    out: dict[int, np.ndarray] = {}

    if isinstance(raw, dict):
        for key, pose in raw.items():
            M = _as_cam_from_world(pose)
            idx = _resolve_image_index(key, lookup)
            if M is not None and idx is not None:
                out[idx] = M
    elif isinstance(raw, (list, tuple)):
        names = (
            recon_result.get("registered_images")
            or recon_result.get("image_names")
            or recon_result.get("images")
        )
        for i, pose in enumerate(raw):
            if isinstance(pose, dict) and any(
                k in pose for k in ("name", "image_name", "image_id")
            ):
                key = pose.get("name", pose.get("image_name", pose.get("image_id")))
                M = _as_cam_from_world(pose)
            else:
                key = names[i] if isinstance(names, (list, tuple)) and i < len(names) else i
                M = _as_cam_from_world(pose)
            idx = _resolve_image_index(key, lookup)
            if M is not None and idx is not None:
                out[idx] = M

    if out:
        return out

    output_dir = recon_result.get("output_dir")
    if output_dir and os.path.isdir(str(output_dir)):
        for name, M in _poses_from_colmap_dir(str(output_dir)).items():
            idx = _resolve_image_index(name, lookup)
            if idx is not None:
                out[idx] = M
    return out


def _extract_gt_poses(
    gt_recon: dict[str, Any],
    lookup: dict[str, int],
) -> dict[int, np.ndarray]:
    """GT cam-from-world poses keyed by ep.image_paths index."""
    raw = gt_recon.get("poses")
    out: dict[int, np.ndarray] = {}
    if isinstance(raw, dict):
        for key, pose in raw.items():
            M = _as_cam_from_world(pose)
            idx = _resolve_image_index(key, lookup)
            if M is not None and idx is not None:
                out[idx] = M
    elif isinstance(raw, (list, tuple, np.ndarray)):
        for i, pose in enumerate(raw):
            M = _as_cam_from_world(pose)
            if M is not None:
                out[i] = M
    return out


# ---------------------------------------------------------------------------
# GT scene normalization
# ---------------------------------------------------------------------------


def _split_gt_scene(
    gt_scene: dict[str, Any] | None,
    ep: SceneRolloutEpisode,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (gt_recon, top_level_scene_info).

    ``gt_scene`` may be a full SceneDataset scene dict (with a nested
    ``gt_recon``) or a bare gt_recon.  ``ep.gt_recon`` is the fallback.
    """
    scene = gt_scene if isinstance(gt_scene, dict) else {}
    if not scene and isinstance(ep.gt_recon, dict):
        scene = ep.gt_recon
    if isinstance(scene.get("gt_recon"), dict):
        return scene["gt_recon"], scene
    return scene, scene


def _gt_doppelganger_keys(
    scene: dict[str, Any], gt_recon: dict[str, Any], lookup: dict[str, int]
) -> set[str]:
    """Canonical pair keys of GT doppelganger pairs, or empty set.

    Reads ``doppelganger_pairs`` / ``doppelgangers`` / ``gt_doppelganger_pairs``
    from the scene dict or gt_recon.  Entries may be ``"a__b"`` strings,
    ``[a, b]`` pairs of ids/paths, or ``[i, j]`` index pairs.
    """
    raw = None
    for source in (scene, gt_recon):
        for key in ("doppelganger_pairs", "doppelgangers", "gt_doppelganger_pairs"):
            if source.get(key):
                raw = source[key]
                break
        if raw:
            break
    if not raw:
        return set()

    def _resolve_alias(v: Any) -> str:
        """Map a GT pair endpoint (index / path / id) to a server id."""
        if isinstance(v, (int, np.integer)):
            return f"img_{int(v):04d}"
        idx = _resolve_image_index(v, lookup)
        if idx is not None:
            return f"img_{idx:04d}"
        return _base_image_id(str(v))

    def _canon(entry: Any) -> str:
        if isinstance(entry, str):
            parts = entry.split("__")
            if len(parts) == 2:
                return _pair_key(_resolve_alias(parts[0]), _resolve_alias(parts[1]))
            return entry
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            return _pair_key(_resolve_alias(entry[0]), _resolve_alias(entry[1]))
        return str(entry)

    return {_canon(e) for e in raw}


# ---------------------------------------------------------------------------
# Metric groups
# ---------------------------------------------------------------------------


def _registration_metrics(
    ep: SceneRolloutEpisode, recon: dict[str, Any], gt_num_images: int
) -> dict[str, Any]:
    num_images = ep.num_images or len(ep.image_paths) or gt_num_images
    num_registered = int(recon.get("num_registered") or 0)
    num_points3d = int(recon.get("num_points3d") or 0)

    # Reprojection error: prefer recon_result, else last inspect result.
    reproj = recon.get("mean_reproj_error", recon.get("mean_reprojection_error"))
    if reproj is None:
        for res in reversed(ep.results or []):
            if isinstance(res, dict):
                reproj = res.get("mean_reproj_error", res.get("mean_reprojection_error"))
                if reproj is not None:
                    break

    n_components = recon.get("num_models", recon.get("num_components"))
    reg_fraction = num_registered / max(num_images, 1)
    return {
        "registration/num_images": int(num_images),
        "registration/num_registered": num_registered,
        "registration/registered_fraction": float(reg_fraction),
        "registration/num_points3d": num_points3d,
        "registration/num_pairs_matched": int(recon.get("num_pairs_matched") or 0),
        "registration/mean_reproj_error_px": (
            float(reproj) if reproj is not None else None
        ),
        "registration/num_components": int(n_components) if n_components else None,
        "registration/success": float(num_registered > 0),
        "registration/full": float(num_images > 0 and num_registered >= num_images),
    }


def _pose_metrics(
    pred_poses: dict[int, np.ndarray],
    gt_poses: dict[int, np.ndarray],
    recon: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Absolute (Sim3-aligned) + pairwise-relative pose errors vs GT.

    Absolute camera-center distances are in GT world units; ``*_rel`` values
    normalize by the mean pairwise GT baseline.  Pairwise relative errors are
    gauge-free and are computed whenever >=2 images share pred + GT poses.
    """
    metrics: dict[str, Any] = {}
    per_image: list[dict[str, Any]] = []
    per_pair: list[dict[str, Any]] = []

    matched = sorted(set(pred_poses) & set(gt_poses))
    metrics["pose/available"] = float(len(matched) > 0)
    metrics["pose/num_images_with_gt"] = len(matched)
    metrics["pose/num_pred_poses"] = len(pred_poses)
    metrics["pose/num_gt_poses"] = len(gt_poses)

    reported = recon.get("mean_pose_error_deg")
    metrics["pose/reported_mean_error_deg"] = (
        float(reported) if reported is not None else None
    )
    if not matched:
        return metrics, per_image, per_pair

    # --- Absolute errors after Sim(3) alignment of camera centers ---------
    # Needs >=3 matched images for a well-constrained similarity; with fewer
    # the rotation gauge is unknown so only gauge-free relative errors are
    # reported below.
    metrics["pose/aligned"] = 0.0
    if len(matched) >= 3:
        # Two-step alignment: fix the rotational gauge from camera rotations
        # first, then fit the residual Sim(3) on camera centers.  Robust to
        # (near-)collinear center sets where center-only alignment leaves
        # one rotation axis unconstrained.
        W = _rotation_gauge(pred_poses, gt_poses, matched)
        pred_w = {
            i: _apply_sim3_to_pose(pred_poses[i], 1.0, W, np.zeros(3))
            for i in matched
        }
        src = np.stack([_camera_center(pred_w[i]) for i in matched])
        dst = np.stack([_camera_center(gt_poses[i]) for i in matched])
        s, R_res, ts = _umeyama(src, dst)
        Rs = R_res @ W
        metrics["pose/alignment_scale"] = float(s)

        # GT scale for relative center-error normalization.
        gt_centers = dst
        diffs = gt_centers[None, :, :] - gt_centers[:, None, :]
        gt_baseline = float(
            np.linalg.norm(diffs, axis=-1)[np.triu_indices(len(matched), 1)].mean()
        )
        metrics["pose/gt_mean_baseline"] = gt_baseline

        rot_errs, trans_errs, trans_errs_rel = [], [], []
        for i in matched:
            aligned = _apply_sim3_to_pose(pred_poses[i], s, Rs, ts)
            rot_err = _rot_geodesic_deg(aligned[:3, :3], gt_poses[i][:3, :3])
            trans_err = float(np.linalg.norm(_camera_center(aligned) - _camera_center(gt_poses[i])))
            trans_rel = trans_err / gt_baseline if gt_baseline > 1e-12 else None
            rot_errs.append(rot_err)
            trans_errs.append(trans_err)
            if trans_rel is not None:
                trans_errs_rel.append(trans_rel)
            per_image.append(
                {
                    "index": i,
                    "rot_err_deg": rot_err,
                    "trans_err": trans_err,
                    "trans_err_rel": trans_rel,
                }
            )

        metrics["pose/abs_rot_err_mean_deg"] = float(np.mean(rot_errs))
        metrics["pose/abs_rot_err_median_deg"] = float(np.median(rot_errs))
        metrics["pose/abs_rot_err_max_deg"] = float(np.max(rot_errs))
        metrics["pose/abs_trans_err_mean"] = float(np.mean(trans_errs))
        metrics["pose/abs_trans_err_median"] = float(np.median(trans_errs))
        metrics["pose/abs_trans_err_max"] = float(np.max(trans_errs))
        if trans_errs_rel:
            metrics["pose/abs_trans_err_rel_mean"] = float(np.mean(trans_errs_rel))
        metrics["pose/aligned"] = 1.0

    # --- Pairwise relative pose errors (gauge-free) -----------------------
    rel_rot, rel_trans = [], []
    for a_i, i in enumerate(matched):
        for j in matched[a_i + 1 :]:
            Rp_i, tp_i = pred_poses[i][:3, :3], pred_poses[i][:3, 3]
            Rp_j, tp_j = pred_poses[j][:3, :3], pred_poses[j][:3, 3]
            Rg_i, tg_i = gt_poses[i][:3, :3], gt_poses[i][:3, 3]
            Rg_j, tg_j = gt_poses[j][:3, :3], gt_poses[j][:3, 3]

            R_rel_p = Rp_j @ Rp_i.T
            t_rel_p = tp_j - R_rel_p @ tp_i
            R_rel_g = Rg_j @ Rg_i.T
            t_rel_g = tg_j - R_rel_g @ tg_i

            r_err = _rot_geodesic_deg(R_rel_p, R_rel_g)
            t_err = _direction_err_deg(t_rel_p, t_rel_g)
            rel_rot.append(r_err)
            rel_trans.append(t_err)
            per_pair.append(
                {"i": i, "j": j, "rel_rot_err_deg": r_err, "rel_trans_err_deg": t_err}
            )

    if rel_rot:
        rel_rot_arr = np.asarray(rel_rot)
        rel_trans_arr = np.asarray(rel_trans)
        metrics["pose/num_rel_pairs"] = len(rel_rot)
        metrics["pose/rel_rot_err_mean_deg"] = float(rel_rot_arr.mean())
        metrics["pose/rel_rot_err_median_deg"] = float(np.median(rel_rot_arr))
        metrics["pose/rel_trans_err_mean_deg"] = float(rel_trans_arr.mean())
        metrics["pose/rel_trans_err_median_deg"] = float(np.median(rel_trans_arr))
        for thr in _POSE_AUC_THRESHOLDS:
            passed = float(np.mean((rel_rot_arr < thr) & (rel_trans_arr < thr)))
            metrics[f"pose/rel_auc_{thr}"] = passed
    return metrics, per_image, per_pair


def _completeness_metrics(
    ep: SceneRolloutEpisode,
    recon: dict[str, Any],
    gt_recon: dict[str, Any],
    scene: dict[str, Any],
    registered_indices: set[int] | None,
) -> dict[str, Any]:
    num_images = ep.num_images or len(ep.image_paths)
    num_registered = int(recon.get("num_registered") or 0)
    num_points3d = int(recon.get("num_points3d") or 0)
    gt_num = int(gt_recon.get("num_images") or num_images or 1)

    out = {
        "completeness/coverage": num_registered / max(num_images, 1),
        "completeness/scene_coverage": num_registered / max(gt_num, 1),
        "completeness/points_per_image": num_points3d / max(num_registered, 1),
    }
    if recon.get("mean_track_length") is not None:
        out["completeness/mean_track_length"] = float(recon["mean_track_length"])
    if recon.get("num_observations") is not None:
        out["completeness/num_observations"] = int(recon["num_observations"])

    # Overlap coverage: does the registered subset span the scene's overlap
    # structure?  Uses the GT overlap_matrix restricted to the episode's
    # image_indices (same convention as the S-GRPO oracle).
    om = scene.get("overlap_matrix")
    image_indices = scene.get("image_indices")
    if om is not None and image_indices is not None and registered_indices:
        try:
            om_arr = np.asarray(om, dtype=np.float64)
            sub = om_arr[np.ix_(image_indices, image_indices)]
            n = len(image_indices)
            iu = np.triu_indices(n, 1)
            if n >= 2:
                out["completeness/mean_overlap_all"] = float(sub[iu].mean())
            reg = sorted(i for i in registered_indices if 0 <= i < n)
            if len(reg) >= 2:
                iu_r = np.triu_indices(len(reg), 1)
                out["completeness/mean_overlap_registered"] = float(
                    sub[np.ix_(reg, reg)][iu_r].mean()
                )
            out["completeness/index_coverage"] = len(reg) / max(n, 1)
        except Exception as e:
            logger.debug(f"overlap coverage failed: {e}")
    return out


def _doppelganger_metrics(
    ep: SceneRolloutEpisode,
    recon: dict[str, Any],
    gt_keys: set[str],
) -> dict[str, Any]:
    checks = ep.doppelganger_checks or {}
    flagged = {k for k, v in checks.items() if _is_flagged_doppelganger(v)}

    out: dict[str, Any] = {
        "doppelganger/num_checks": len(checks),
        "doppelganger/num_flagged": len(flagged),
        "doppelganger/num_gt_pairs": len(gt_keys),
        "doppelganger/num_filtered": int(recon.get("num_doppelgangers_filtered") or 0),
        "doppelganger/num_present": int(recon.get("num_doppelgangers_present") or 0),
    }
    n_present = out["doppelganger/num_present"]
    if n_present:
        out["doppelganger/filter_rate"] = (
            out["doppelganger/num_filtered"] / n_present
        )

    if gt_keys:
        tp = len(flagged & gt_keys)
        fp = len(flagged - gt_keys)
        fn = len((gt_keys & set(checks)) - flagged)          # checked but not flagged
        tn = len(set(checks) - gt_keys - flagged)            # checked, correctly clear
        missed = len(gt_keys - set(checks))                  # never checked at all
        total_checked = len(checks)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        detection = tp / max(tp + fn + missed, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        out.update(
            {
                "doppelganger/tp": tp,
                "doppelganger/fp": fp,
                "doppelganger/fn": fn,
                "doppelganger/tn": tn,
                "doppelganger/missed_gt": missed,
                "doppelganger/precision": precision,
                "doppelganger/recall": recall,
                "doppelganger/f1": f1,
                "doppelganger/accuracy": (tp + tn) / max(total_checked, 1),
                # Fraction of GT doppelgangers actually excluded, counting
                # never-checked pairs as misses.
                "doppelganger/detection_rate": detection,
            }
        )
    return out


def _efficiency_metrics(ep: SceneRolloutEpisode, reg_fraction: float) -> dict[str, Any]:
    calls = ep.tool_calls or []
    results = ep.results or []
    non_done = [tc for tc in calls if getattr(tc, "tool", None) != "done"]
    n_errors = sum(1 for r in results if isinstance(r, dict) and r.get("error"))

    per_tool: dict[str, int] = {}
    for tc in calls:
        tool = getattr(tc, "tool", None) or "unknown"
        per_tool[tool] = per_tool.get(tool, 0) + 1

    n_aligned, n_redundant = _ntep_process_reward_counts(calls, results)

    n_calls = len(calls)
    out: dict[str, Any] = {
        "efficiency/num_tool_calls": n_calls,
        "efficiency/num_executed_calls": len(non_done),
        "efficiency/num_error_results": n_errors,
        "efficiency/error_rate": n_errors / max(len(results), 1),
        "efficiency/quality_per_call": reg_fraction / max(n_calls, 1),
        "efficiency/reward_per_call": float(ep.reward) / max(n_calls, 1),
        "efficiency/n_aligned_calls": n_aligned,
        "efficiency/n_redundant_calls": n_redundant,
        "efficiency/redundant_fraction": n_redundant / max(len(non_done), 1),
        "efficiency/pairs_matched": len(ep.pair_matches or {}),
    }
    for tool, count in sorted(per_tool.items()):
        out[f"efficiency/calls_{tool}"] = count
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_scene_metrics(
    ep: SceneRolloutEpisode,
    gt_scene: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute all scene-level metrics for one episode.

    Args:
        ep: finished ``SceneRolloutEpisode`` (or any object with the same
            fields: ``recon_result``, ``tool_calls``, ``results``,
            ``doppelganger_checks``, ``image_paths``, ``num_images``,
            ``reward``, ``reward_components``).
        gt_scene: full scene dict from ``SceneDataset.from_megadepth`` or a
            bare ``gt_recon`` dict.  Falls back to ``ep.gt_recon``.

    Returns:
        Flat dict of ``group/metric`` scalars (JSON/wandb-safe) plus
        ``per_image``/``per_pair`` detail lists.
    """
    gt_recon, scene = _split_gt_scene(gt_scene, ep)
    recon = ep.recon_result or {}
    lookup = _index_lookup(ep)

    gt_num_images = int(gt_recon.get("num_images") or 0)
    metrics: dict[str, Any] = {
        "scene/scene_id": ep.scene_id,
        "scene/reward": float(ep.reward),
        "scene/done": float(bool(ep.done)),
    }

    # Registration
    metrics.update(_registration_metrics(ep, recon, gt_num_images))
    reg_fraction = metrics["registration/registered_fraction"]

    # Poses
    pred_poses = _extract_pred_poses(recon, lookup)
    gt_poses = _extract_gt_poses(gt_recon, lookup)
    pose_m, per_image, per_pair = _pose_metrics(pred_poses, gt_poses, recon)
    metrics.update(pose_m)

    # Registered index set (for overlap coverage) — from pred poses or an
    # explicit name list in recon_result.
    registered_indices: set[int] | None = set(pred_poses) if pred_poses else None
    if registered_indices is None:
        names = (
            recon.get("registered_images")
            or recon.get("image_names")
            or recon.get("images")
        )
        if isinstance(names, (list, tuple)):
            registered_indices = {
                idx
                for idx in (_resolve_image_index(n, lookup) for n in names)
                if idx is not None
            }

    # Completeness
    metrics.update(
        _completeness_metrics(ep, recon, gt_recon, scene, registered_indices)
    )

    # Doppelgangers
    gt_keys = _gt_doppelganger_keys(scene, gt_recon, lookup)
    metrics.update(_doppelganger_metrics(ep, recon, gt_keys))

    # Tool efficiency
    metrics.update(_efficiency_metrics(ep, reg_fraction))

    # Reward components pass-through
    for key, value in (ep.reward_components or {}).items():
        if isinstance(value, (int, float)) and np.isfinite(value):
            metrics[f"reward/{key}"] = float(value)

    # Detail lists (not logged as scalars — for inspection/debugging)
    metrics["per_image"] = per_image
    metrics["per_pair"] = per_pair
    return metrics


def _numeric_items(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        k: float(v)
        for k, v in metrics.items()
        if isinstance(v, (int, float))
        and not isinstance(v, bool)
        and np.isfinite(v)
    }


def compute_efficiency_frontier(
    per_scene_metrics: list[dict[str, Any]],
) -> list[dict[str, float]]:
    """Pareto frontier of (num_tool_calls, registered_fraction) across scenes.

    Points sorted by tool calls; keeps each call-count's best quality and
    then only the running max — the efficiency frontier of the evaluated
    policy.  Compare frontiers across methods/checkpoints for the quality-
    per-tool-call trade-off.
    """
    best_per_calls: dict[int, float] = {}
    for m in per_scene_metrics:
        calls = int(m.get("efficiency/num_tool_calls") or 0)
        quality = float(m.get("registration/registered_fraction") or 0.0)
        best_per_calls[calls] = max(best_per_calls.get(calls, 0.0), quality)

    frontier: list[dict[str, float]] = []
    best_q = -1.0
    for calls in sorted(best_per_calls):
        q = best_per_calls[calls]
        if q > best_q:
            frontier.append({"num_tool_calls": calls, "registered_fraction": q})
            best_q = q
    return frontier


def aggregate_scene_metrics(
    per_scene_metrics: list[dict[str, Any]],
    prefix: str = "scene_eval/",
) -> dict[str, Any]:
    """Aggregate per-scene metric dicts into a summary for logging.

    Emits ``prefix + mean_<key>`` / ``median_<key>`` for every numeric leaf,
    plus headline rates (success, full registration, pose availability) and
    the tool-call efficiency frontier.
    """
    summary: dict[str, Any] = {f"{prefix}num_scenes": len(per_scene_metrics)}
    if not per_scene_metrics:
        return summary

    by_key: dict[str, list[float]] = {}
    for m in per_scene_metrics:
        for k, v in _numeric_items(m).items():
            by_key.setdefault(k, []).append(v)

    for key, values in sorted(by_key.items()):
        if not values:
            continue
        summary[f"{prefix}mean_{key.replace('/', '_')}"] = float(np.mean(values))
        summary[f"{prefix}median_{key.replace('/', '_')}"] = float(np.median(values))

    # Headline rates
    summary[f"{prefix}success_rate"] = float(
        np.mean([m.get("registration/success", 0.0) for m in per_scene_metrics])
    )
    summary[f"{prefix}full_registration_rate"] = float(
        np.mean([m.get("registration/full", 0.0) for m in per_scene_metrics])
    )
    summary[f"{prefix}pose_available_rate"] = float(
        np.mean([m.get("pose/available", 0.0) for m in per_scene_metrics])
    )
    summary[f"{prefix}reward_mean"] = float(
        np.mean([m.get("scene/reward", 0.0) for m in per_scene_metrics])
    )
    summary[f"{prefix}efficiency_frontier"] = compute_efficiency_frontier(
        per_scene_metrics
    )
    summary[f"{prefix}per_scene"] = [
        {
            "scene_id": m.get("scene/scene_id"),
            "reward": m.get("scene/reward"),
            "registered_fraction": m.get("registration/registered_fraction"),
            "num_tool_calls": m.get("efficiency/num_tool_calls"),
            "rel_auc_10": m.get("pose/rel_auc_10"),
            "doppelganger_f1": m.get("doppelganger/f1"),
        }
        for m in per_scene_metrics
    ]
    return summary
