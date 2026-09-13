"""Geometry helpers: MAGSAC pose, crop-adjusted K, match quality, keep-best."""

from __future__ import annotations

from typing import Any

import numpy as np


def camera_matrix(
    size_wh: tuple[int, int],
    K: np.ndarray | list | None = None,
) -> np.ndarray:
    """3x3 intrinsics. Falls back to a pinhole guess from image size."""
    if K is not None:
        arr = np.asarray(K, dtype=np.float64).reshape(3, 3)
        if np.isfinite(arr).all() and arr[0, 0] > 1.0:
            return arr
    w, h = float(size_wh[0]), float(size_wh[1])
    f = 1.2 * max(w, h)
    return np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def adjust_K_for_crop(
    K: np.ndarray | list | None,
    origin_xy: tuple[float, float],
    size_wh: tuple[int, int] | None = None,
) -> np.ndarray | None:
    """Shift principal point when matching a crop of the original image.

    ``K`` is assumed to be the full-image camera matrix. If ``K`` is None,
    return None so the caller can guess from the *current* crop size.
    """
    if K is None:
        return None
    arr = camera_matrix(size_wh or (1, 1), K).copy()
    arr[0, 2] -= float(origin_xy[0])
    arr[1, 2] -= float(origin_xy[1])
    return arr


def estimate_relative_pose(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    size_a: tuple[int, int],
    size_b: tuple[int, int],
    K_a: np.ndarray | list | None = None,
    K_b: np.ndarray | list | None = None,
    threshold: float = 1.0,
) -> dict[str, Any]:
    """Relative pose via MAGSAC essential matrix. Keypoints in original pixels.

    Returns num_inliers, inlier_ratio, pose, plus geometric-consistency
    diagnostics used by the doppelganger detector:
      - ``mean_inlier_residual``: mean Sampson *distance* of inliers in
        normalized image coordinates (sqrt of the squared Sampson error).
      - ``residual_units``: mean residual divided by the RANSAC threshold.
        Values ~1 mean inliers sit right at the verification boundary —
        typical of degenerate fits on doppelganger pairs.
    """
    pts_a = np.asarray(pts_a, dtype=np.float64).reshape(-1, 2)
    pts_b = np.asarray(pts_b, dtype=np.float64).reshape(-1, 2)
    empty: dict[str, Any] = {
        "num_inliers": 0,
        "inlier_ratio": 0.0,
        "pose": None,
        "mean_inlier_residual": None,
        "residual_units": None,
    }
    if len(pts_a) < 8 or len(pts_b) < 8:
        return empty

    try:
        import cv2
    except ImportError:
        return empty

    Ka = camera_matrix(size_a, K_a)
    Kb = camera_matrix(size_b, K_b)
    pts_a_cv = pts_a.reshape(-1, 1, 2)
    pts_b_cv = pts_b.reshape(-1, 1, 2)
    n_a = cv2.undistortPoints(pts_a_cv, Ka, None)
    n_b = cv2.undistortPoints(pts_b_cv, Kb, None)
    f = 0.5 * (Ka[0, 0] + Kb[0, 0])
    norm_thresh = float(threshold) / max(f, 1.0)
    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    E, mask = cv2.findEssentialMat(
        n_a, n_b, focal=1.0, pp=(0.0, 0.0),
        method=method, threshold=norm_thresh, prob=0.999,
    )
    if E is None or mask is None:
        return empty

    num_inliers = int(mask.sum())
    _, R, t, _ = cv2.recoverPose(E, n_a, n_b, np.eye(3), mask=mask)

    # Mean Sampson distance of the inlier correspondences (normalized coords).
    inlier_idx = mask.ravel().astype(bool)
    mean_resid = None
    resid_units = None
    if inlier_idx.any():
        E3 = E[:3] if E.ndim == 2 and E.shape[0] >= 3 else E
        x1 = np.concatenate([n_a.reshape(-1, 2), np.ones((len(n_a), 1))], axis=1)
        x2 = np.concatenate([n_b.reshape(-1, 2), np.ones((len(n_b), 1))], axis=1)
        Ex1 = (E3 @ x1.T).T
        Etx2 = (E3.T @ x2.T).T
        numer = (x2 * Ex1).sum(axis=1) ** 2
        denom = (
            Ex1[:, 0] ** 2 + Ex1[:, 1] ** 2 + Etx2[:, 0] ** 2 + Etx2[:, 1] ** 2
        )
        sampson_sq = numer / np.maximum(denom, 1e-24)
        mean_resid = float(np.sqrt(sampson_sq[inlier_idx]).mean())
        resid_units = mean_resid / max(norm_thresh, 1e-12)

    return {
        "num_inliers": num_inliers,
        "inlier_ratio": num_inliers / max(len(pts_a), 1),
        "pose": {"R": R.tolist(), "t": t.flatten().tolist()},
        "mean_inlier_residual": mean_resid,
        "residual_units": resid_units,
    }


def match_quality(result: dict[str, Any] | None) -> float:
    """Scalar for keep-best: inliers plus ratio. Errors score below zero."""
    if not result or result.get("error"):
        return -1.0
    n = float(result.get("num_inliers") or 0)
    r = float(result.get("inlier_ratio") or 0.0)
    return n + 50.0 * r


# Heuristic overlap boxes for oracle SFT (normalized [x1,y1,x2,y2]).
ORACLE_CROP_BOXES: list[list[float]] = [
    [0.15, 0.15, 0.85, 0.85],
    [0.00, 0.00, 0.70, 0.70],
    [0.30, 0.00, 1.00, 0.70],
    [0.00, 0.30, 0.70, 1.00],
    [0.30, 0.30, 1.00, 1.00],
    [0.20, 0.00, 0.80, 1.00],
    [0.00, 0.20, 1.00, 0.80],
]


def iter_oracle_crops() -> list[tuple[str, list[float], str]]:
    """(crop_image_id, bbox, other_image_id) for heuristic overlap search."""
    jobs: list[tuple[str, list[float], str]] = []
    for bbox in ORACLE_CROP_BOXES:
        jobs.append(("img_a", list(bbox), "img_b"))
        jobs.append(("img_b", list(bbox), "img_a"))
    return jobs


def keep_best_match(
    current: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Keep the higher-quality match result."""
    if candidate is None:
        return current
    if match_quality(candidate) >= match_quality(current):
        return candidate
    return current


def k_for_image(
    K_full: np.ndarray | list | None,
    origin_xy: tuple[float, float] | list | None,
    size_wh: tuple[int, int],
) -> np.ndarray | list | None:
    """Return K for the image actually being matched (full frame or crop)."""
    if origin_xy is None:
        return K_full
    adj = adjust_K_for_crop(K_full, (float(origin_xy[0]), float(origin_xy[1])), size_wh)
    return adj.tolist() if adj is not None else K_full


def crop_image_id(result: dict[str, Any]) -> str | None:
    """Unify crop_id / cropped_image_id from either tool server."""
    return result.get("cropped_image_id") or result.get("crop_id")


def crop_pil_from_result(result: dict[str, Any] | None):
    """Load a crop as PIL from ``path`` or ``image_b64`` (nested ``crop`` ok)."""
    if not result:
        return None
    import base64
    import io
    import os

    from PIL import Image

    candidates = [result]
    nested = result.get("crop")
    if isinstance(nested, dict):
        candidates.append(nested)
    for blob in candidates:
        path = blob.get("path")
        if path and os.path.exists(path):
            try:
                return Image.open(path).convert("RGB")
            except Exception:
                pass
        b64 = blob.get("image_b64")
        if b64:
            try:
                return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
            except Exception:
                pass
    return None
