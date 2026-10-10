"""Feed-forward pose-prior adapters: point-map models (VGGT/Pi3/MapAnything) → correspondences.

These functions are pure numpy so they can be unit-tested without torch or
model weights. Model-specific code (loading, preprocessing, forward pass)
stays in the tool server; this module only converts model outputs into the
same shape the existing matcher backends produce.

Conventions (must match ``agentic_sfm.geometry``):
  - Relative pose maps camera-A coordinates to camera-B coordinates:
    ``X_b = R @ X_a + t`` with ``R`` a rotation matrix and ``t`` unit-norm
    (the convention ``cv2.recoverPose`` returns and ``estimate_relative_pose``
    forwards unchanged).
  - Camera extrinsics are world-to-camera: ``X_cam = R @ X_world + t``.
  - Pixel coordinates are ``(x, y)`` in the frame of the point map they were
    read from. The caller is responsible for scaling to original pixels when
    the model ran at a different resolution.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "reciprocal_nn_matches",
    "extrinsics_to_relative_pose",
    "relative_pose_angular_error",
]


def _flatten_points(points: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """(H, W, 3) → ((H*W, 3) float64, (H, W)).

    Only the dense point-map layout is accepted: an (N, 3) array carries no
    pixel grid, so returned ``(x, y)`` coordinates would be meaningless and
    silently wrong downstream. Callers with sparse point lists must scatter
    them into an (H, W, 3) map first.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 3 and pts.shape[2] == 3:
        hw = (pts.shape[0], pts.shape[1])
        pts = pts.reshape(-1, 3)
    else:
        raise ValueError(
            f"points must be a dense (H, W, 3) point map so returned pixel "
            f"coordinates have a frame; got shape {pts.shape}. Scatter sparse "
            f"(N, 3) points into an (H, W, 3) map before calling."
        )
    return pts, hw


def _nn_query(tree, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """k=1 nearest neighbour with an explicit, documented tie-break.

    cKDTree.query does not specify which of several equidistant points it
    returns, so we resolve ties ourselves: after finding the nearest
    distance, gather every point within that distance (plus a tiny epsilon
    for float noise) and take the smallest index. This makes the function's
    determinism guarantee rest on our code, not on scipy internals.
    """
    dist, _ = tree.query(pts, k=1, workers=-1)
    balls = tree.query_ball_point(
        pts, r=dist * (1.0 + 1e-9) + 1e-12, workers=-1
    )
    idx = np.fromiter((min(b) for b in balls), dtype=np.int64, count=len(pts))
    return dist, idx


def _select_indices(
    conf: np.ndarray | None, num_points: int, max_points: int
) -> np.ndarray:
    """Deterministic subsampling of point-map pixels.

    With confidence available, keep the highest-confidence points (ties broken
    by index) — that is where the model is actually predicting geometry. With
    no confidence, stride uniformly so the subset spreads over the image.
    """
    if num_points <= max_points:
        return np.arange(num_points, dtype=np.int64)
    if conf is None:
        stride = max(1, num_points // max_points)
        return np.arange(0, num_points, stride, dtype=np.int64)[:max_points]
    order = np.argsort(-np.asarray(conf, dtype=np.float64), kind="stable")
    return np.sort(order[:max_points])


def reciprocal_nn_matches(
    points_a: np.ndarray,
    points_b: np.ndarray,
    conf_a: np.ndarray | None = None,
    conf_b: np.ndarray | None = None,
    *,
    conf_threshold: float | None = None,
    dist_threshold: float | None = None,
    max_points: int = 20000,
) -> tuple[np.ndarray, np.ndarray]:
    """Reciprocal nearest-neighbour correspondences in 3D point-map space.

    For a pair of images the model predicts, per pixel, a 3D point in a shared
    frame. Two pixels correspond if they are each other's nearest neighbour in
    that frame. Returns pixel coordinates ``(N, 2)`` as ``(x, y)`` float64 in
    each image's point-map frame, in mutual-pair order.

    ``points_a`` / ``points_b`` must be dense ``(H, W, 3)`` maps; an ``(N, 3)``
    array has no pixel frame and raises ``ValueError``. Entries with any
    NaN/inf coordinate are treated as background ("no point here") and are
    masked out before matching — sparse maps should use NaN (or inf) padding,
    not a finite sentinel, which would be matchable.

    Deterministic: same inputs give the same pairs, so cached tool outputs are
    reproducible across rollouts.
    """
    if not isinstance(max_points, (int, np.integer)) or max_points < 1:
        raise ValueError(
            f"max_points must be a positive integer, got {max_points!r}"
        )

    pts_a, (h_a, w_a) = _flatten_points(points_a)
    pts_b, (h_b, w_b) = _flatten_points(points_b)

    conf_a_flat = None if conf_a is None else np.asarray(conf_a, dtype=np.float64).reshape(-1)
    conf_b_flat = None if conf_b is None else np.asarray(conf_b, dtype=np.float64).reshape(-1)

    keep_a = np.arange(len(pts_a), dtype=np.int64)
    keep_b = np.arange(len(pts_b), dtype=np.int64)
    # Drop non-finite (NaN/inf) points up front: cKDTree does not reject them
    # and NaN coordinates silently poison every distance query, which would
    # turn background pixels of a sparse map into bogus matches.
    keep_a = keep_a[np.isfinite(pts_a[keep_a]).all(axis=1)]
    keep_b = keep_b[np.isfinite(pts_b[keep_b]).all(axis=1)]
    if conf_threshold is not None:
        if conf_a_flat is not None:
            keep_a = keep_a[conf_a_flat[keep_a] >= conf_threshold]
        if conf_b_flat is not None:
            keep_b = keep_b[conf_b_flat[keep_b] >= conf_threshold]
    if len(keep_a) == 0 or len(keep_b) == 0:
        return np.zeros((0, 2)), np.zeros((0, 2))

    sel_a = keep_a[
        _select_indices(
            None if conf_a_flat is None else conf_a_flat[keep_a], len(keep_a), max_points
        )
    ]
    sel_b = keep_b[
        _select_indices(
            None if conf_b_flat is None else conf_b_flat[keep_b], len(keep_b), max_points
        )
    ]

    from scipy.spatial import cKDTree

    tree_b = cKDTree(pts_b[sel_b])
    dist_ab, nn_ab = _nn_query(tree_b, pts_a[sel_a])
    tree_a = cKDTree(pts_a[sel_a])
    _, nn_ba = _nn_query(tree_a, pts_b[sel_b])

    mutual = nn_ba[nn_ab] == np.arange(len(sel_a), dtype=np.int64)
    if dist_threshold is not None:
        mutual &= dist_ab <= dist_threshold
    if not mutual.any():
        return np.zeros((0, 2)), np.zeros((0, 2))

    idx_a = sel_a[mutual]
    idx_b = sel_b[nn_ab[mutual]]

    pts_a_px = np.stack([idx_a % w_a, idx_a // w_a], axis=1).astype(np.float64)
    pts_b_px = np.stack([idx_b % w_b, idx_b // w_b], axis=1).astype(np.float64)
    return pts_a_px, pts_b_px


def extrinsics_to_relative_pose(
    ext_a: np.ndarray, ext_b: np.ndarray
) -> dict[str, list]:
    """World-to-camera extrinsics → relative pose in ``estimate_relative_pose`` form.

    ``ext`` is 3x4 (``[R | t]``, world-to-camera) or 4x4. Returns
    ``{"R": 3x3, "t": 3}`` with ``X_b = R @ X_a + t`` and ``t`` unit-norm, so
    the result is directly comparable to the MAGSAC-based pose from the
    classical matcher backends.
    """
    ea = np.asarray(ext_a, dtype=np.float64).reshape(3, 4)
    eb = np.asarray(ext_b, dtype=np.float64).reshape(3, 4)
    r_a, t_a = ea[:, :3], ea[:, 3]
    r_b, t_b = eb[:, :3], eb[:, 3]

    r_rel = r_b @ r_a.T
    t_rel = t_b - r_rel @ t_a
    norm = float(np.linalg.norm(t_rel))
    if norm < 1e-12:
        t_rel = np.zeros(3)
    else:
        t_rel = t_rel / norm
    return {"R": r_rel.tolist(), "t": t_rel.tolist()}


def relative_pose_angular_error(
    R_gt: np.ndarray, t_gt: np.ndarray, R_est: np.ndarray, t_est: np.ndarray
) -> tuple[float, float]:
    """IMC-style relative-pose error in degrees: (rotation, translation-direction).

    Rotation error is the geodesic angle between the two rotations. Translation
    error is the angle between the two (unit) translation vectors, without a
    sign flip — both ``recoverPose`` and the GT convention fix cheirality, so a
    sign flip would hide a real failure. Callers take the worst of the two.
    """
    r_gt = np.asarray(R_gt, dtype=np.float64).reshape(3, 3)
    r_est = np.asarray(R_est, dtype=np.float64).reshape(3, 3)
    trace = np.clip((np.trace(r_gt.T @ r_est) - 1.0) / 2.0, -1.0, 1.0)
    rot_err = float(np.degrees(np.arccos(trace)))

    t_gt = np.asarray(t_gt, dtype=np.float64).reshape(3)
    t_est = np.asarray(t_est, dtype=np.float64).reshape(3)
    n_gt = float(np.linalg.norm(t_gt))
    n_est = float(np.linalg.norm(t_est))
    if n_gt < 1e-12 or n_est < 1e-12:
        return rot_err, float("nan")
    cos = float(np.clip(np.dot(t_gt / n_gt, t_est / n_est), -1.0, 1.0))
    return rot_err, float(np.degrees(np.arccos(cos)))
