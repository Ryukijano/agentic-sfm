"""Tests for agentic_sfm.pose_prior using a real synthetic two-view scene.

Scene construction: 3D world points are projected through two pinhole
cameras with different intrinsics/extrinsics, then rasterized into dense
(H, W, 3) point maps. Non-visited pixels are NaN — the module documents
NaN/inf entries as "background, no point here" and masks them before
building the KD-tree (finite sentinels would be matchable, so NaN is the
required padding). The stored 3D value is the point in the shared model
frame (we use camera-A coordinates), matching what VGGT-style point-map
models emit.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentic_sfm.pose_prior import (
    extrinsics_to_relative_pose,
    reciprocal_nn_matches,
    relative_pose_angular_error,
)
from agentic_sfm.geometry import estimate_relative_pose

H, W = 480, 640
N_PTS = 300
# Correspondences are rasterized at integer pixels, so the essential-matrix
# path would normally see ~0.5px quantization noise (amplified to >0.01 deg
# of rotation error). To keep the convention test clean we build pairs that
# are *epipolar-exact*: pick integer pixels in A, take the integer pixel in B
# closest to the epipolar line (<EPI_TOL px away), and place the 3D point on
# ray A. Residual reprojection noise is then <EPI_TOL px, not 0.5 px.
EPI_TOL = 0.001


def _rot(axis, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


# Fixed seed: the epipolar-exact generation leaves ~3e-4 px of sub-pixel
# residual, and MAGSAC's stochastic fit amplifies it by a seed-dependent
# factor; this seed keeps the rotation agreement ~30x under the 0.01 deg
# assertion so the test is stable, not marginal.
def _make_scene(seed=2):
    """Two INDEPENDENT cameras; ext are world-to-camera 3x4.

    The baseline has norm ~2.7 (deliberately not 1) so the unit-norm
    normalisation inside ``extrinsics_to_relative_pose`` is exercised.

    Correspondences come from random world points projected through the two
    extrinsics/intrinsics independently — never from a pre-chosen relative
    pose. The only adjustment is sub-pixel: after projecting world point P
    into camera A and rounding to pixel q_a, we pick the integer pixel q_b
    closest to the epipolar line ``F @ q_a`` (<EPI_TOL px away) and re-solve
    the depth on ray A so its projection in B lands on the line (see EPI_TOL
    comment above — plain rounding would inject ~0.5px noise that the
    essential-matrix path amplifies past the 0.01-degree assertion). F is
    derived directly from ext_a/ext_b, not from the function under test, so
    the cross-check against MAGSAC remains an independent code path.
    """
    rng = np.random.default_rng(seed)

    R_a = _rot("y", 8.0) @ _rot("x", -4.0)
    t_a = np.array([0.15, -0.05, 0.4])
    ext_a = np.hstack([R_a, t_a[:, None]])

    R_b = _rot("z", 6.0) @ _rot("y", -5.0) @ R_a
    t_b = np.array([1.8, 1.2, -1.5])  # norm ~2.66 — NOT unit length
    ext_b = np.hstack([R_b, t_b[:, None]])
    # Closed-form relative pose (what the function should reproduce):
    R_rel = R_b @ R_a.T
    t_vec = t_b - R_rel @ t_a
    baseline = float(np.linalg.norm(t_vec))
    assert abs(baseline - 1.0) > 0.5, "scene should exercise normalisation"
    t_hat = t_vec / baseline

    K_a = np.array([[1400.0, 0.0, W / 2.0], [0.0, 1400.0, H / 2.0], [0.0, 0.0, 1.0]])
    K_b = np.array(
        [[1500.0, 0.0, W / 2.0 - 3.0], [0.0, 1450.0, H / 2.0 + 2.0], [0.0, 0.0, 1.0]]
    )

    K_a_inv = np.linalg.inv(K_a)
    K_b_inv = np.linalg.inv(K_b)
    F = K_b_inv.T @ _skew(t_vec) @ R_rel @ K_a_inv

    Xg, Yg = np.meshgrid(np.arange(W), np.arange(H))
    map_a = np.full((H, W, 3), np.nan)
    map_b = np.full((H, W, 3), np.nan)
    px_a_list, px_b_list, Xa_list = [], [], []
    used_a, used_b = set(), set()
    attempts = 0
    while len(Xa_list) < N_PTS and attempts < 2_000_000:
        attempts += 1
        # Independent random world point → integer pixel in A.
        P = np.array(
            [rng.uniform(-1.5, 1.5), rng.uniform(-1.1, 1.1), rng.uniform(1.5, 10.0)]
        )
        Xa_seed = R_a @ P + t_a
        if Xa_seed[2] < 0.5:
            continue
        ua = (K_a @ (Xa_seed / Xa_seed[2]))[:2]
        xa, ya = np.round(ua).astype(int)
        if not (0 <= xa < W and 0 <= ya < H) or (int(xa), int(ya)) in used_a:
            continue
        h = np.array([xa, ya, 1.0])
        l = F @ h  # epipolar line in B
        denom = math.hypot(l[0], l[1])
        if denom < 1e-12:
            continue
        dist = np.abs(l[0] * Xg + l[1] * Yg + l[2]) / denom
        j = int(np.argmin(dist))
        if dist.flat[j] > EPI_TOL:
            continue
        xb, yb = j % W, j // W
        if (xb, yb) in used_b:
            continue
        # Closest point on the line to (xb, yb), then depth d on ray A whose
        # projection in B hits it exactly (reproj error = dist, < EPI_TOL).
        s = (l[0] * xb + l[1] * yb + l[2]) / denom**2
        px, py = xb - l[0] * s, yb - l[1] * s
        a = R_rel @ (K_a_inv @ h)  # X_b = d*a + t_vec
        pn = K_b_inv @ np.array([px, py, 1.0])
        m = a[:2] - pn[:2] * a[2]
        d = float(((pn[:2] * t_vec[2] - t_vec[:2]) * m).sum() / (m * m).sum())
        X_a = d * (K_a_inv @ h)
        X_b_resolved = R_rel @ X_a + t_vec
        if not (1.5 < X_a[2] < 15.0 and X_b_resolved[2] > 0.5):
            continue
        used_a.add((int(xa), int(ya)))
        used_b.add((xb, yb))
        map_a[ya, xa] = X_a  # shared model frame = camera-A coords
        map_b[yb, xb] = X_a
        px_a_list.append((int(xa), int(ya)))
        px_b_list.append((xb, yb))
        Xa_list.append(X_a)

    assert len(Xa_list) == N_PTS, "scene generation failed to find enough pairs"
    return {
        "ext_a": ext_a,
        "ext_b": ext_b,
        "K_a": K_a,
        "K_b": K_b,
        "map_a": map_a,
        "map_b": map_b,
        "X_a": np.array(Xa_list),
        "px_a": np.array(px_a_list),
        "px_b": np.array(px_b_list),
        "R_rel": R_rel,
        "t_hat": t_hat,
        "t_vec": t_vec,
        "baseline": baseline,
    }


SCENE = _make_scene()


# ---------------------------------------------------------------- matching

def test_reciprocal_nn_recovers_true_pairs():
    m_a, m_b = reciprocal_nn_matches(SCENE["map_a"], SCENE["map_b"])
    got = {(tuple(a), tuple(b)) for a, b in zip(m_a.astype(int), m_b.astype(int))}
    want = {
        (tuple(a), tuple(b))
        for a, b in zip(
            np.round(SCENE["px_a"]).astype(int), np.round(SCENE["px_b"]).astype(int)
        )
    }
    assert len(m_a) == N_PTS and m_a.shape == (N_PTS, 2) and m_b.shape == (N_PTS, 2)
    assert got == want


def test_reciprocal_nn_deterministic_bitwise():
    a1, b1 = reciprocal_nn_matches(SCENE["map_a"], SCENE["map_b"])
    a2, b2 = reciprocal_nn_matches(SCENE["map_a"], SCENE["map_b"])
    assert np.array_equal(a1, a2) and np.array_equal(b1, b2)
    # Not just self-consistent: the pairs must be the true correspondences,
    # so a silent always-empty or permuted output cannot pass.
    want = {
        (tuple(a), tuple(b))
        for a, b in zip(SCENE["px_a"].astype(int), SCENE["px_b"].astype(int))
    }
    got = {(tuple(a), tuple(b)) for a, b in zip(a1.astype(int), b1.astype(int))}
    assert got == want


def test_empty_maps_return_empty():
    empty = np.full((H, W, 3), np.nan)
    m_a, m_b = reciprocal_nn_matches(empty, empty)
    assert m_a.shape == (0, 2) and m_b.shape == (0, 2)
    m_a, m_b = reciprocal_nn_matches(np.zeros((0, W, 3)), np.zeros((0, W, 3)))
    assert m_a.shape == (0, 2) and m_b.shape == (0, 2)


def test_conf_threshold_filtering_everything():
    conf = np.zeros((H, W))
    m_a, m_b = reciprocal_nn_matches(
        SCENE["map_a"], SCENE["map_b"], conf, conf, conf_threshold=0.5
    )
    assert m_a.shape == (0, 2) and m_b.shape == (0, 2)


def test_max_points_caps_each_side_and_keeps_known_pair():
    cap = 10
    # Confidence forces one specific true correspondence into the survivor
    # set: only its two pixels get conf=1, everything else gets 0.
    conf_a = np.zeros((H, W))
    conf_b = np.zeros((H, W))
    qa = tuple(SCENE["px_a"][17].astype(int))
    qb = tuple(SCENE["px_b"][17].astype(int))
    conf_a[qa[1], qa[0]] = 1.0
    conf_b[qb[1], qb[0]] = 1.0
    m_a, m_b = reciprocal_nn_matches(
        SCENE["map_a"], SCENE["map_b"], conf_a, conf_b, max_points=cap
    )
    assert len(m_a) <= cap and len(m_b) <= cap
    assert len(m_a) > 0  # a real pair must survive the cap, not just be bounded
    assert (qa, qb) in {
        (tuple(a), tuple(b)) for a, b in zip(m_a.astype(int), m_b.astype(int))
    }


def test_max_points_zero_raises():
    with pytest.raises(ValueError, match="max_points"):
        reciprocal_nn_matches(SCENE["map_a"], SCENE["map_b"], max_points=0)


def test_dist_threshold_rejects_moved_point():
    # Small dedicated scene: a's point p1 has b's copy moved +0.4 in x — far
    # enough to exceed dist_threshold but still its mutual nearest neighbour.
    map_a = np.full((4, 4, 3), np.nan)
    map_b = np.full((4, 4, 3), np.nan)
    map_a[0, 0] = [0.0, 0.0, 5.0]
    map_a[3, 3] = [4.0, 0.0, 5.0]
    map_b[1, 1] = [0.4, 0.0, 5.0]  # moved copy of p1 (dist 0.4)
    map_b[2, 2] = [4.0, 0.0, 5.0]  # exact copy of p2
    m_a, m_b = reciprocal_nn_matches(map_a, map_b)
    pairs = {(tuple(a), tuple(b)) for a, b in zip(m_a.astype(int), m_b.astype(int))}
    assert ((0, 0), (1, 1)) in pairs  # present WITHOUT the threshold
    m_a2, m_b2 = reciprocal_nn_matches(map_a, map_b, dist_threshold=0.2)
    pairs2 = {(tuple(a), tuple(b)) for a, b in zip(m_a2.astype(int), m_b2.astype(int))}
    assert ((0, 0), (1, 1)) not in pairs2  # absent WITH it
    assert ((3, 3), (2, 2)) in pairs2  # exact pair still passes the threshold


def test_tie_break_prefers_smallest_flat_index():
    # a's origin point is equidistant from b's [-1,0,0] (flat idx 0) and
    # [1,0,0] (flat idx 2): explicit tie-break must pick the smaller index.
    map_a = np.full((1, 1, 3), np.nan)
    map_b = np.full((1, 3, 3), np.nan)
    map_a[0, 0] = [0.0, 0.0, 0.0]
    map_b[0, 0] = [-1.0, 0.0, 0.0]
    map_b[0, 2] = [1.0, 0.0, 0.0]
    m_a, m_b = reciprocal_nn_matches(map_a, map_b)
    assert m_a.tolist() == [[0.0, 0.0]]
    assert m_b.tolist() == [[0.0, 0.0]]  # b flat index 0, not 2


def test_inf_entries_masked_like_nan():
    map_b = SCENE["map_b"].copy()
    map_b[~np.isfinite(map_b).all(axis=2)] = np.inf  # inf padding instead of NaN
    m_a, m_b = reciprocal_nn_matches(SCENE["map_a"], map_b)
    assert len(m_a) == N_PTS


def test_nd2_input_raises():
    with pytest.raises(ValueError, match="H, W, 3"):
        reciprocal_nn_matches(np.zeros((10, 3)), SCENE["map_b"])
    with pytest.raises(ValueError, match="H, W, 3"):
        reciprocal_nn_matches(SCENE["map_a"], np.zeros((10, 3)))


# ------------------------------------------------------------------ poses

def test_extrinsics_to_relative_pose_matches_closed_form():
    # Baseline norm is ~2.7, so the unit-t normalisation is really exercised.
    rel = extrinsics_to_relative_pose(SCENE["ext_a"], SCENE["ext_b"])
    R = np.asarray(rel["R"])
    t = np.asarray(rel["t"])

    R_a, t_a = SCENE["ext_a"][:, :3], SCENE["ext_a"][:, 3]
    R_b, t_b = SCENE["ext_b"][:, :3], SCENE["ext_b"][:, 3]
    R_cf = R_b @ R_a.T
    t_cf = t_b - R_cf @ t_a
    np.testing.assert_allclose(R, R_cf, atol=1e-9)
    np.testing.assert_allclose(t, t_cf / np.linalg.norm(t_cf), atol=1e-9)
    assert abs(np.linalg.norm(t) - 1.0) < 1e-12

    # For every 3D point: X_b = R @ X_a + s*t with s = baseline length.
    X_a = SCENE["X_a"]
    P = np.linalg.solve(R_a, (X_a - t_a).T).T  # world points
    X_b = (R_b @ P.T).T + t_b                  # projected through ext_b only
    np.testing.assert_allclose(
        R @ X_a.T + SCENE["baseline"] * t[:, None], X_b.T, atol=1e-9
    )


def test_two_pose_paths_share_convention():
    m_a, m_b = reciprocal_nn_matches(SCENE["map_a"], SCENE["map_b"])
    res = estimate_relative_pose(
        m_a, m_b, (W, H), (W, H), K_a=SCENE["K_a"], K_b=SCENE["K_b"]
    )
    assert res["pose"] is not None
    rel = extrinsics_to_relative_pose(SCENE["ext_a"], SCENE["ext_b"])
    r_err, t_err = relative_pose_angular_error(
        rel["R"], rel["t"], res["pose"]["R"], res["pose"]["t"]
    )
    assert r_err < 0.01
    assert t_err < 0.5


# ---------------------------------------------------------- angular error

def _I():
    return np.eye(3)


def test_angular_error_identical():
    r, t = relative_pose_angular_error(_I(), [1, 0, 0], _I(), [1, 0, 0])
    assert r == 0.0 and t == 0.0


def test_angular_error_90deg_z():
    r, _ = relative_pose_angular_error(_I(), [1, 0, 0], _rot("z", 90.0), [1, 0, 0])
    assert abs(r - 90.0) < 1e-6


def test_angular_error_orthogonal_t():
    _, t = relative_pose_angular_error(_I(), [1, 0, 0], _I(), [0, 1, 0])
    assert abs(t - 90.0) < 1e-9


def test_angular_error_antiparallel_t():
    # No sign flip: recoverPose/GT fix cheirality, so -t is a genuine 180° error.
    _, t = relative_pose_angular_error(_I(), [1, 0, 0], _I(), [-1, 0, 0])
    assert abs(t - 180.0) < 1e-9


def test_angular_error_zero_norm_t():
    _, t = relative_pose_angular_error(_I(), [0, 0, 0], _I(), [1, 0, 0])
    assert math.isnan(t)
    _, t = relative_pose_angular_error(_I(), [1, 0, 0], _I(), [0, 0, 0])
    assert math.isnan(t)
