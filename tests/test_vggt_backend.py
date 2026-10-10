"""Tests for the VGGT matcher backend on the tool server.

``server._run_vggt_pair`` is monkeypatched with a synthetic two-view scene
(random 3D points projected into dense point maps + world-to-camera
extrinsics), so the matcher logic — reciprocal NN correspondences, pose
RANSAC, session match-cache, prior fields — runs with no model, no torch
forward, and no weights.
"""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).parent.parent
_spec = importlib.util.spec_from_file_location(
    "tools_server_vggt", ROOT / "tools_server" / "server.py"
)
TS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TS)

H = W = 140
F = 200.0
K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1]])


def _rot(deg_y, deg_x):
    ay, ax = np.radians([deg_y, deg_x])
    ry = np.array([[np.cos(ay), 0, np.sin(ay)],
                   [0, 1, 0],
                   [-np.sin(ay), 0, np.cos(ay)]])
    rx = np.array([[1, 0, 0],
                   [0, np.cos(ax), -np.sin(ax)],
                   [0, np.sin(ax), np.cos(ax)]])
    return ry @ rx


def _project(Xw, ext, K):
    """World points -> pixel coords (u, v); returns coords + valid mask."""
    Xc = Xw @ ext[:, :3].T + ext[:, 3]
    valid = Xc[:, 2] > 1.0
    uv = np.full((len(Xw), 2), -1.0)
    uv[valid] = (Xc[valid] @ K.T)[:, :2] / Xc[valid, 2:3]
    in_frame = (
        valid
        & (uv[:, 0] >= 0) & (uv[:, 0] < W)
        & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    )
    return uv, in_frame


def _build_stub_output():
    """Two-view synthetic scene -> _run_vggt_pair-shaped dict.

    Points are projected into the point maps at UNIQUE pixels per view —
    two world points sharing a projected pixel get dropped entirely, because
    the survivor-take-all overwrite would manufacture structured false
    mutuals (pixel of X_j paired with pixel of X_k) that can swing the
    essential-matrix fit to a wrong branch (observed: t flipped ~145 deg).
    """
    rng = np.random.default_rng(42)
    n = 600
    Xw = np.stack([
        rng.uniform(-1.5, 1.5, n),
        rng.uniform(-1.5, 1.5, n),
        rng.uniform(2.0, 6.0, n),
    ], axis=1)

    R_b = _rot(8, 4)
    t_b = np.array([0.3, 0.1, 0.05])
    ext_a = np.hstack([np.eye(3), np.zeros((3, 1))])
    ext_b = np.hstack([R_b, t_b[:, None]])

    def _unique_pixels(uv, ok):
        """Drop any point whose projected pixel is already claimed."""
        claimed = {}
        keep = ok.copy()
        for i in np.flatnonzero(ok):
            k = (int(uv[i, 1]), int(uv[i, 0]))
            if k in claimed:
                keep[claimed[k]] = False
                keep[i] = False
            else:
                claimed[k] = i
        return keep

    uv_a, ok_a = _project(Xw, ext_a, K)
    uv_b, ok_b = _project(Xw, ext_b, K)
    common = _unique_pixels(uv_a, ok_a) & _unique_pixels(uv_b, ok_b)

    points_a = np.full((H, W, 3), np.nan, dtype=np.float32)
    points_b = np.full((H, W, 3), np.nan, dtype=np.float32)
    # Unmatched pixels get unique far-away junk so they can never pair.
    points_a[:, :, 0] = 4000 + np.arange(W)[None, :]
    points_a[:, :, 1] = 4000 + np.arange(H)[:, None]
    points_a[:, :, 2] = 9000
    points_b[:, :, 0] = 6000 + np.arange(W)[None, :]
    points_b[:, :, 1] = 6000 + np.arange(H)[:, None]
    points_b[:, :, 2] = 9000

    for i in np.flatnonzero(common):
        va, ua = int(uv_a[i, 1]), int(uv_a[i, 0])
        vb, ub = int(uv_b[i, 1]), int(uv_b[i, 0])
        points_a[va, ua] = Xw[i]
        points_b[vb, ub] = Xw[i]

    return {
        "points_a": points_a,
        "points_b": points_b,
        "conf_a": np.full((H, W), 5.0, dtype=np.float32),
        "conf_b": np.full((H, W), 5.0, dtype=np.float32),
        "ext_a": ext_a,
        "ext_b": ext_b,
        # identity mapping: pm frame == original pixels (140x140)
        "map_a": {"scale": 1.0, "scale_y": 1.0, "pad_x": 0.0, "pad_y": 0.0},
        "map_b": {"scale": 1.0, "scale_y": 1.0, "pad_x": 0.0, "pad_y": 0.0},
    }


_STUB_OUT = _build_stub_output()


def _make_image(path, seed):
    rng = np.random.default_rng(seed)
    arr = (rng.random((H, W, 3)) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


@pytest.fixture
def scene(tmp_path, monkeypatch):
    pa = str(tmp_path / "vggt_a.png")
    pb = str(tmp_path / "vggt_b.png")
    _make_image(pa, 1)
    _make_image(pb, 2)
    TS.register_image("vggt_a", pa)
    TS.register_image("vggt_b", pb)

    monkeypatch.setattr(
        TS, "_run_vggt_pair",
        lambda img_a, img_b, max_size: {k: v.copy() for k, v in _STUB_OUT.items()},
    )
    sess = "t1_vggt"
    yield {"pa": pa, "pb": pb, "session": sess}
    TS._match_store.pop(sess, None)


K_LIST = K.tolist()


class TestVggtBackend:
    def test_match_contract(self, scene):
        res = TS.tool_match("vggt_a", "vggt_b", matcher="vggt",
                            K_a=K_LIST, K_b=K_LIST, session=scene["session"])
        assert "error" not in res, res.get("error")
        for key in ("matcher", "num_matches", "num_matches_raw", "num_inliers", "pose",
                    "inlier_ratio", "mean_residual", "residual_units",
                    "keypoints_a", "keypoints_b", "inlier_mask",
                    "prior_pose", "prior_confidence", "pointmap_size"):
            assert key in res, f"missing {key}"
        assert res["matcher"] == "vggt"
        assert res["num_matches"] > 0
        R = np.asarray(res["pose"]["R"])
        t = np.asarray(res["pose"]["t"], dtype=float)
        assert R.shape == (3, 3)
        assert np.isclose(np.linalg.norm(t), 1.0)
        assert res["prior_pose"]["R"]
        assert res["pointmap_size"] == [W, H]
        assert isinstance(res["prior_confidence"], float)

        # GT pose: the stub's relative pose is exactly ext_b (ext_a = I).
        # Catches R transposes, A<->B flips, and t sign errors.
        from agentic_sfm.pose_prior import relative_pose_angular_error
        R_gt = _STUB_OUT["ext_b"][:, :3]
        t_gt = _STUB_OUT["ext_b"][:, 3]
        # prior_pose is pure algebra on extrinsics — noiseless, so 1 deg.
        rot_err, t_err = relative_pose_angular_error(
            R_gt, t_gt,
            np.asarray(res["prior_pose"]["R"]),
            np.asarray(res["prior_pose"]["t"]))
        assert rot_err < 1.0, f"prior_pose rotation error {rot_err} deg"
        assert t_err < 1.0, f"prior_pose translation error {t_err} deg"
        # RANSAC pose sees integer-quantized pm pixels — measured ~0.4 deg
        # on this stub; 2 deg bound still catches transposes/sign flips
        # (those produce >=16 deg rotation / ~180 deg translation).
        rot_err, t_err = relative_pose_angular_error(
            R_gt, t_gt,
            np.asarray(res["pose"]["R"]), np.asarray(res["pose"]["t"]))
        assert rot_err < 2.0, f"pose rotation error {rot_err} deg"
        assert t_err < 2.0, f"pose translation error {t_err} deg"

        # every returned keypoint in-bounds of its original image
        h_a, w_a = H, W
        for x, y in res["keypoints_a"]:
            assert 0 <= x < w_a and 0 <= y < h_a
        for x, y in res["keypoints_b"]:
            assert 0 <= x < w_a and 0 <= y < h_a
        # no NaN anywhere in the response
        def _check(v):
            if isinstance(v, float):
                assert np.isfinite(v)
            elif isinstance(v, dict):
                [_check(x) for x in v.values()]
            elif isinstance(v, list):
                [_check(x) for x in v]
        _check(res)

    def test_full_correspondences_cached(self, scene):
        sess = scene["session"]
        res = TS.tool_match("vggt_a", "vggt_b", matcher="vggt",
                            K_a=K_LIST, K_b=K_LIST, session=sess)
        assert res["num_matches"] > 0
        assert sess in TS._match_store
        bucket = TS._match_store[sess]
        assert len(bucket) == 1
        entry = next(iter(bucket.values()))
        assert entry["matcher"] == "vggt"
        # full set cached, not the [:100] API truncation
        assert len(entry["keypoints_a"]) == res["num_matches"]
        assert len(entry["keypoints_b"]) == res["num_matches"]
        assert len(entry["keypoints_a"]) == len(entry["keypoints_b"])

    def test_determinism(self, scene):
        kw = dict(matcher="vggt", K_a=K_LIST, K_b=K_LIST,
                  session=scene["session"])
        r1 = TS.tool_match("vggt_a", "vggt_b", **kw)
        r2 = TS.tool_match("vggt_a", "vggt_b", **kw)
        assert r1["keypoints_a"] == r2["keypoints_a"]
        assert r1["keypoints_b"] == r2["keypoints_b"]

    def test_fallback_to_loftr(self, scene, monkeypatch):
        def boom(*a, **k):
            raise ImportError("no vggt")
        monkeypatch.setattr(TS, "_run_vggt_pair", boom)
        sentinel = {"matcher": "loftr", "num_matches": 3,
                    "num_inliers": 2, "pose": None}
        monkeypatch.setattr(TS, "_match_loftr", lambda *a, **k: dict(sentinel))
        res = TS.tool_match("vggt_a", "vggt_b", matcher="vggt",
                            K_a=K_LIST, K_b=K_LIST, session=scene["session"])
        assert res["matcher_fallback"] == "vggt->loftr"
        assert res["matcher"] == "loftr"

    def test_unknown_matcher_lists_vggt(self, scene):
        res = TS.tool_match("vggt_a", "vggt_b", matcher="bogus",
                            session=scene["session"])
        assert "error" in res
        assert "vggt" in res["error"]

    def test_fallback_on_runtime_error(self, scene, monkeypatch):
        """Any model failure (not just ImportError) falls back to LoFTR."""
        def boom(*a, **k):
            raise RuntimeError("CUDA OOM")
        monkeypatch.setattr(TS, "_run_vggt_pair", boom)
        sentinel = {"matcher": "loftr", "num_matches": 0,
                    "num_inliers": 0, "pose": None}
        monkeypatch.setattr(TS, "_match_loftr", lambda *a, **k: dict(sentinel))
        res = TS.tool_match("vggt_a", "vggt_b", matcher="vggt",
                            session=scene["session"])
        assert res["matcher_fallback"] == "vggt->loftr"
        assert res["matcher"] == "loftr"

    def test_pad_region_matches_filtered(self, scene, monkeypatch):
        """Pad-bar correspondences (inverted to negative/overflowing original
        coords) must be dropped BEFORE pose RANSAC and the match cache —
        bar-to-bar matches can be mutually consistent."""
        pm = 518
        stub = {k: (v.copy() if hasattr(v, "copy") else dict(v))
                for k, v in _STUB_OUT.items()}
        # shrink pm to 518 canvas: rebuild with pad-bearing maps
        idx = np.arange(pm * pm, dtype=np.float32).reshape(pm, pm)
        stub["points_a"] = np.stack(
            [100000 + idx, idx, np.full((pm, pm), 9000.0)], -1).astype(np.float32)
        stub["points_b"] = np.stack(
            [200000 + idx, idx, np.full((pm, pm), 9000.0)], -1).astype(np.float32)
        stub["conf_a"] = np.zeros((pm, pm), np.float32)
        stub["conf_b"] = np.zeros((pm, pm), np.float32)
        # pair 1: inside content -> in-bounds (y_pm-63 within [0,140));
        # pair 2: inside pad bar (y_pm=5 < pad_y=63 -> y_orig < 0) both sides
        stub["points_a"][100, 100] = [1.0, 2.0, 3.0]
        stub["points_b"][110, 120] = [1.0, 2.0, 3.0]
        stub["points_a"][5, 100] = [4.0, 5.0, 6.0]
        stub["points_b"][6, 120] = [4.0, 5.0, 6.0]
        stub["conf_a"][100, 100] = stub["conf_b"][110, 120] = 6.0
        stub["conf_a"][5, 100] = stub["conf_b"][6, 120] = 6.0
        stub["map_a"] = {"scale": 1.0, "scale_y": 1.0,
                         "pad_x": 0.0, "pad_y": 63.0}
        stub["map_b"] = {"scale": 1.0, "scale_y": 1.0,
                         "pad_x": 0.0, "pad_y": 63.0}
        monkeypatch.setattr(TS, "_run_vggt_pair",
                            lambda *a, **k: stub)
        sess = scene["session"]
        res = TS.tool_match("vggt_a", "vggt_b", matcher="vggt", session=sess)
        assert res["num_matches"] == 1
        assert res["num_matches_raw"] == 2
        # the surviving match is the in-bounds pair (y_pm-63 = 37, 47)
        assert res["keypoints_a"][0][1] == pytest.approx(37.0)
        # cache contains only the filtered correspondence
        entry = next(iter(TS._match_store[sess].values()))
        assert len(entry["keypoints_a"]) == 1

    def test_proportional_scaling_fallback(self, scene, monkeypatch):
        """Stubs without map_* keys fall back to proportional pm->orig
        scaling (pm frame treated as the full image)."""
        # baseline with identity maps (fixture stub): pm frame == orig frame
        r1 = TS.tool_match("vggt_a", "vggt_b", matcher="vggt",
                           session=scene["session"])
        stub = {k: v for k, v in _STUB_OUT.items()}
        stub.pop("map_a"); stub.pop("map_b")
        monkeypatch.setattr(TS, "_run_vggt_pair", lambda *a, **k: stub)
        res = TS.tool_match("vggt_a", "vggt_b", matcher="vggt",
                            session=scene["session"])
        assert res["num_matches"] > 0
        # 140x140 images, 140x140 pm -> proportional scale is exactly 1.0,
        # so the fallback path must produce identical keypoints
        assert res["keypoints_a"] == r1["keypoints_a"]


def _fake_preds(canvas):
    import torch
    return {
        "world_points": torch.zeros(1, 2, canvas, canvas, 3),
        "world_points_conf": torch.ones(1, 2, canvas, canvas),
        "pose_enc": torch.zeros(1, 2, 9),
    }


class TestRunVggtPair:
    """The ONE-forward contract: a single-view forward puts a point map in
    that view's own canonical frame, so mismatched shapes must NEVER trigger
    a second independent forward."""

    def test_one_forward_for_different_shapes(self, tmp_path, monkeypatch):
        calls = []

        class FakeModel:
            def __call__(self, t):
                calls.append(t.shape)
                return _fake_preds(t.shape[-1])

        monkeypatch.setattr(TS, "_get_vggt", lambda: FakeModel())
        monkeypatch.setattr(TS, "_torch_device", lambda: "cpu")

        img_a = np.zeros((240, 320, 3), dtype=np.uint8)
        img_b = np.zeros((480, 640, 3), dtype=np.uint8)
        out = TS._run_vggt_pair(img_a, img_b, 518)

        assert len(calls) == 1, f"expected 1 forward, got {len(calls)}"
        assert tuple(calls[0]) == (2, 3, 518, 518)
        # shared frame: both point maps at the common canvas size
        assert out["points_a"].shape == (518, 518, 3)
        assert out["points_b"].shape == (518, 518, 3)
        assert out["ext_a"].shape == (3, 4)
        assert out["ext_b"].shape == (3, 4)

    def test_letterbox_maps_exact(self, tmp_path, monkeypatch):
        """map_a/map_b must encode the exact inverse to original pixels,
        including non-zero padding on an axis."""
        monkeypatch.setattr(
            TS, "_get_vggt", lambda: (lambda t: _fake_preds(t.shape[-1])))
        monkeypatch.setattr(TS, "_torch_device", lambda: "cpu")

        img_a = np.zeros((240, 320, 3), dtype=np.uint8)  # wider
        img_b = np.zeros((480, 640, 3), dtype=np.uint8)
        out = TS._run_vggt_pair(img_a, img_b, 518)

        ma, mb = out["map_a"], out["map_b"]
        # a: 320x240 -> scale 518/320; new_w=518, new_h=round(240*518/320/14)*14=392
        assert ma["scale"] == pytest.approx(518 / 320)
        assert ma["pad_x"] == 0.0
        assert ma["pad_y"] == pytest.approx((518 - 392) / 2)
        # b: 640x480 -> scale 518/640; new_w=518, new_h=392, pad_y=63
        assert mb["scale"] == pytest.approx(518 / 640)
        assert mb["pad_y"] == pytest.approx((518 - 392) / 2)
        # inverse check: pm pixel -> original pixel
        for m, (w0, h0) in ((ma, (320, 240)), (mb, (640, 480))):
            # center of the letterboxed content maps to image center
            cx_pm = m["pad_x"] + (518 - 2 * m["pad_x"]) / 2
            assert (cx_pm - m["pad_x"]) / m["scale"] == pytest.approx(w0 / 2, abs=0.5)

    def test_tiny_max_size_floors_canvas(self, tmp_path, monkeypatch):
        """max_size < 14 floors the canvas at 14 (never errors, never
        exceeds 518)."""
        monkeypatch.setattr(
            TS, "_get_vggt", lambda: (lambda t: _fake_preds(t.shape[-1])))
        monkeypatch.setattr(TS, "_torch_device", lambda: "cpu")
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        out = TS._run_vggt_pair(img, img, 10)
        assert out["points_a"].shape == (14, 14, 3)
        assert out["points_a"].shape[0] <= 518

    def test_maps_agree_with_pointmap_shapes(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            TS, "_get_vggt", lambda: (lambda t: _fake_preds(t.shape[-1])))
        monkeypatch.setattr(TS, "_torch_device", lambda: "cpu")

        img_a = np.zeros((200, 150, 3), dtype=np.uint8)
        img_b = np.zeros((300, 400, 3), dtype=np.uint8)
        out = TS._run_vggt_pair(img_a, img_b, 518)

        for k in ("points_a", "points_b"):
            assert out[k].shape[0] == out[k].shape[1] == 518
        for mk in ("map_a", "map_b"):
            m = out[mk]
            for key in ("scale", "pad_x", "pad_y"):
                assert key in m and np.isfinite(m[key])
            # letterboxed content fits inside the pm frame
            assert 0 <= m["pad_x"] < 518 and 0 <= m["pad_y"] < 518

    def test_inverse_mapping_end_to_end(self, tmp_path, monkeypatch):
        """A known pm-pixel correspondence must come back at the analytically
        expected original-pixel coords (pad/scale exact inverse)."""
        sess = "t1_map"
        pm = 518
        # two mutual points only: pm_a[200,200] == pm_b[300,300]
        xa_pm, ya_pm = 200.0, 200.0
        xb_pm, yb_pm = 300.0, 300.0
        idx = np.arange(pm * pm, dtype=np.float32).reshape(pm, pm)
        points_a = np.stack([100000 + idx, idx, np.full((pm, pm), 9000.0)],
                            axis=-1).astype(np.float32)
        points_b = np.stack([200000 + idx, idx, np.full((pm, pm), 9000.0)],
                            axis=-1).astype(np.float32)
        shared = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        points_a[int(ya_pm), int(xa_pm)] = shared
        points_b[int(yb_pm), int(xb_pm)] = shared
        # conf 0 junk: only the shared pixel pair survives conf_threshold=1.5
        conf_a = np.zeros((pm, pm), dtype=np.float32)
        conf_b = np.zeros((pm, pm), dtype=np.float32)
        conf_a[int(ya_pm), int(xa_pm)] = 5.0
        conf_b[int(yb_pm), int(xb_pm)] = 5.0

        ma = {"scale": 518 / 320, "scale_y": 392 / 240, "pad_x": 0.0, "pad_y": 63.0}
        mb = {"scale": 518 / 640, "scale_y": 392 / 480, "pad_x": 0.0, "pad_y": 63.0}
        stub = {
            "points_a": points_a, "points_b": points_b,
            "conf_a": conf_a, "conf_b": conf_b,
            "ext_a": np.hstack([np.eye(3), np.zeros((3, 1))]),
            "ext_b": np.hstack([np.eye(3), np.array([[0.1], [0], [0]])]),
            "map_a": ma, "map_b": mb,
        }
        monkeypatch.setattr(
            TS, "_run_vggt_pair",
            lambda *a, **k: {k2: (v.copy() if hasattr(v, "copy") else dict(v))
                            for k2, v in stub.items()})

        pa = str(tmp_path / "ma.png")
        pb = str(tmp_path / "mb.png")
        Image.fromarray(np.zeros((240, 320, 3), np.uint8)).save(pa)
        Image.fromarray(np.zeros((480, 640, 3), np.uint8)).save(pb)
        TS.register_image("map_a", pa)
        TS.register_image("map_b", pb)
        try:
            res = TS.tool_match("map_a", "map_b", matcher="vggt",
                                session=sess)
        finally:
            TS._match_store.pop(sess, None)

        assert res["num_matches"] == 1
        xa, ya = res["keypoints_a"][0]
        xb, yb = res["keypoints_b"][0]
        assert xa == pytest.approx((xa_pm - ma["pad_x"]) / ma["scale"], abs=1e-9)
        assert ya == pytest.approx((ya_pm - ma["pad_y"]) / ma["scale_y"], abs=1e-9)
        assert xb == pytest.approx((xb_pm - mb["pad_x"]) / mb["scale"], abs=1e-9)
        assert yb == pytest.approx((yb_pm - mb["pad_y"]) / mb["scale_y"], abs=1e-9)
