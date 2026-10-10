"""Tests for scene-level evaluation metrics (Phase 2)."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "src"))

from agentic_sfm.agent.policy import ToolCall
from agentic_sfm.eval.scene_eval import (
    _as_cam_from_world,
    _pair_key,
    _qvec_to_rotmat,
    _rot_geodesic_deg,
    _umeyama,
    aggregate_scene_metrics,
    compute_efficiency_frontier,
    compute_scene_metrics,
    scene_eval_score,
    scene_reconstruction_success,
)
from agentic_sfm.rl.scene_episode import SceneRolloutEpisode


def _pose4(R=None, t=None):
    M = np.eye(4)
    if R is not None:
        M[:3, :3] = R
    if t is not None:
        M[:3, 3] = np.asarray(t, dtype=np.float64)
    return M


def _rot_y(deg):
    a = np.radians(deg)
    return np.array(
        [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]
    )


def _gt_poses(n=4):
    """n cam-from-world poses: identity rotations, centers on a 3D arc."""
    centers = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.3, 0.2], [1.8, 1.0, -0.4], [0.6, 1.5, 0.8]]
    )[:n]
    return {str(i): _pose4(t=-centers[i]).tolist() for i in range(len(centers))}


def _sim3_transform_poses(poses: dict[str, list], s: float, Rs: np.ndarray, ts: np.ndarray):
    """Apply world-frame Sim(3) X' = s Rs X + ts to a dict of cam-from-world poses."""
    out = {}
    for k, p in poses.items():
        M = _as_cam_from_world(p)
        R_c, t_c = M[:3, :3], M[:3, 3]
        C = -R_c.T @ t_c
        R_new = R_c @ Rs.T
        C_new = s * Rs @ C + ts
        M_new = np.eye(4)
        M_new[:3, :3] = R_new
        M_new[:3, 3] = -R_new @ C_new
        out[k] = M_new.tolist()
    return out


def _make_episode(**kwargs):
    defaults = dict(
        scene_id="0015",
        image_paths=[f"images/{i}.jpg" for i in range(4)],
        num_images=4,
        done=True,
        recon_result={"num_registered": 4, "num_points3d": 1200},
        tool_calls=[
            ToolCall(tool="retrieve", args={}),
            ToolCall(tool="match", args={"image_a": "img_0000", "image_b": "img_0001"}),
            ToolCall(tool="sfm_run", args={}),
            ToolCall(tool="done", args={}),
        ],
        results=[
            {"pairs": [], "num_pairs": 0},
            {"num_inliers": 100, "inlier_ratio": 0.5},
            {"num_registered": 4, "num_points3d": 1200},
        ],
        reward=1.2,
        reward_components={"total_reward": 1.2, "registration_reward": 0.4},
    )
    defaults.update(kwargs)
    return SceneRolloutEpisode(**defaults)


def test_staged_colmap_image_names_resolve_to_gt_indices():
    from agentic_sfm.eval.scene_eval import _index_lookup, _resolve_image_index

    ep = _make_episode(image_paths=["original/a.jpg", "different/a.jpg", "original/c.png"])
    lookup = _index_lookup(ep)
    assert [_resolve_image_index(name, lookup) for name in ("img_0000.jpg", "img_0001.jpg", "img_0002.png")] == [0, 1, 2]


class TestPoseHelpers:
    def test_as_cam_from_world_formats(self):
        M = _pose4(R=_rot_y(30), t=[1, 2, 3])
        assert np.allclose(_as_cam_from_world(M.tolist()), M)
        assert np.allclose(_as_cam_from_world(M[:3].tolist()), M)
        assert np.allclose(_as_cam_from_world(M.reshape(-1).tolist()), M)
        d = _as_cam_from_world({"R": M[:3, :3].tolist(), "t": [1, 2, 3]})
        assert np.allclose(d, M)
        assert _as_cam_from_world({"bogus": 1}) is None
        assert _as_cam_from_world([1, 2, 3]) is None

    def test_qvec_identity(self):
        assert np.allclose(_qvec_to_rotmat([1, 0, 0, 0]), np.eye(3))

    def test_rot_geodesic(self):
        assert _rot_geodesic_deg(np.eye(3), np.eye(3)) == pytest.approx(0.0)
        assert _rot_geodesic_deg(_rot_y(10), np.eye(3)) == pytest.approx(10.0)

    def test_umeyama_recovers_transform(self):
        rng = np.random.default_rng(0)
        src = rng.normal(size=(10, 3))
        s_true, R_true, t_true = 2.5, _rot_y(20), np.array([1.0, -2.0, 0.5])
        dst = s_true * (src @ R_true.T) + t_true
        s, R, t = _umeyama(src, dst)
        assert s == pytest.approx(s_true, rel=1e-6)
        assert np.allclose(R, R_true, atol=1e-6)
        assert np.allclose(t, t_true, atol=1e-6)


class TestSceneMetrics:
    def test_perfect_reconstruction(self):
        gt = _gt_poses(4)
        # Pred = GT warped by a similarity — alignment should recover ~0 error.
        pred = _sim3_transform_poses(gt, s=1.7, Rs=_rot_y(15), ts=np.array([3, 1, -2]))
        ep = _make_episode(
            recon_result={
                "num_registered": 4,
                "num_points3d": 1200,
                "mean_reproj_error": 0.9,
                "poses": pred,
            }
        )
        scene = {"gt_recon": {"num_images": 4, "poses": gt}}
        m = compute_scene_metrics(ep, scene)

        assert m["registration/registered_fraction"] == pytest.approx(1.0)
        assert m["registration/num_points3d"] == 1200
        assert m["registration/mean_reproj_error_px"] == pytest.approx(0.9)
        assert m["pose/aligned"] == 1.0
        assert m["pose/abs_rot_err_mean_deg"] == pytest.approx(0.0, abs=1e-4)
        assert m["pose/abs_trans_err_mean"] == pytest.approx(0.0, abs=1e-4)
        assert m["pose/rel_rot_err_mean_deg"] == pytest.approx(0.0, abs=1e-4)
        assert m["pose/rel_auc_5"] == pytest.approx(1.0)
        assert m["completeness/coverage"] == pytest.approx(1.0)
        assert m["efficiency/num_tool_calls"] == 4
        assert m["efficiency/calls_match"] == 1
        assert m["reward/total_reward"] == pytest.approx(1.2)

    def test_collinear_centers_rotation_gauge(self):
        """Degenerate center sets: rotation gauge fixed by camera rotations."""
        centers = [np.array([float(i), 0.0, 0.0]) for i in range(4)]
        Rs = [_rot_y(5 * i) for i in range(4)]
        gt = {}
        for i, (R, C) in enumerate(zip(Rs, centers)):
            M = np.eye(4)
            M[:3, :3] = R
            M[:3, 3] = -R @ C  # t = -R C -> camera center at C
            gt[str(i)] = M.tolist()
        pred = _sim3_transform_poses(gt, 1.0, _rot_y(25), np.array([0.5, 0, 0]))
        ep = _make_episode(recon_result={"num_registered": 4, "poses": pred})
        m = compute_scene_metrics(ep, {"gt_recon": {"num_images": 4, "poses": gt}})
        assert m["pose/aligned"] == 1.0
        assert m["pose/abs_rot_err_mean_deg"] == pytest.approx(0.0, abs=1e-3)
        assert m["pose/abs_trans_err_mean"] == pytest.approx(0.0, abs=1e-6)

    def test_noisy_pose_detected(self):
        gt = _gt_poses(4)
        pred = _sim3_transform_poses(gt, s=1.0, Rs=np.eye(3), ts=np.zeros(3))
        # Rotate camera 0 by ~30 deg (pose error survives alignment).
        M = _as_cam_from_world(pred["0"])
        M[:3, :3] = _rot_y(30) @ M[:3, :3]
        pred["0"] = M.tolist()

        ep = _make_episode(recon_result={"num_registered": 4, "poses": pred})
        m = compute_scene_metrics(ep, {"gt_recon": {"num_images": 4, "poses": gt}})
        assert m["pose/rel_rot_err_mean_deg"] > 1.0
        assert m["pose/rel_auc_5"] < 1.0

    def test_failed_reconstruction(self):
        ep = _make_episode(
            recon_result={"error": "colmap failed", "num_registered": 0, "num_points3d": 0},
            reward=-0.5,
        )
        m = compute_scene_metrics(ep, {"gt_recon": {"num_images": 4, "poses": _gt_poses(4)}})
        assert m["registration/num_registered"] == 0
        assert m["registration/success"] == 0.0
        assert m["reconstruction/success"] == 0.0
        assert m["pose/available"] == 0.0
        assert "pose/abs_rot_err_mean_deg" not in m

    def test_no_gt(self):
        ep = _make_episode()
        m = compute_scene_metrics(ep, None)
        assert m["registration/num_registered"] == 4
        assert m["pose/num_gt_poses"] == 0

    def test_pred_poses_from_name_list(self):
        """poses as a parallel list to registered_images (COLMAP names)."""
        gt = _gt_poses(4)
        names = [f"{i}.jpg" for i in range(4)]  # basenames match image_paths
        ep = _make_episode(
            recon_result={
                "num_registered": 4,
                "poses": [gt[str(i)] for i in range(4)],
                "registered_images": names,
            }
        )
        m = compute_scene_metrics(ep, {"gt_recon": {"num_images": 4, "poses": gt}})
        assert m["pose/num_images_with_gt"] == 4
        assert m["pose/rel_auc_10"] == pytest.approx(1.0)

    def test_gt_poses_list_form(self):
        gt = _gt_poses(4)
        ep = _make_episode(
            recon_result={"num_registered": 4, "poses": {k: v for k, v in gt.items()}}
        )
        m = compute_scene_metrics(ep, {"num_images": 4, "poses": [gt[str(i)] for i in range(4)]})
        assert m["pose/num_images_with_gt"] == 4

    def test_doppelganger_metrics(self):
        ep = _make_episode(
            doppelganger_checks={
                "img_0000__img_0001": {"is_doppelganger": True, "confidence": 0.9},
                "img_0002__img_0003": {"is_doppelganger": False, "confidence": 0.1},
            },
            recon_result={
                "num_registered": 4,
                "num_doppelgangers_present": 1,
                "num_doppelgangers_filtered": 1,
            },
        )
        scene = {"gt_recon": {"num_images": 4}, "doppelganger_pairs": [[0, 1]]}
        m = compute_scene_metrics(ep, scene)
        assert m["doppelganger/tp"] == 1
        assert m["doppelganger/tn"] == 1
        assert m["doppelganger/fp"] == 0
        assert m["doppelganger/f1"] == pytest.approx(1.0)
        assert m["doppelganger/detection_rate"] == pytest.approx(1.0)
        assert m["doppelganger/filter_rate"] == pytest.approx(1.0)

    def test_doppelganger_missed(self):
        ep = _make_episode(
            doppelganger_checks={
                "img_0000__img_0001": {"is_doppelganger": False, "confidence": 0.1},
            },
        )
        scene = {"doppelganger_pairs": [[0, 1], ["img_0002", "img_0003"]]}
        m = compute_scene_metrics(ep, scene)
        assert m["doppelganger/fn"] == 1          # checked but not flagged
        assert m["doppelganger/missed_gt"] == 1   # never checked
        assert m["doppelganger/detection_rate"] == pytest.approx(0.0)

    def test_overlap_coverage(self):
        gt = _gt_poses(4)
        om = np.full((4, 4), 0.5)
        np.fill_diagonal(om, 1.0)
        ep = _make_episode(
            recon_result={"num_registered": 4, "poses": gt},
        )
        scene = {
            "gt_recon": {"num_images": 4, "poses": gt},
            "overlap_matrix": om,
            "image_indices": [0, 1, 2, 3],
        }
        m = compute_scene_metrics(ep, scene)
        assert m["completeness/mean_overlap_all"] == pytest.approx(0.5)
        assert m["completeness/mean_overlap_registered"] == pytest.approx(0.5)
        assert m["completeness/index_coverage"] == pytest.approx(1.0)


class TestAggregation:
    def test_aggregate(self):
        eps = []
        for n_reg, n_calls, pose in [(4, 3, 2.0), (2, 6, 5.0), (0, 10, None)]:
            recon = {"num_registered": n_reg, "num_points3d": 100 * n_reg}
            if pose is not None:
                recon["mean_pose_error_deg"] = pose
            ep = _make_episode(
                recon_result=recon,
                tool_calls=[ToolCall(tool="match", args={})] * max(n_calls - 1, 0)
                + [ToolCall(tool="done", args={})],
                results=[{"num_inliers": 10}] * max(n_calls - 1, 0),
            )
            eps.append(compute_scene_metrics(ep, {"num_images": 4}))

        agg = aggregate_scene_metrics(eps)
        assert agg["scene_eval/num_scenes"] == 3
        assert agg["scene_eval/mean_registration_registered_fraction"] == pytest.approx(
            (1.0 + 0.5 + 0.0) / 3
        )
        # 4/4 @ 2° and 2/4 @ 5° pass the pose gate; 0 registered fails.
        assert agg["scene_eval/success_rate"] == pytest.approx(2 / 3)
        assert agg["scene_eval/any_registered_rate"] == pytest.approx(2 / 3)
        assert agg["scene_eval/eval_score"] > 0.0
        frontier = agg["scene_eval/efficiency_frontier"]
        assert frontier[0] == {"num_tool_calls": 3, "registered_fraction": 1.0}
        assert agg["scene_eval/per_scene"][0]["scene_id"] == "0015"

    def test_frontier(self):
        metrics = [
            {"efficiency/num_tool_calls": 5, "registration/registered_fraction": 0.5},
            {"efficiency/num_tool_calls": 3, "registration/registered_fraction": 0.8},
            {"efficiency/num_tool_calls": 8, "registration/registered_fraction": 0.9},
            {"efficiency/num_tool_calls": 10, "registration/registered_fraction": 0.4},
        ]
        frontier = compute_efficiency_frontier(metrics)
        assert frontier == [
            {"num_tool_calls": 3, "registered_fraction": 0.8},
            {"num_tool_calls": 8, "registered_fraction": 0.9},
        ]

    def test_pair_key(self):
        assert _pair_key("b", "a") == "a__b"
        assert _pair_key("img_0001_crop_0_0_10_10", "img_0002") == "img_0001__img_0002"


class TestSceneReconstructionSuccess:
    def test_one_camera_is_not_success(self):
        assert scene_reconstruction_success(1, 10, 2.0) is False

    def test_half_registered_good_pose_is_success(self):
        assert scene_reconstruction_success(5, 10, 5.0) is True

    def test_good_coverage_bad_pose_is_not_success(self):
        assert scene_reconstruction_success(8, 10, 40.0) is False

    def test_missing_pose_is_not_success(self):
        assert scene_reconstruction_success(8, 10, None) is False

    def test_eval_score_zero_without_pose(self):
        assert scene_eval_score(1.0, None) == 0.0

    def test_aggregate_high_coverage_bad_pose_is_not_success(self):
        ep = _make_episode(
            recon_result={
                "num_registered": 8,
                "num_points3d": 800,
                "mean_pose_error_deg": 77.0,
            },
            num_images=10,
            image_paths=[f"images/{i}.jpg" for i in range(10)],
        )
        m = compute_scene_metrics(ep, {"num_images": 10})
        assert m["registration/success"] == 1.0
        assert m["reconstruction/success"] == 0.0
        assert m["reconstruction/eval_score"] == 0.0
        agg = aggregate_scene_metrics([m])
        assert agg["scene_eval/success_rate"] == 0.0
        assert agg["scene_eval/any_registered_rate"] == 1.0
        assert agg["scene_eval/eval_score"] == 0.0

    def test_eval_score_penalizes_pose_error(self):
        perfect = scene_eval_score(1.0, 0.0)
        mid = scene_eval_score(1.0, 10.0)
        assert perfect == pytest.approx(1.0)
        assert mid == pytest.approx(0.5)
        assert scene_eval_score(1.0, 40.0) == 0.0
        assert scene_eval_score(0.5, 0.0) == pytest.approx(0.5)

    def test_success_uses_computed_abs_rot_when_reported_missing(self):
        """G2: direct left mean_pose_error_deg unset; abs rot was already computed."""
        gt = _gt_poses(4)
        ep = _make_episode(
            recon_result={"num_registered": 4, "num_points3d": 400, "poses": gt},
        )
        m = compute_scene_metrics(ep, {"gt_recon": {"num_images": 4, "poses": gt}})
        assert m["pose/reported_mean_error_deg"] is None
        assert m["pose/abs_rot_err_mean_deg"] == pytest.approx(0.0, abs=1e-4)
        assert m["reconstruction/success"] == 1.0
        assert m["reconstruction/eval_score"] == pytest.approx(1.0)


class TestDirectBaselinePoseInjection:
    def test_direct_injects_pose_before_reward(self, monkeypatch):
        from scripts.eval_scene import run_direct_scene_episode

        gt = {"num_images": 4, "poses": _gt_poses(4)}
        scene = {
            "scene_id": "0015",
            "image_paths": [f"images/{i}.jpg" for i in range(4)],
            "gt_recon": gt,
        }

        client = MagicMock()
        client.register_image.return_value = {"status": "ok"}
        client.sfm_run.return_value = {
            "num_registered": 4,
            "num_points3d": 400,
            # No mean_pose_error_deg — the pre-G2 direct bug.
        }

        def _fake_inject(ep, gt_recon):
            assert gt_recon is gt
            ep.recon_result["mean_pose_error_deg"] = 4.0

        monkeypatch.setattr("scripts.eval_scene._inject_pose_error", _fake_inject)
        ep = run_direct_scene_episode(scene, client, output_dir="/tmp/direct_g2")

        assert ep.recon_result["mean_pose_error_deg"] == pytest.approx(4.0)
        assert ep.recon_result["output_dir"] == "/tmp/direct_g2/0015"
        assert ep.reward_components["pose_reward"] > 0.0
        client.sfm_run.assert_called_once()
        assert client.sfm_run.call_args.kwargs["output_dir"] == "/tmp/direct_g2/0015"
