"""Unit tests for mathematical and visual 3D reconstruction losses."""

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "src"))

from agentic_sfm.rewards.pose_rewards import compute_scene_reward
from agentic_sfm.rewards.reconstruction_loss import (
    align_poses_and_points_sim3,
    compute_all_pairs_pose_maa,
    compute_graph_rigidity_metrics,
    compute_intrinsic_ba_metrics,
    compute_multiview_patch_consistency,
    compute_orthographic_3d_render_loss,
    compute_pointcloud_f1,
    compute_projected_depth_loss,
    compute_splat_render_loss,
    huber_loss,
    render_diagnostic_4panel,
    render_pointcloud_splat,
)


def _make_pose(R=None, C=None):
    """Build cam-from-world 4x4 from rotation R and world camera center C."""
    M = np.eye(4, dtype=np.float64)
    if R is not None:
        M[:3, :3] = np.asarray(R, dtype=np.float64)
    if C is not None:
        c = np.asarray(C, dtype=np.float64)
        M[:3, 3] = -M[:3, :3] @ c
    return M


def _rot_z(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array(
        [[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


class TestIntrinsicMathematicalLosses:
    def test_ba_metrics_penalize_2view_only_and_high_reproj(self):
        # 2-view tracks only -> 0 rigidity even with low reprojection error
        m_2view = compute_intrinsic_ba_metrics([0.2, 0.3, 0.4], [2, 2, 2])
        assert m_2view["frac_tracks_ge3"] == 0.0
        assert m_2view["intrinsic_ba_reward"] == 0.0

        # Long tracks + low reprojection error -> high BA reward
        m_good = compute_intrinsic_ba_metrics([0.3, 0.4, 0.5, 0.2], [5, 6, 6, 7])
        assert m_good["frac_tracks_ge3"] == 1.0
        assert m_good["intrinsic_ba_reward"] > 0.7

        # Long tracks + high reprojection error -> low BA reward
        m_noisy = compute_intrinsic_ba_metrics([8.0, 10.0, 12.0, 9.0], [5, 6, 6, 7])
        assert m_noisy["intrinsic_ba_reward"] < m_good["intrinsic_ba_reward"]

    def test_graph_rigidity_fiedler_and_cycle_consistency(self):
        # Disconnected 4-node graph -> Fiedler lambda_2 == 0
        disc = compute_graph_rigidity_metrics(4, [(0, 1, np.eye(3)), (2, 3, np.eye(3))])
        assert disc["fiedler_lambda2"] == pytest.approx(0.0, abs=1e-7)
        assert disc["graph_rigidity_reward"] == pytest.approx(0.0, abs=1e-7)

        # Connected triangle + consistent rotations -> positive lambda_2 & 0 cycle error
        R01 = _rot_z(15)
        R12 = _rot_z(25)
        R20 = (R12 @ R01).T
        conn = compute_graph_rigidity_metrics(
            3, [(0, 1, R01), (1, 2, R12), (2, 0, R20)]
        )
        assert conn["fiedler_lambda2"] > 0.5
        assert conn["num_3cycles"] == 1
        assert conn["mean_cycle_error_deg"] == pytest.approx(0.0, abs=1e-5)
        assert conn["graph_rigidity_reward"] > 0.9

        # Corrupted symmetry edge in the 3-cycle -> large cycle residual
        bad = compute_graph_rigidity_metrics(
            3, [(0, 1, R01), (1, 2, R12), (2, 0, _rot_z(90))]
        )
        assert bad["mean_cycle_error_deg"] > 40.0
        assert bad["graph_rigidity_reward"] < conn["graph_rigidity_reward"]


class TestExtrinsicMathematicalLosses:
    def test_all_pairs_maa_penalizes_abstention(self):
        gt = {
            0: _make_pose(C=[0, 0, 0]),
            1: _make_pose(C=[1, 0, 0]),
            2: _make_pose(C=[2, 1, 0]),
            3: _make_pose(C=[0, 2, 1]),
        }
        # Full registration with exact poses -> mAA = 1.0
        full = compute_all_pairs_pose_maa(gt, gt, num_total_images=4)
        assert full["all_pairs_maa"] == pytest.approx(1.0)

        # Registering only 2 of 4 images (1 pair out of 6) -> mAA = 1/6
        partial = compute_all_pairs_pose_maa({0: gt[0], 1: gt[1]}, gt, num_total_images=4)
        assert partial["all_pairs_maa"] == pytest.approx(1.0 / 6.0)

    def test_sim3_alignment_recovers_points_and_ate(self):
        gt_centers = np.array(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 2.0, 0.0], [0.5, 1.0, 1.5]]
        )
        gt_poses = {i: _make_pose(R=_rot_z(i * 10), C=gt_centers[i]) for i in range(4)}
        gt_pts = np.array([[0.5, 0.5, 3.0], [1.2, -0.3, 4.0], [-0.4, 1.1, 2.5]])

        # Apply an arbitrary Sim(3) gauge to pred poses and points: X_gt = s * Rs @ X_pred + ts
        s_true = 2.5
        R_true = _rot_z(35)
        t_true = np.array([1.0, -2.0, 0.5])
        pred_pts = (R_true.T @ ((gt_pts - t_true) / s_true).T).T
        pred_poses = {}
        for i, M in gt_poses.items():
            R_c = M[:3, :3] @ R_true
            C_p = R_true.T @ ((gt_centers[i] - t_true) / s_true)
            pred_poses[i] = _make_pose(R=R_c, C=C_p)

        info = align_poses_and_points_sim3(pred_poses, gt_poses, points3d=pred_pts)
        assert info["mean_rot_error_deg"] == pytest.approx(0.0, abs=1e-4)
        assert info["mean_ate_m"] == pytest.approx(0.0, abs=1e-4)
        assert info["ate_reward"] == pytest.approx(1.0, abs=1e-4)
        assert np.allclose(info["aligned_points3d"], gt_pts, atol=1e-4)

    def test_pointcloud_f1_and_projected_depth_loss(self):
        rng = np.random.default_rng(7)
        gt_pts = rng.uniform(-1.0, 1.0, size=(200, 3))
        gt_pts[:, 2] += 3.0  # in front of camera at z in [2, 4]

        # Identical cloud -> F1 = 1.0
        f1_exact = compute_pointcloud_f1(gt_pts, gt_pts, tau=0.10)
        assert f1_exact["f1_3d"] == pytest.approx(1.0)
        assert f1_exact["chamfer_l1"] == pytest.approx(0.0, abs=1e-7)

        # Shifted outlier cloud -> F1 drops
        f1_shifted = compute_pointcloud_f1(gt_pts + 2.0, gt_pts, tau=0.10)
        assert f1_shifted["f1_3d"] == 0.0

        # Depth map consistency (dense room depth surface + foreground points)
        K = np.array([[60.0, 0.0, 32.0], [0.0, 60.0, 32.0], [0.0, 0.0, 1.0]])
        pose0 = np.eye(4)
        depth = np.full((64, 64), 4.5, dtype=np.float64)
        for p in gt_pts:
            u = int(round(K[0, 0] * p[0] / p[2] + K[0, 2]))
            v = int(round(K[1, 1] * p[1] / p[2] + K[1, 2]))
            if 0 <= u < 64 and 0 <= v < 64:
                if p[2] < depth[v, u]:
                    depth[v, u] = p[2]

        d_good = compute_projected_depth_loss(gt_pts, {0: pose0}, {0: K}, {0: depth})
        assert d_good["depth_loss"] == pytest.approx(0.0, abs=1e-6)
        assert d_good["depth_consistency_reward"] == pytest.approx(1.0, abs=1e-6)

        # Wall-doubled / depth-corrupted points -> high depth loss
        corrupted = gt_pts.copy()
        corrupted[::2, 2] *= 2.0
        d_bad = compute_projected_depth_loss(corrupted, {0: pose0}, {0: K}, {0: depth})
        assert d_bad["depth_loss"] > 0.15
        assert d_bad["depth_consistency_reward"] < d_good["depth_consistency_reward"]


class TestVisual3DReconstructionLosses:
    def test_orthographic_3d_render_loss_catches_doppelganger_fold(self):
        rng = np.random.default_rng(12)
        # L-shaped room walls in X-Y
        wall_a = np.column_stack([rng.uniform(0, 4, 300), np.zeros(300), rng.uniform(0, 2, 300)])
        wall_b = np.column_stack([np.zeros(300), rng.uniform(0, 4, 300), rng.uniform(0, 2, 300)])
        gt_pts = np.vstack([wall_a, wall_b])
        gt_cols = np.tile(np.array([[0.8, 0.3, 0.2]]), (len(gt_pts), 1))

        # Accurate reconstruction matches BEV & elevations
        good = compute_orthographic_3d_render_loss(
            gt_pts, gt_cols, gt_pts, gt_cols, grid_size=64
        )
        assert good["ortho_bev_iou"] == pytest.approx(1.0)
        assert good["ortho_visual_reward"] == pytest.approx(1.0)

        # Doppelganger symmetry fold: wall_b rotated to negative Y outside the room footprint
        folded_b = np.column_stack([np.zeros(300), -rng.uniform(1, 4, 300), rng.uniform(0, 2, 300)])
        folded_pts = np.vstack([wall_a, folded_b])
        bad = compute_orthographic_3d_render_loss(
            folded_pts, gt_cols, gt_pts, gt_cols, grid_size=64
        )
        assert bad["ortho_bev_iou"] < 0.6
        assert bad["ortho_visual_reward"] < good["ortho_visual_reward"]

    def test_splat_render_and_multiview_patch_consistency(self):
        # Create a textured 3D planar grid in front of two cameras
        xs, ys = np.meshgrid(np.linspace(-0.8, 0.8, 16), np.linspace(-0.8, 0.8, 16))
        pts = np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, 3.0)])
        cols = np.column_stack([
            (xs.ravel() + 1.0) * 0.5,
            (ys.ravel() + 1.0) * 0.5,
            np.full(xs.size, 0.4),
        ])
        K = np.array([[48.0, 0.0, 32.0], [0.0, 48.0, 32.0], [0.0, 0.0, 1.0]])
        pose0 = _make_pose(C=[0.0, 0.0, 0.0])
        pose1 = _make_pose(C=[0.2, 0.0, 0.0])

        r0 = render_pointcloud_splat(pts, cols, K, pose0, width=64, height=64, splat_radius_px=2)
        r1 = render_pointcloud_splat(pts, cols, K, pose1, width=64, height=64, splat_radius_px=2)
        assert r0["opacity"].max() > 0.5

        # Splat loss against the ground-truth rendered views -> high visual reward
        splat_good = compute_splat_render_loss(
            pts, cols, {0: pose0, 1: pose1}, {0: K, 1: K},
            {0: r0["rgb"], 1: r1["rgb"]}, render_size=64, splat_radius_px=2
        )
        assert splat_good["splat_visual_reward"] > 0.85

        # Multi-view patch consistency: true poses agree, wrong pose mismatches
        mv_good = compute_multiview_patch_consistency(
            pts, {0: pose0, 1: pose1}, {0: K, 1: K}, {0: r0["rgb"], 1: r1["rgb"]}
        )
        wrong_pose1 = _make_pose(R=_rot_z(90), C=[0.2, 0.0, 0.0])
        mv_bad = compute_multiview_patch_consistency(
            pts, {0: pose0, 1: wrong_pose1}, {0: K, 1: K}, {0: r0["rgb"], 1: r1["rgb"]}
        )
        assert mv_good["patch_consistency_reward"] > mv_bad["patch_consistency_reward"]

    def test_diagnostic_4panel_renders_cleanly(self):
        import matplotlib.pyplot as plt

        pts = np.array([[0.0, 0.0, 2.0], [1.0, 0.2, 3.0], [-0.5, -0.3, 2.5]])
        cols = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        cams = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.1]])
        errs = np.array([0.4, 1.2, 3.5])
        fig = render_diagnostic_4panel(pts, cols, cam_centers=cams, reproj_errors=errs)
        assert len(fig.axes) == 4
        plt.close(fig)


class TestComputeSceneRewardIntegration:
    def test_scene_reward_includes_math_and_visual_3d_terms(self):
        recon = {
            "num_registered": 18,
            "mean_pose_error_deg": 4.0,
            "intrinsic_ba_reward": 0.8,
            "all_pairs_maa": 0.75,
            "ate_reward": 0.9,
            "f1_3d": 0.7,
            "depth_consistency_reward": 0.85,
            "ortho_visual_reward": 0.8,
            "splat_visual_reward": 0.7,
        }
        gt = {"num_images": 20}
        res = compute_scene_reward(
            recon,
            gt_recon=gt,
            first_call_validity_weight=0.0,
            ba_weight=0.3,
            all_pairs_maa_weight=0.5,
            ate_weight=0.2,
            pointcloud_f1_weight=0.4,
            depth_weight=0.3,
            visual_3d_weight=0.6,
        )
        assert res["intrinsic_ba_reward"] == pytest.approx(0.3 * 0.8)
        assert res["all_pairs_maa_reward"] == pytest.approx(0.5 * 0.75)
        assert res["ate_reward"] == pytest.approx(0.2 * 0.9)
        assert res["pointcloud_f1_reward"] == pytest.approx(0.4 * 0.7)
        assert res["depth_consistency_reward"] == pytest.approx(0.3 * 0.85)
        assert res["visual_3d_reward"] == pytest.approx(0.6 * 0.75)
        expected_extra = (
            0.3 * 0.8 + 0.5 * 0.75 + 0.2 * 0.9 + 0.4 * 0.7 + 0.3 * 0.85 + 0.6 * 0.75
        )
        base_res = compute_scene_reward(recon, gt_recon=gt, first_call_validity_weight=0.0)
        assert res["total_reward"] == pytest.approx(base_res["total_reward"] + expected_extra)
