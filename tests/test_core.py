"""Tests for agentic SfM core modules."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentic_sfm.rewards.pose_rewards import (
    PoseError,
    compute_pose_error,
    compute_pair_reward,
    pose_auc_score,
    compute_doppelganger_reward,
)
from agentic_sfm.data.hard_pairs import (
    difficulty_bin,
    compute_overlap_score_from_recon,
    HardPairDataset,
    ImagePair,
    colmap_K,
    colmap_rotation_translation,
)
from agentic_sfm.agent.policy import parse_tool_call, format_observation, ToolCall, execute_sfm_tool
from agentic_sfm.geometry import (
    camera_matrix,
    crop_image_id,
    crop_pil_from_result,
    estimate_relative_pose,
    iter_oracle_crops,
    k_for_image,
    keep_best_match,
    match_quality,
    ORACLE_CROP_BOXES,
)


class TestRewards:
    def test_pose_error_identity(self):
        R = np.eye(3)
        t = np.array([0, 0, 1])
        pe = compute_pose_error(R, t, R, t)
        assert pe.rotation_error_deg < 0.1
        assert pe.translation_error_deg < 0.1
        assert pe.pose_auc_5 == 1.0
        assert pe.pose_auc_10 == 1.0
        assert pe.pose_auc_20 == 1.0

    def test_pose_error_large_rotation(self):
        R_gt = np.eye(3)
        R_pred = np.array([
            [np.cos(np.radians(30)), -np.sin(np.radians(30)), 0],
            [np.sin(np.radians(30)), np.cos(np.radians(30)), 0],
            [0, 0, 1],
        ])
        t = np.array([0, 0, 1])
        pe = compute_pose_error(R_pred, t, R_gt, t)
        assert 25 < pe.rotation_error_deg < 35
        assert pe.translation_error_deg < 0.1
        assert pe.pose_auc_5 == 0.0
        assert pe.pose_auc_10 == 0.0
        assert pe.pose_auc_20 == 0.0  # 30° > 20° threshold

    def test_pose_auc_score(self):
        pe = PoseError(3, 3, 1, 1, 1)
        assert pose_auc_score(pe) == 1.0

        pe = PoseError(15, 15, 0, 0, 1)
        assert pose_auc_score(pe) == pytest.approx(1/3)

    def test_pair_reward(self):
        match_result = {
            "num_matches": 100,
            "num_inliers": 80,
            "inlier_ratio": 0.8,
            "pose": {"R": np.eye(3).tolist(), "t": [0, 0, 1]},
        }
        gt_pose = {"R": np.eye(3).tolist(), "t": [0, 0, 1]}
        reward = compute_pair_reward(
            match_result, gt_pose, num_tool_calls=2, num_valid_calls=2
        )
        assert reward["total_reward"] > 0
        # Accumulative tool reward: positive when outcome is correct (PyVision-RL)
        assert reward["accumulative_tool_reward"] > 0
        assert reward["tool_cost"] == 0.0  # legacy per-call penalty disabled
        assert reward["pose_reward"] == pytest.approx(1.0)
        # Identity pose: errors ~0 must NOT inflate the total.
        assert reward["total_reward"] < 2.0
        assert abs(reward["total_reward"] - (
            reward["format_reward"] + reward["invalid_penalty"]
            + reward["inlier_reward"] + reward["pose_reward"] + reward["tool_cost"]
            + reward["accumulative_tool_reward"]
        )) < 1e-6

    def test_pair_reward_large_error_not_in_total(self):
        R_bad = np.array([
            [np.cos(np.radians(90)), -np.sin(np.radians(90)), 0],
            [np.sin(np.radians(90)), np.cos(np.radians(90)), 0],
            [0, 0, 1],
        ])
        match_result = {
            "num_inliers": 10,
            "inlier_ratio": 0.2,
            "pose": {"R": R_bad.tolist(), "t": [1, 0, 0]},
        }
        gt_pose = {"R": np.eye(3).tolist(), "t": [0, 0, 1]}
        reward = compute_pair_reward(match_result, gt_pose, num_tool_calls=1, num_valid_calls=1)
        assert reward["rotation_error_deg"] > 80
        assert reward["pose_reward"] == 0.0
        assert reward["total_reward"] < 1.0
        assert reward["total_reward"] > -1.0

    def test_pair_reward_empty_match_still_has_format(self):
        reward = compute_pair_reward(
            {}, gt_pose=None, num_tool_calls=2, num_valid_calls=1, num_invalid_calls=1
        )
        assert reward["pose_reward"] == 0.0
        assert reward["format_reward"] > 0
        assert reward["invalid_penalty"] < 0
        assert abs(reward["total_reward"] - (
            reward["format_reward"] + reward["invalid_penalty"]
            + reward["inlier_reward"] + reward["pose_reward"] + reward["tool_cost"]
            + reward["accumulative_tool_reward"]
        )) < 1e-6

    def test_pair_reward_ignores_unknown_kwargs(self):
        match_result = {"num_inliers": 10, "inlier_ratio": 0.2}
        reward = compute_pair_reward(
            match_result, pose_auc_thresholds=[5, 10, 20], extra_junk=True
        )
        assert "total_reward" in reward

    def test_doppelganger_reward(self):
        assert compute_doppelganger_reward(True, True) == 1.0
        assert compute_doppelganger_reward(False, False) == 1.0
        assert compute_doppelganger_reward(True, False) == -1.0

    def test_ntep_intent_reward_positive(self):
        """Each valid evidence-seeking call earns +ntep_intent_coef."""
        tool_calls = [
            ToolCall(tool="match", args={"image_a": "img_a", "image_b": "img_b"}),
            ToolCall(tool="crop_and_match", args={
                "image_id": "img_a", "bbox": [0.1, 0.1, 0.6, 0.6], "image_b": "img_b",
            }),
            ToolCall(tool="doppelganger_check", args={"image_a": "img_a", "image_b": "img_b"}),
            ToolCall(tool="done", args={}),
        ]
        tool_results = [
            {"num_inliers": 50, "inlier_ratio": 0.5},
            {"num_inliers": 30, "inlier_ratio": 0.4},
            {"is_doppelganger": False, "score": 0.1},
        ]
        reward = compute_pair_reward(
            {}, gt_pose=None, num_tool_calls=4, num_valid_calls=4,
            tool_calls=tool_calls, tool_results=tool_results,
            use_ntep_rewards=True,
        )
        assert reward["ntep_intent_reward"] == pytest.approx(3 * 0.05)
        assert reward["ntep_redundancy_penalty"] == 0.0
        assert reward["ntep_intent_reward"] > 0
        # NTEP components participate in the total when enabled.
        assert reward["total_reward"] == pytest.approx(
            reward["format_reward"] + reward["invalid_penalty"]
            + reward["inlier_reward"] + reward["pose_reward"] + reward["tool_cost"]
            + reward["accumulative_tool_reward"] + reward["ntep_intent_reward"]
            + reward["ntep_redundancy_penalty"]
        )

    def test_ntep_intent_requires_nonerror_result(self):
        """Error results and unmet evidence checks earn no intent reward."""
        tool_calls = [
            ToolCall(tool="match", args={"image_a": "img_a", "image_b": "img_b"}),
            ToolCall(tool="crop_and_match", args={
                "image_id": "img_a", "bbox": [0.1, 0.1, 0.6, 0.6], "image_b": "img_b",
            }),
            ToolCall(tool="doppelganger_check", args={"image_a": "img_a", "image_b": "img_b"}),
        ]
        tool_results = [
            {"error": "boom"},                      # error → no reward
            {"num_inliers": 0, "inlier_ratio": 0.0},  # no inliers → intent not met
            {"num_matches": 80},                    # missing is_doppelganger → not met
        ]
        reward = compute_pair_reward(
            {}, gt_pose=None, num_tool_calls=3, num_valid_calls=3,
            tool_calls=tool_calls, tool_results=tool_results,
            use_ntep_rewards=True,
        )
        assert reward["ntep_intent_reward"] == 0.0

    def test_ntep_redundant_calls_penalized(self):
        """Same tool + same image + IoU > 0.5 → redundant, -penalty each."""
        tool_calls = [
            ToolCall(tool="crop_and_match", args={
                "image_id": "img_a", "bbox": [0.0, 0.0, 0.5, 0.5], "image_b": "img_b",
            }),
            ToolCall(tool="crop_and_match", args={
                "image_id": "img_a", "bbox": [0.05, 0.05, 0.55, 0.55], "image_b": "img_b",
            }),  # IoU ≈ 0.68 > 0.5 → redundant
            ToolCall(tool="crop_and_match", args={
                "image_id": "img_a", "bbox": [0.5, 0.5, 1.0, 1.0], "image_b": "img_b",
            }),  # different region → not redundant
        ]
        tool_results = [
            {"num_inliers": 10}, {"num_inliers": 12}, {"num_inliers": 40},
        ]
        reward = compute_pair_reward(
            {}, gt_pose=None, num_tool_calls=3, num_valid_calls=3,
            tool_calls=tool_calls, tool_results=tool_results,
            use_ntep_rewards=True,
        )
        assert reward["ntep_redundancy_penalty"] == pytest.approx(-0.05)
        # All three still earn intent reward (valid evidence-seeking).
        assert reward["ntep_intent_reward"] == pytest.approx(3 * 0.05)

    def test_ntep_repeated_match_goal_penalized(self):
        """Repeating the same match on the same pair+matcher is redundant."""
        tool_calls = [
            ToolCall(tool="match", args={"image_a": "img_a", "image_b": "img_b", "matcher": "loftr"}),
            ToolCall(tool="match", args={"image_a": "img_b", "image_b": "img_a", "matcher": "loftr"}),  # same pair
            ToolCall(tool="match", args={"image_a": "img_a", "image_b": "img_b", "matcher": "mast3r"}),  # new matcher
        ]
        tool_results = [{"num_inliers": 5}, {"num_inliers": 6}, {"num_inliers": 20}]
        reward = compute_pair_reward(
            {}, gt_pose=None, num_tool_calls=3, num_valid_calls=3,
            tool_calls=tool_calls, tool_results=tool_results,
            use_ntep_rewards=True,
        )
        assert reward["ntep_redundancy_penalty"] == pytest.approx(-0.05)
        assert reward["ntep_intent_reward"] == pytest.approx(3 * 0.05)

    def test_ntep_disabled_by_default(self):
        """NTEP rewards are 0 unless use_ntep_rewards=True (backward compat)."""
        tool_calls = [
            ToolCall(tool="match", args={"image_a": "img_a", "image_b": "img_b"}),
        ]
        tool_results = [{"num_inliers": 50}]
        reward = compute_pair_reward(
            {}, gt_pose=None, num_tool_calls=1, num_valid_calls=1,
            tool_calls=tool_calls, tool_results=tool_results,
        )
        assert reward["ntep_intent_reward"] == 0.0
        assert reward["ntep_redundancy_penalty"] == 0.0

        # Enabled flag without tool_calls → still 0.
        reward = compute_pair_reward(
            {}, gt_pose=None, num_tool_calls=1, num_valid_calls=1,
            use_ntep_rewards=True,
        )
        assert reward["ntep_intent_reward"] == 0.0
        assert reward["ntep_redundancy_penalty"] == 0.0


class TestData:
    def test_difficulty_bin(self):
        assert difficulty_bin(0.9) == "easy"
        assert difficulty_bin(0.5) == "medium"
        assert difficulty_bin(0.2) == "hard"
        assert difficulty_bin(0.05) == "extreme"

    def test_overlap_score(self):
        score = compute_overlap_score_from_recon(50, 100, 100)
        assert score == 0.5

    def test_dataset_save_load(self, tmp_path):
        R = np.eye(3)
        t = np.array([0, 0, 1])
        pair = ImagePair(
            pair_id="test_0",
            image_a="a.png",
            image_b="b.png",
            gt_R=R,
            gt_t=t,
            overlap_score=0.5,
            difficulty="medium",
        )
        ds = HardPairDataset([pair])
        path = str(tmp_path / "test.json")
        ds.save(path)
        loaded = HardPairDataset.load(path)
        assert len(loaded) == 1
        assert loaded[0].pair_id == "test_0"
        assert loaded[0].difficulty == "medium"

    def test_dataset_split(self):
        pairs = []
        for i in range(40):
            overlap = np.random.uniform(0, 1)
            pairs.append(ImagePair(
                pair_id=f"test_{i}",
                image_a="a.png",
                image_b="b.png",
                gt_R=np.eye(3),
                gt_t=np.array([0, 0, 1]),
                overlap_score=overlap,
                difficulty=difficulty_bin(overlap),
            ))
        ds = HardPairDataset(pairs)
        train, val = ds.split(val_ratio=0.2)
        assert len(val) < len(train)


class TestAgent:
    def test_parse_tool_call_crop(self):
        text = 'I will crop the image: {"tool": "crop", "args": {"image_id": "img_a", "bbox": [0.1, 0.2, 0.8, 0.9]}}'
        tc = parse_tool_call(text)
        assert tc is not None
        assert tc.tool == "crop"
        assert tc.args["image_id"] == "img_a"

    def test_parse_tool_call_match(self):
        text = '{"tool": "match", "args": {"image_a": "img_a", "image_b": "img_b", "matcher": "loftr"}}'
        tc = parse_tool_call(text)
        assert tc is not None
        assert tc.tool == "match"
        assert tc.args["matcher"] == "loftr"

    def test_parse_tool_call_crop_and_match(self):
        text = '{"tool": "crop_and_match", "args": {"image_id": "img_a", "bbox": [0.1, 0.2, 0.9, 0.8], "image_b": "img_b"}}'
        tc = parse_tool_call(text)
        assert tc is not None
        assert tc.tool == "crop_and_match"
        assert tc.args["image_id"] == "img_a"

    def test_parse_tool_call_done(self):
        text = '{"tool": "done", "args": {}}'
        tc = parse_tool_call(text)
        assert tc is not None
        assert tc.tool == "done"

    def test_parse_tool_call_invalid(self):
        tc = parse_tool_call("no json here")
        assert tc is None

    def test_format_observation_match(self):
        result = {"num_matches": 100, "num_inliers": 80, "inlier_ratio": 0.8}
        obs = format_observation(result)
        assert "100" in obs
        assert "80" in obs

    def test_format_observation_error(self):
        result = {"error": "model not found"}
        obs = format_observation(result)
        assert "model not found" in obs

    def test_format_observation_crop_id(self):
        obs = format_observation({"crop_id": "img_a_crop_1", "crop_size": [64, 64]})
        assert "img_a_crop_1" in obs
        assert "cropped_image_id" in obs

    def test_execute_unknown_tool(self):
        class Dummy:
            pass

        result = execute_sfm_tool(Dummy(), ToolCall(tool="explode", args={}))
        assert "error" in result


class TestGeometry:
    def test_keep_best_match(self):
        weak = {"num_inliers": 5, "inlier_ratio": 0.1}
        strong = {"num_inliers": 80, "inlier_ratio": 0.8}
        assert keep_best_match(weak, strong) is strong
        assert keep_best_match(strong, weak) is strong
        assert keep_best_match(None, weak) is weak

    def test_crop_image_id(self):
        assert crop_image_id({"cropped_image_id": "a"}) == "a"
        assert crop_image_id({"crop_id": "b"}) == "b"
        assert crop_image_id({}) is None

    def test_match_quality_error(self):
        assert match_quality({"error": "fail"}) < 0

    def test_keep_best_ignores_error_with_fake_inliers(self):
        weak = {"num_inliers": 5, "inlier_ratio": 0.1}
        err = {"error": "fail", "num_inliers": 999}
        assert keep_best_match(weak, err) is weak

    def test_k_for_image_shifts_principal_point(self):
        K = camera_matrix((1000, 800), None)
        adj = np.array(k_for_image(K, [100, 50], (200, 200)))
        assert adj[0, 2] == pytest.approx(K[0, 2] - 100)
        assert adj[1, 2] == pytest.approx(K[1, 2] - 50)
        assert k_for_image(K, None, (200, 200)) is K

    def test_crop_pil_from_b64(self):
        import base64
        from PIL import Image
        import io

        buf = io.BytesIO()
        Image.new("RGB", (8, 8), color=(12, 34, 56)).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        im = crop_pil_from_result({"image_b64": b64})
        assert im is not None
        assert im.size == (8, 8)
        nested = crop_pil_from_result({"crop": {"image_b64": b64}})
        assert nested is not None

    def test_estimate_relative_pose_known_translation(self):
        pytest.importorskip("cv2")
        rng = np.random.default_rng(0)
        K = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
        xyz = rng.uniform([-1.0, -1.0, 4.0], [1.0, 1.0, 8.0], size=(80, 3))
        t = np.array([0.4, 0.05, 0.02])

        def project(X, R, tvec):
            Xc = (R @ X.T).T + tvec
            uv = (K @ Xc.T).T
            return uv[:, :2] / uv[:, 2:3]

        pts_a = project(xyz, np.eye(3), np.zeros(3))
        pts_b = project(xyz, np.eye(3), t)
        pts_a = pts_a + rng.normal(0, 0.15, pts_a.shape)
        pts_b = pts_b + rng.normal(0, 0.15, pts_b.shape)
        out = estimate_relative_pose(pts_a, pts_b, (640, 480), (640, 480), K_a=K, K_b=K)
        assert out["num_inliers"] >= 40
        assert out["pose"] is not None
        pred_t = np.array(out["pose"]["t"]).reshape(3)
        pred_t = pred_t / (np.linalg.norm(pred_t) + 1e-8)
        gt_t = t / np.linalg.norm(t)
        assert abs(float(np.dot(pred_t, gt_t))) > 0.9


class TestColmapHelpers:
    def test_rotation_translation_rigid3d_style(self):
        class Rot:
            def matrix(self):
                return np.eye(3)

        class Cfw:
            rotation = Rot()
            translation = np.array([1.0, 2.0, 3.0])

        class Img:
            cam_from_world = Cfw()

        R, t = colmap_rotation_translation(Img())
        assert np.allclose(R, np.eye(3))
        assert np.allclose(t, [1.0, 2.0, 3.0])

    def test_colmap_k_from_calibration_matrix(self):
        K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])

        class Cam:
            def calibration_matrix(self):
                return K

        class Img:
            camera_id = 1

        class Recon:
            cameras = {1: Cam()}

        got = colmap_K(Recon(), Img())
        assert np.allclose(got, K)

    def test_colmap_k_from_pinhole_params(self):
        class Cam:
            params = np.array([700.0, 710.0, 321.0, 241.0])

        class Img:
            camera_id = 0

        class Recon:
            cameras = {0: Cam()}

        got = colmap_K(Recon(), Img())
        assert got[0, 0] == pytest.approx(700.0)
        assert got[1, 1] == pytest.approx(710.0)
        assert got[0, 2] == pytest.approx(321.0)


class TestOracleCrops:
    def test_boxes_normalized_and_ordered(self):
        assert len(ORACLE_CROP_BOXES) >= 4
        for box in ORACLE_CROP_BOXES:
            x1, y1, x2, y2 = box
            assert 0.0 <= x1 < x2 <= 1.0
            assert 0.0 <= y1 < y2 <= 1.0

    def test_iter_covers_both_images(self):
        jobs = iter_oracle_crops()
        assert len(jobs) == 2 * len(ORACLE_CROP_BOXES)
        sides = {j[0] for j in jobs}
        others = {j[2] for j in jobs}
        assert sides == {"img_a", "img_b"}
        assert others == {"img_a", "img_b"}
        for crop_id, bbox, other_id in jobs:
            assert crop_id != other_id
            assert bbox in ORACLE_CROP_BOXES


class TestPolicyDefaults:
    def test_qwen35_4b_policy(self):
        from agentic_sfm.constants import DEFAULT_LORA_TARGET_MODULES, DEFAULT_POLICY_MODEL

        from agentic_sfm.constants import _version_tuple

        assert DEFAULT_POLICY_MODEL == "Qwen/Qwen3-VL-2B-Instruct"
        assert "q_proj" in DEFAULT_LORA_TARGET_MODULES
        assert "gate_proj" in DEFAULT_LORA_TARGET_MODULES
        assert "in_proj_qkv" not in DEFAULT_LORA_TARGET_MODULES  # no GDN in Qwen3-VL
        assert _version_tuple("4.57.6") < (5, 0)
        assert _version_tuple("0.11.0") < (0, 17)
        assert _version_tuple("5.14.1") >= (5, 0)

    def test_default_matcher_loftr(self):
        from agentic_sfm.constants import DEFAULT_MATCHER

        assert DEFAULT_MATCHER == "loftr"

    def test_execute_match_defaults_to_loftr(self):
        class Dummy:
            def match(self, image_a, image_b, matcher, **kwargs):
                return {"matcher": matcher, "image_a": image_a, "image_b": image_b}

        result = execute_sfm_tool(
            Dummy(),
            ToolCall(tool="match", args={"image_a": "img_a", "image_b": "img_b"}),
        )
        assert result["matcher"] == "loftr"
