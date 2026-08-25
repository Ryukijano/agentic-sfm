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
)
from agentic_sfm.agent.policy import parse_tool_call, format_observation, ToolCall


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
        reward = compute_pair_reward(match_result, gt_pose, num_tool_calls=2)
        assert reward["total_reward"] > 0
        assert reward["tool_cost"] < 0

    def test_doppelganger_reward(self):
        assert compute_doppelganger_reward(True, True) == 1.0
        assert compute_doppelganger_reward(False, False) == 1.0
        assert compute_doppelganger_reward(True, False) == -1.0


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
